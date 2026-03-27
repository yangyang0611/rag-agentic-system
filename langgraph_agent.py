"""LangGraph-based agentic loop with reflection and human-in-the-loop.

State graph:
  rewrite → tool_select → execute_tool → reflect → (loop back or finish)
                                            ↓
                                      human checkpoint (pause)

Reflection node: after each tool execution, the LLM evaluates whether the
gathered information is sufficient, contradictory, or needs a different angle.
This turns a simple tool-executor into a reasoning agent.
"""

import json as _json
from typing import Any, Literal

from langgraph.graph import StateGraph, END
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from tools import AGENT_TOOLS, TOOL_DISPATCH


# ── State schema ──

class AgentState(dict):
    """TypedDict-like state for the agent graph."""
    # We use plain dict for langgraph compatibility

    @staticmethod
    def default(session_id: str, query: str, history: list) -> "AgentState":
        return AgentState(
            session_id=session_id,
            original_query=query,
            messages=list(history),
            all_sources=[],
            tools_used=[],
            round=0,
            max_rounds=5,
            status="running",       # running | paused | done
            pause_data=None,        # data to show user when paused
            reflection="",          # latest reflection text
        )


def build_agent_graph(openai_client: OpenAI, model_name: str = "google/gemini-2.5-flash"):
    """Build and compile the LangGraph agent with reflection."""

    def chat(messages: list, **kwargs):
        return openai_client.chat.completions.create(
            model=model_name, messages=messages, max_tokens=4096, **kwargs
        )

    # ── Node: rewrite query ──
    def rewrite(state: AgentState) -> AgentState:
        history = [m for m in state["messages"] if m.get("role") in ("user", "assistant")]
        response = chat([
            {"role": "system", "content": (
                "You are a query rewriter. Rewrite the user's question into a better search query "
                "optimized for both vector database semantic search and web search. "
                "Keep the original language if it's not English. "
                "If there is conversation history, resolve pronouns and references. "
                "Output ONLY the rewritten query, nothing else."
            )},
            *history,
            {"role": "user", "content": state["original_query"]},
        ])
        rewritten = response.choices[0].message.content.strip()
        print(f"\n{'='*60}")
        print(f"[LangGraph Agent] Original: {state['original_query']}")
        print(f"[LangGraph Agent] Rewritten: {rewritten}")
        print(f"{'='*60}")

        state["messages"].extend([
            {"role": "system", "content": (
                "You are a helpful assistant with access to tools. "
                "ALWAYS search the local database (query_docs) first. "
                "Only use web_search if the database returns no relevant results. "
                "Be direct and concise in your final answer."
            )},
            {"role": "user", "content": f"Original question: {state['original_query']}\n\nOptimized search query: {rewritten}"},
        ])
        return state

    # ── Node: ask LLM to select tool ──
    def tool_select(state: AgentState) -> AgentState:
        round_num = state["round"]
        print(f"[LangGraph Agent] Round {round_num + 1}: Asking LLM to select tool...")

        if round_num >= state["max_rounds"]:
            state["status"] = "max_rounds"
            return state

        try:
            response = chat(state["messages"], tools=AGENT_TOOLS)
        except Exception as e:
            print(f"[LangGraph Agent] Tool call failed ({e}), generating direct answer")
            response = chat(state["messages"])
            state["_final_message"] = response.choices[0].message
            state["status"] = "done"
            return state

        msg = response.choices[0].message

        if not msg.tool_calls:
            print(f"[LangGraph Agent] No tool calls → ready to answer")
            state["_final_message"] = msg
            state["status"] = "done"
            return state

        state["_pending_tool_calls"] = msg
        return state

    # ── Node: execute tools ──
    def execute_tool(state: AgentState) -> AgentState:
        msg = state.pop("_pending_tool_calls")
        round_num = state["round"]
        round_tools = []
        round_sources = []

        state["messages"].append(msg.to_dict())

        for tc in msg.tool_calls:
            args = _json.loads(tc.function.arguments)
            print(f"[LangGraph Agent] Round {round_num + 1}: {tc.function.name}({args})")
            results = TOOL_DISPATCH[tc.function.name](args)
            print(f"[LangGraph Agent] Round {round_num + 1}: → {len(results)} results")

            state["tools_used"].append(tc.function.name)
            state["all_sources"].extend(results)
            round_tools.append({"name": tc.function.name, "args": args})
            round_sources.extend(results)

            state["messages"].append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": _json.dumps(results, ensure_ascii=False),
            })

        state["round"] = round_num + 1
        state["_round_tools"] = round_tools
        state["_round_sources"] = round_sources
        return state

    # ── Node: reflection (the key differentiator) ──
    def reflect(state: AgentState) -> AgentState:
        """LLM evaluates: is the info sufficient? contradictory? need a different angle?"""
        round_num = state["round"]
        print(f"[LangGraph Agent] Round {round_num}: Reflecting...")

        response = chat([
            {"role": "system", "content": (
                "You are a critical evaluator. Given the search results so far, assess:\n"
                "1. Is the information SUFFICIENT to answer the original question?\n"
                "2. Are there any CONTRADICTIONS in the results?\n"
                "3. Should we search from a DIFFERENT ANGLE or with different keywords?\n\n"
                "Respond in this JSON format:\n"
                '{"sufficient": true/false, "reason": "brief explanation", "next_action": "answer" or "search_again", "suggested_query": "new query if searching again"}'
            )},
            {"role": "user", "content": (
                f"Original question: {state['original_query']}\n\n"
                f"Results gathered so far ({len(state['all_sources'])} chunks from {state['round']} rounds):\n"
                + "\n".join(f"- {s.get('content', '')[:200]}" for s in state["all_sources"][-6:])
            )},
        ])

        reflection_text = response.choices[0].message.content.strip()
        state["reflection"] = reflection_text
        print(f"[LangGraph Agent] Reflection: {reflection_text[:200]}")

        # Parse reflection to decide next step
        try:
            reflection = _json.loads(reflection_text)
        except _json.JSONDecodeError:
            reflection = {"sufficient": True, "next_action": "answer", "reason": "could not parse reflection"}

        if reflection.get("next_action") == "answer" or reflection.get("sufficient"):
            state["_reflection_decision"] = "answer"
        else:
            state["_reflection_decision"] = "search_again"
            # Inject the suggested query for the next round
            suggested = reflection.get("suggested_query", "")
            if suggested:
                state["messages"].append({
                    "role": "user",
                    "content": f"The previous results were insufficient. Please search again with a different angle: {suggested}",
                })

        # ── PAUSE for human-in-the-loop ──
        state["status"] = "paused"
        state["pause_data"] = {
            "round": state["round"],
            "tools_called": state.get("_round_tools", []),
            "sources": state.get("_round_sources", []),
            "reflection": reflection_text,
            "reflection_decision": state["_reflection_decision"],
        }
        return state

    # ── Node: generate final answer ──
    def final_answer(state: AgentState) -> AgentState:
        print(f"[LangGraph Agent] Generating final answer...")

        if "_final_message" in state:
            msg = state.pop("_final_message")
        else:
            state["messages"].append({
                "role": "user",
                "content": "Please provide your final answer now based on all the information gathered.",
            })
            response = chat(state["messages"])
            msg = response.choices[0].message

        # Deduplicate sources
        seen = set()
        unique_sources = []
        for s in state["all_sources"]:
            key = s.get("content", "")
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        # Source label
        tu = state["tools_used"]
        if "query_docs" in tu and "web_search" in tu:
            label = "DB + Web"
        elif "query_docs" in tu:
            label = "DB"
        elif "web_search" in tu:
            label = "Web"
        else:
            label = "LLM"

        state["status"] = "done"
        state["result"] = {
            "status": "done",
            "answer": f"*[Source: {label}]*\n\n{msg.content}",
            "sources": unique_sources,
        }
        return state

    # ── Conditional edges ──
    def after_tool_select(state: AgentState) -> str:
        if state["status"] in ("done", "max_rounds"):
            return "final_answer"
        if "_pending_tool_calls" in state:
            return "execute_tool"
        return "final_answer"

    def after_reflect(state: AgentState) -> str:
        # Always pause here — the resume logic will route to next step
        return END  # We pause and return to user; resume continues the graph

    # ── Build graph ──
    graph = StateGraph(dict)

    graph.add_node("rewrite", rewrite)
    graph.add_node("tool_select", tool_select)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("reflect", reflect)
    graph.add_node("final_answer", final_answer)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "tool_select")
    graph.add_conditional_edges("tool_select", after_tool_select, {
        "execute_tool": "execute_tool",
        "final_answer": "final_answer",
    })
    graph.add_edge("execute_tool", "reflect")
    graph.add_edge("reflect", END)           # pause for HITL
    graph.add_edge("final_answer", END)

    compiled = graph.compile()

    # ── Public API ──

    def run_initial(session_id: str, query: str, history: list) -> tuple[AgentState, dict]:
        """Start the agent. Returns (state, response_data)."""
        state = AgentState.default(session_id, query, history)
        result_state = compiled.invoke(state)

        if result_state["status"] == "done":
            return result_state, result_state["result"]
        elif result_state["status"] == "paused":
            return result_state, {"status": "thinking", **result_state["pause_data"]}
        else:
            return result_state, {"status": "error", "message": "Unexpected state"}

    def run_continue(state: AgentState) -> tuple[AgentState, dict]:
        """User pressed Continue — run next round from tool_select."""
        state["status"] = "running"
        decision = state.get("_reflection_decision", "search_again")

        if decision == "answer":
            # Reflection said sufficient, but user wants to continue anyway — generate answer
            result_state = compiled.invoke(state, {"configurable": {"entry_point": "tool_select"}})
        else:
            # Reflection said search again — go to tool_select
            result_state = compiled.invoke(state, {"configurable": {"entry_point": "tool_select"}})

        if result_state["status"] == "done":
            return result_state, result_state["result"]
        elif result_state["status"] == "paused":
            return result_state, {"status": "thinking", **result_state["pause_data"]}
        else:
            return result_state, {"status": "error", "message": "Unexpected state"}

    def run_stop(state: AgentState) -> tuple[AgentState, dict]:
        """User pressed Stop — force final answer."""
        state["status"] = "running"
        # Run only the final_answer node
        result_state = final_answer(state)
        return result_state, result_state["result"]

    return run_initial, run_continue, run_stop