"""Structure-aware document ingestion — vectorless retrieval path.

Parses documents into a hierarchical tree (DocNode) based on headings,
preserving document structure for precise section-level retrieval.

Pipeline:
  Document → structural parser → DocNode tree → JSON index (./struct_index/)
  Query    → LLM picks sections from TOC → retrieve exact section content
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, asdict
from statistics import mode

import fitz  # pymupdf
import requests
from bs4 import BeautifulSoup

INDEX_DIR = "./struct_index"
os.makedirs(INDEX_DIR, exist_ok=True)

# In-memory cache: source_id -> DocNode dict
_index_cache: dict[str, dict] = {}


@dataclass
class DocNode:
    id: str
    type: str              # "document" | "heading" | "paragraph"
    level: int             # 0=root, 1=h1, 2=h2, ...
    title: str
    content: str           # full text (leaf nodes only)
    source: str            # original URL or filename
    path: str              # breadcrumb: "Experience > NVIDIA"
    page: int | None = None
    children: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["children"] = [c.to_dict() if isinstance(c, DocNode) else c for c in self.children]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DocNode":
        d["children"] = [cls.from_dict(c) for c in d.get("children", [])]
        return cls(**d)

    def get_toc(self, max_depth: int = 3) -> list[dict]:
        """Extract table of contents (heading titles + paths only)."""
        toc = []
        if self.type == "heading" and self.level <= max_depth:
            toc.append({"title": self.title, "path": self.path, "level": self.level})
        for child in self.children:
            if isinstance(child, DocNode):
                toc.extend(child.get_toc(max_depth))
        return toc

    def get_section_content(self) -> str:
        """Recursively collect all text content under this node."""
        parts = []
        if self.content:
            parts.append(self.content)
        for child in self.children:
            if isinstance(child, DocNode):
                parts.append(child.get_section_content())
        return "\n\n".join(parts)

    def find_by_path(self, target_path: str) -> "DocNode | None":
        """Find a node by its breadcrumb path."""
        if self.path == target_path:
            return self
        for child in self.children:
            if isinstance(child, DocNode):
                result = child.find_by_path(target_path)
                if result:
                    return result
        return None


# ── PDF Structural Parsing ──

def _detect_body_font_size(doc: fitz.Document) -> float:
    """Find the most common font size in the document (= body text size)."""
    sizes = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:  # text blocks only
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if len(text) >= 10:  # skip short fragments
                        sizes.append(round(span["size"], 1))
    if not sizes:
        return 12.0
    try:
        return mode(sizes)
    except Exception:
        return sorted(sizes)[len(sizes) // 2]


# Regex for numbered headings: "1.", "1.1", "1.1.1", "Chapter 1", "Section 2.3"
_NUMBERED_HEADING_RE = re.compile(
    r"^(?:(?:Chapter|Section|Part)\s+)?\d+(?:\.\d+)*\.?\s+\S",
    re.IGNORECASE,
)


def _is_heading(span_size: float, body_size: float, text: str, is_bold: bool) -> int | None:
    """Determine if a text span is a heading. Returns heading level or None.

    Uses multiple heuristics:
    1. Font size relative to body text (primary)
    2. ALL CAPS detection (fallback)
    3. Numbered heading patterns like "1.1 Introduction" (fallback)
    """
    stripped = text.strip()
    if len(stripped) < 2 or len(stripped) > 200:
        return None

    size_ratio = span_size / body_size if body_size > 0 else 1.0

    # Primary: font size based detection
    if size_ratio >= 1.6:
        return 1
    elif size_ratio >= 1.3:
        return 2
    elif size_ratio >= 1.1 or (is_bold and size_ratio >= 1.0 and len(stripped) < 80):
        return 3

    # Fallback: ALL CAPS lines (short, no lowercase) → likely a heading
    if (stripped.isupper() and len(stripped) < 60
            and len(stripped.split()) <= 8 and not stripped.startswith("-")):
        return 1

    # Fallback: numbered headings like "1.1 Introduction"
    if _NUMBERED_HEADING_RE.match(stripped) and len(stripped) < 80:
        # Count dots to estimate depth: "1" → level 1, "1.1" → level 2
        num_part = stripped.split()[0].rstrip(".")
        depth = num_part.count(".") + 1
        return min(depth, 3)

    return None


def parse_pdf_structure(path: str, source: str = "") -> DocNode:
    """Parse a PDF into a hierarchical DocNode tree based on font size."""
    source = source or os.path.basename(path)
    doc = fitz.open(path)
    body_size = _detect_body_font_size(doc)

    root = DocNode(
        id=hashlib.md5(source.encode()).hexdigest(),
        type="document",
        level=0,
        title=source,
        content="",
        source=source,
        path=source,
    )

    # Stack tracks the current heading hierarchy: [(level, DocNode)]
    heading_stack: list[tuple[int, DocNode]] = [(0, root)]
    current_paragraphs: list[str] = []

    def flush_paragraph():
        """Attach accumulated paragraph text to the current heading node."""
        nonlocal current_paragraphs
        if not current_paragraphs:
            return
        text = "\n".join(current_paragraphs)
        parent = heading_stack[-1][1]
        para_id = hashlib.md5(f"{source}_{parent.path}_{len(parent.children)}".encode()).hexdigest()
        parent.children.append(DocNode(
            id=para_id,
            type="paragraph",
            level=parent.level + 1,
            title=text[:80] + ("..." if len(text) > 80 else ""),
            content=text,
            source=source,
            path=parent.path,
            page=None,
        ))
        current_paragraphs = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                line_text = "".join(span["text"] for span in line["spans"]).strip()
                if not line_text:
                    continue

                # Check if this line is a heading (use the largest span's size)
                max_span = max(line["spans"], key=lambda s: s["size"])
                is_bold = "bold" in max_span["font"].lower()
                heading_level = _is_heading(max_span["size"], body_size, line_text, is_bold)

                if heading_level is not None:
                    # Flush any accumulated paragraph text
                    flush_paragraph()

                    # Pop stack back to parent level
                    while heading_stack and heading_stack[-1][0] >= heading_level:
                        heading_stack.pop()

                    parent = heading_stack[-1][1] if heading_stack else root
                    path = f"{parent.path} > {line_text}" if parent.path != source else line_text

                    heading_node = DocNode(
                        id=hashlib.md5(f"{source}_{path}".encode()).hexdigest(),
                        type="heading",
                        level=heading_level,
                        title=line_text,
                        content="",
                        source=source,
                        path=path,
                        page=page_num + 1,
                    )
                    parent.children.append(heading_node)
                    heading_stack.append((heading_level, heading_node))
                else:
                    # Regular text — accumulate as paragraph
                    if len(line_text) >= 5:
                        current_paragraphs.append(line_text)

    # Flush remaining paragraphs
    flush_paragraph()
    doc.close()

    return root


# ── HTML Structural Parsing ──

def parse_html_structure(url: str, html: str = "") -> DocNode:
    """Parse HTML into a hierarchical DocNode tree based on heading tags."""
    if not html:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        html = resp.text

    soup = BeautifulSoup(html, "html.parser")

    # Remove script, style, nav, footer
    for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    root = DocNode(
        id=hashlib.md5(url.encode()).hexdigest(),
        type="document",
        level=0,
        title=soup.title.string.strip() if soup.title and soup.title.string else url,
        content="",
        source=url,
        path=soup.title.string.strip() if soup.title and soup.title.string else url,
    )

    heading_stack: list[tuple[int, DocNode]] = [(0, root)]
    heading_tags = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

    body = soup.body or soup
    for element in body.children:
        _process_html_element(element, heading_tags, heading_stack, root, url)

    return root


def _process_html_element(element, heading_tags, heading_stack, root, source):
    """Recursively process an HTML element into the DocNode tree."""
    if isinstance(element, str):
        text = element.strip()
        if len(text) >= 5:
            parent = heading_stack[-1][1]
            para_id = hashlib.md5(f"{source}_{parent.path}_{len(parent.children)}".encode()).hexdigest()
            parent.children.append(DocNode(
                id=para_id,
                type="paragraph",
                level=parent.level + 1,
                title=text[:80] + ("..." if len(text) > 80 else ""),
                content=text,
                source=source,
                path=parent.path,
            ))
        return

    tag_name = getattr(element, "name", None)
    if not tag_name:
        return

    if tag_name in heading_tags:
        heading_level = heading_tags[tag_name]
        heading_text = element.get_text(strip=True)
        if not heading_text:
            return

        while heading_stack and heading_stack[-1][0] >= heading_level:
            heading_stack.pop()

        parent = heading_stack[-1][1] if heading_stack else root
        path = f"{parent.path} > {heading_text}" if parent.level > 0 else heading_text

        heading_node = DocNode(
            id=hashlib.md5(f"{source}_{path}".encode()).hexdigest(),
            type="heading",
            level=heading_level,
            title=heading_text,
            content="",
            source=source,
            path=path,
        )
        parent.children.append(heading_node)
        heading_stack.append((heading_level, heading_node))
    elif tag_name in ("p", "li", "td", "blockquote", "pre", "code"):
        text = element.get_text(strip=True)
        if len(text) >= 5:
            parent = heading_stack[-1][1]
            para_id = hashlib.md5(f"{source}_{parent.path}_{len(parent.children)}".encode()).hexdigest()
            parent.children.append(DocNode(
                id=para_id,
                type="paragraph",
                level=parent.level + 1,
                title=text[:80] + ("..." if len(text) > 80 else ""),
                content=text,
                source=source,
                path=parent.path,
            ))
    else:
        # Recurse into container elements (div, section, article, etc.)
        for child in element.children:
            _process_html_element(child, heading_tags, heading_stack, root, source)


# ── Markdown Structural Parsing ──

def parse_markdown_structure(text: str, source: str = "") -> DocNode:
    """Parse Markdown into a hierarchical DocNode tree based on # headings."""
    source = source or "markdown"

    root = DocNode(
        id=hashlib.md5(source.encode()).hexdigest(),
        type="document",
        level=0,
        title=source,
        content="",
        source=source,
        path=source,
    )

    heading_stack: list[tuple[int, DocNode]] = [(0, root)]
    current_paragraphs: list[str] = []

    def flush_paragraph():
        nonlocal current_paragraphs
        if not current_paragraphs:
            return
        text_block = "\n".join(current_paragraphs)
        parent = heading_stack[-1][1]
        para_id = hashlib.md5(f"{source}_{parent.path}_{len(parent.children)}".encode()).hexdigest()
        parent.children.append(DocNode(
            id=para_id,
            type="paragraph",
            level=parent.level + 1,
            title=text_block[:80] + ("..." if len(text_block) > 80 else ""),
            content=text_block,
            source=source,
            path=parent.path,
        ))
        current_paragraphs = []

    heading_re = re.compile(r"^(#{1,6})\s+(.+)$")

    for line in text.splitlines():
        match = heading_re.match(line)
        if match:
            flush_paragraph()
            heading_level = len(match.group(1))  # number of '#' characters
            heading_text = match.group(2).strip()

            while heading_stack and heading_stack[-1][0] >= heading_level:
                heading_stack.pop()

            parent = heading_stack[-1][1] if heading_stack else root
            path = f"{parent.path} > {heading_text}" if parent.path != source else heading_text

            heading_node = DocNode(
                id=hashlib.md5(f"{source}_{path}".encode()).hexdigest(),
                type="heading",
                level=heading_level,
                title=heading_text,
                content="",
                source=source,
                path=path,
            )
            parent.children.append(heading_node)
            heading_stack.append((heading_level, heading_node))
        else:
            stripped = line.strip()
            if stripped:
                current_paragraphs.append(stripped)
            elif current_paragraphs:
                # Empty line = paragraph break
                flush_paragraph()

    flush_paragraph()
    return root


# ── Word (.docx) Structural Parsing ──

def parse_docx_structure(path: str, source: str = "") -> DocNode:
    """Parse a Word document into a hierarchical DocNode tree based on heading styles."""
    from docx import Document as DocxDocument

    source = source or os.path.basename(path)
    doc = DocxDocument(path)

    root = DocNode(
        id=hashlib.md5(source.encode()).hexdigest(),
        type="document",
        level=0,
        title=source,
        content="",
        source=source,
        path=source,
    )

    heading_stack: list[tuple[int, DocNode]] = [(0, root)]
    current_paragraphs: list[str] = []

    def flush_paragraph():
        nonlocal current_paragraphs
        if not current_paragraphs:
            return
        text = "\n".join(current_paragraphs)
        parent = heading_stack[-1][1]
        para_id = hashlib.md5(f"{source}_{parent.path}_{len(parent.children)}".encode()).hexdigest()
        parent.children.append(DocNode(
            id=para_id,
            type="paragraph",
            level=parent.level + 1,
            title=text[:80] + ("..." if len(text) > 80 else ""),
            content=text,
            source=source,
            path=parent.path,
        ))
        current_paragraphs = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            if current_paragraphs:
                flush_paragraph()
            continue

        # Word heading styles: "Heading 1", "Heading 2", etc.
        style_name = para.style.name if para.style else ""
        heading_level = None
        if style_name.startswith("Heading"):
            try:
                heading_level = int(style_name.split()[-1])
            except (ValueError, IndexError):
                pass
        # Also detect "Title" style as level 1
        if style_name == "Title":
            heading_level = 1

        if heading_level is not None:
            flush_paragraph()

            while heading_stack and heading_stack[-1][0] >= heading_level:
                heading_stack.pop()

            parent = heading_stack[-1][1] if heading_stack else root
            path = f"{parent.path} > {text}" if parent.path != source else text

            heading_node = DocNode(
                id=hashlib.md5(f"{source}_{path}".encode()).hexdigest(),
                type="heading",
                level=heading_level,
                title=text,
                content="",
                source=source,
                path=path,
            )
            parent.children.append(heading_node)
            heading_stack.append((heading_level, heading_node))
        else:
            if len(text) >= 3:
                current_paragraphs.append(text)

    flush_paragraph()
    return root


# ── Persistence ──

def _source_id(source: str) -> str:
    return hashlib.md5(source.encode()).hexdigest()


def save_index(node: DocNode):
    """Save a DocNode tree to disk and update the in-memory cache."""
    sid = _source_id(node.source)
    filepath = os.path.join(INDEX_DIR, f"{sid}.struct.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(node.to_dict(), f, ensure_ascii=False, indent=2)
    _index_cache[sid] = node


def load_all_indexes() -> dict[str, DocNode]:
    """Load all structured indexes from disk into memory."""
    global _index_cache
    _index_cache = {}
    if not os.path.exists(INDEX_DIR):
        return _index_cache
    for filename in os.listdir(INDEX_DIR):
        if filename.endswith(".struct.json"):
            filepath = os.path.join(INDEX_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            node = DocNode.from_dict(data)
            sid = filename.replace(".struct.json", "")
            _index_cache[sid] = node
    return _index_cache


def get_all_toc(max_depth: int = 3) -> list[dict]:
    """Get combined TOC from all indexed documents."""
    toc = []
    for node in _index_cache.values():
        toc.extend(node.get_toc(max_depth))
    return toc


def find_section(path: str) -> DocNode | None:
    """Find a section by its breadcrumb path across all indexed documents."""
    for node in _index_cache.values():
        result = node.find_by_path(path)
        if result:
            return result
    return None


# ── Public Ingestion API ──

def ingest_pdf_structured(path: str, source: str = "") -> dict:
    """Parse PDF into structural tree and save to index."""
    source = source or os.path.basename(path)
    root = parse_pdf_structure(path, source)
    save_index(root)
    toc = root.get_toc()
    return {
        "source": source,
        "sections_indexed": len(toc),
        "toc": [t["title"] for t in toc],
    }


def ingest_url_structured(url: str) -> dict:
    """Fetch webpage HTML and parse into structural tree."""
    root = parse_html_structure(url)
    save_index(root)
    toc = root.get_toc()
    return {
        "source": url,
        "sections_indexed": len(toc),
        "toc": [t["title"] for t in toc],
    }


def ingest_markdown_structured(path: str, source: str = "") -> dict:
    """Parse a Markdown file into structural tree and save to index."""
    source = source or os.path.basename(path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    root = parse_markdown_structure(text, source)
    save_index(root)
    toc = root.get_toc()
    return {
        "source": source,
        "sections_indexed": len(toc),
        "toc": [t["title"] for t in toc],
    }


def ingest_docx_structured(path: str, source: str = "") -> dict:
    """Parse a Word document into structural tree and save to index."""
    source = source or os.path.basename(path)
    root = parse_docx_structure(path, source)
    save_index(root)
    toc = root.get_toc()
    return {
        "source": source,
        "sections_indexed": len(toc),
        "toc": [t["title"] for t in toc],
    }


def ingest_file_structured(path: str, source: str = "") -> dict:
    """Auto-detect file type and ingest with structural parsing."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return ingest_pdf_structured(path, source)
    elif ext in (".md", ".markdown"):
        return ingest_markdown_structured(path, source)
    elif ext in (".docx",):
        return ingest_docx_structured(path, source)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .md, .docx")


# Load existing indexes on import
load_all_indexes()
