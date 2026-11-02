"""
Default-safe publishing policy for externally authored Jupyter notebooks.

External notebooks are untrusted input. A notebook can carry active content in
three channels:

1. Raw HTML embedded in Markdown cells (e.g. ``<script>``, ``onerror`` handlers).
2. Rich output MIME bundles, where nbconvert prefers ``text/html`` and
   ``image/svg+xml`` over the static ``image/png`` / ``text/plain`` alternatives.
3. Resource files written into the page bundle (extracted SVG outputs and
   attachments), where an SVG may itself contain scripts or event handlers.

The policy applied here is fail-closed by default:

* dangerous markup is stripped from Markdown/HTML text,
* dangerous rich outputs fall back to a static raster image or plain text,
* dangerous SVG resources are neutralised or blocked,
* executable MIME types (JavaScript, widgets) are dropped,
* code examples are always preserved verbatim (code cell sources, fenced code
  blocks and inline code spans are never rewritten).

Pass ``trusted=True`` (CLI: ``--trusted``) to opt out and keep the original rich
media when the notebook author is trusted.
"""

import base64
import binascii
import html
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from nbconvert.preprocessors import Preprocessor
from traitlets import Bool

log = logging.getLogger(__name__)

# Static raster images, in preference order for fallback when a rich output is unsafe.
STATIC_IMAGE_MIMES = ["image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp", "image/tiff"]

# Textual MIME types that are safe to render when the rich representation is dropped.
SAFE_TEXT_MIMES = ["text/plain", "text/latex", "text/markdown"]

# Executable / interactive MIME types that never have a safe Markdown rendering.
ACTIVE_MIMES = {
    "application/javascript",
    "text/javascript",
    "application/x-javascript",
    "application/ecmascript",
    "text/ecmascript",
    "application/vnd.jupyter.widget-view+json",
    "application/vnd.jupyter.widget-state+json",
    "application/x-shockwave-flash",
    "application/java",
    "application/x-java-applet",
}

# Resource extensions that must never be written into a page bundle in untrusted mode.
BLOCKED_RESOURCE_EXTENSIONS = {".html", ".htm", ".xhtml", ".shtml", ".js", ".mjs", ".swf"}

# Elements whose opening tag, closing tag and everything in between is removed.
_DROP_WITH_CONTENT_TAGS = (
    "script",
    "iframe",
    "object",
    "embed",
    "applet",
    "frame",
    "frameset",
    "noembed",
    "noframes",
    "foreignobject",
    "title",
)

# Elements that carry no content payload; the tag alone is removed.
_DROP_VOID_TAGS = ("meta", "link", "base", "param", "isindex", "command")

# Attributes whose value may navigate to or embed an active resource.
_URL_ATTRIBUTES = {
    "href",
    "src",
    "xlink:href",
    "poster",
    "cite",
    "action",
    "formaction",
    "background",
    "longdesc",
    "lowsrc",
    "ping",
    "manifest",
}

# A tiny valid SVG used when an unsafe SVG cannot be sanitised but is still referenced.
_BLOCKED_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="48">'
    '<text x="8" y="28" font-family="sans-serif" font-size="14">[SVG removed for safety]</text>'
    "</svg>"
)

_COMMENT_RE = re.compile(r"<!--[\s\S]*?-->")
_CDATA_RE = re.compile(r"<!\[CDATA\[[\s\S]*?\]\]>")
_PROCESSING_INSTRUCTION_RE = re.compile(r"<\?[\s\S]*?\?>")
_DECLARATION_RE = re.compile(r"<!\s*[a-zA-Z][^>]*>")
_START_TAG_RE = re.compile(r"<(/?)([a-zA-Z][\w:.-]*)((?:[\s/]+[^<>]*?)?)\s*(/?)>", re.DOTALL)
_CODE_SPAN_RE = re.compile(r"(`+)([^`]*?)\1")
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")
_INDENTED_CODE_RE = re.compile(r"^(?: {4}|\t)")

# Markdown link/image: `[text](target "optional title")`, target may be wrapped in angle brackets.
# Targets allow one level of nested parentheses, which real URLs (e.g. Wikipedia) use.
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\(\s*(?:<([^>]*)>|((?:[^()\s]|\([^()]*\))*))(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)")
# Markdown reference definition: `[id]: target`.
_MD_REFERENCE_DEF_RE = re.compile(r"^(\s*\[[^\]]+\]:\s*)<?([^\s>]+?)>?\s*(?:\"[^\"]*\"|'[^']*')?$", re.MULTILINE)
# Markdown autolink: `<scheme:url>`.
_MD_AUTOLINK_RE = re.compile(r"<((?:javascript|vbscript|livescript):[^<>\s]*?)>", re.IGNORECASE)

# Signals that an SVG carries active content. Kept tag-scoped to avoid matching prose.
_SVG_ACTIVE_PATTERNS = (
    re.compile(r"<\s*script\b", re.IGNORECASE),
    re.compile(r"<\s*foreignobject\b", re.IGNORECASE),
    re.compile(r"<!\s*(?:doctype|entity|element|attlist)", re.IGNORECASE),
    re.compile(r"<\?", re.DOTALL),
    re.compile(r"<[^<>]*\son[a-z]+\s*=", re.IGNORECASE | re.DOTALL),
    re.compile(
        r"<[^<>]*(?:href|xlink:href)\s*=\s*[\"']?\s*(?:javascript|vbscript|data:text/html|data:image/svg)",
        re.IGNORECASE,
    ),
)


def _to_text(value: Any) -> str:
    """Normalise an nbformat string-or-list-of-lines field to text."""
    if isinstance(value, (list, tuple)):
        return "".join(str(part) for part in value)
    return str(value)


def _normalise_url(value: str) -> str:
    """Decode entities and strip whitespace/control characters for scheme checks."""
    return re.sub(r"[\s\x00-\x1f]+", "", html.unescape(value)).lower()


def _url_is_unsafe(value: str) -> bool:
    url = _normalise_url(value)
    return url.startswith(("javascript:", "vbscript:", "livescript:")) or url.startswith("data:text/html") or url.startswith("data:image/svg")


def _iter_attributes(attr_text: str):
    """
    Tokenise tag attributes the way an HTML parser would.

    ``/`` is treated as an attribute separator (HTML5 self-closing flag), which
    matters for vectors such as ``<img/onerror=...>``. Quoted values are read
    verbatim; unquoted values terminate at whitespace or ``/``.
    """
    i, length = 0, len(attr_text)
    while i < length:
        while i < length and attr_text[i] in " \t\n\r\f/":
            i += 1
        if i >= length:
            break
        start = i
        while i < length and attr_text[i] not in " \t\n\r\f=/>":
            i += 1
        name = attr_text[start:i]
        value = None
        while i < length and attr_text[i] in " \t\n\r\f/":
            i += 1
        if i < length and attr_text[i] == "=":
            i += 1
            while i < length and attr_text[i] in " \t\n\r\f":
                i += 1
            if i < length and attr_text[i] in "\"'":
                quote = attr_text[i]
                i += 1
                value_start = i
                while i < length and attr_text[i] != quote:
                    i += 1
                value = attr_text[value_start:i]
                if i < length:
                    i += 1
            else:
                value_start = i
                while i < length and attr_text[i] not in " \t\n\r\f/>":
                    i += 1
                value = attr_text[value_start:i]
        yield name, value


def _scrub_tag(match: "re.Match[str]") -> Tuple[str, bool]:
    """Rebuild one HTML/SVG tag without event handlers and unsafe URLs."""
    slash, tag_name, attr_text, self_closing = match.groups()
    # Determine self-closing from the raw tag, since the attribute zone may end in ``/``.
    is_self_closing = bool(self_closing) or match.group(0).rstrip().endswith("/>")
    rebuilt_attrs: List[str] = []
    changed = False
    for name, raw_value in _iter_attributes(attr_text):
        lowered_name = name.lower()
        if lowered_name.startswith("on") and len(lowered_name) > 2:
            changed = True
            continue
        if raw_value is not None:
            if lowered_name in _URL_ATTRIBUTES and _url_is_unsafe(raw_value):
                changed = True
                continue
            if lowered_name == "style":
                style = _normalise_url(raw_value)
                if "javascript:" in style or "vbscript:" in style or "expression(" in style:
                    changed = True
                    continue
            escaped_value = raw_value.replace('"', "&quot;")
            rebuilt_attrs.append(f'{name}="{escaped_value}"')
        else:
            rebuilt_attrs.append(name)
    if not changed:
        return match.group(0), False
    # Preserve the original tag name casing (SVG relies on camelCase such as `linearGradient`).
    attrs = "".join(f" {attr}" for attr in rebuilt_attrs)
    return f"<{slash}{tag_name}{attrs}{'/' if is_self_closing else ''}>", True


def _drop_elements(text: str, tags, content: bool) -> Tuple[str, bool]:
    """Remove elements; with ``content=True`` the element body is removed as well."""
    changed = False
    for tag in tags:
        if content:
            pattern = re.compile(rf"<\s*{tag}\b[^<>]*>[\s\S]*?<\s*/\s*{tag}\s*>", re.IGNORECASE)
            text, count = pattern.subn("", text)
            changed = changed or count > 0
        # Orphan/self-closing/variantly-cased open and close tags.
        pattern = re.compile(rf"<\s*/?\s*{tag}\b[^<>]*/?>", re.IGNORECASE)
        text, count = pattern.subn("", text)
        changed = changed or count > 0
    return text, changed


def sanitize_html_fragment(text: str) -> Tuple[str, bool]:
    """
    Remove active constructs from raw HTML embedded in Markdown.

    Returns the sanitised text and a flag indicating whether active content was
    removed. Safe markup is returned byte-for-byte unchanged.
    """
    changed = False
    for pattern in (_COMMENT_RE, _CDATA_RE, _PROCESSING_INSTRUCTION_RE, _DECLARATION_RE):
        text, count = pattern.subn("", text)
        changed = changed or count > 0

    text, dropped = _drop_elements(text, _DROP_WITH_CONTENT_TAGS, content=True)
    changed = changed or dropped
    text, dropped = _drop_elements(text, _DROP_VOID_TAGS, content=False)
    changed = changed or dropped

    def _replace(match: "re.Match[str]") -> str:
        nonlocal changed
        replacement, tag_changed = _scrub_tag(match)
        changed = changed or tag_changed
        return replacement

    text = _START_TAG_RE.sub(_replace, text)
    return text, changed


def html_fragment_is_active(text: str) -> bool:
    """Return whether ``text`` contains active HTML that the sanitizer would remove."""
    return sanitize_html_fragment(text)[1]


def sanitize_markdown_urls(text: str) -> Tuple[str, bool]:
    """
    Neutralise active URLs carried by Markdown syntax (links/images, autolinks
    and reference definitions). The visible link text is always preserved.
    """
    changed = False

    def _replace_link(match: "re.Match[str]") -> str:
        nonlocal changed
        label = match.group(1)
        target = match.group(2) if match.group(2) is not None else match.group(3)
        if _url_is_unsafe(target):
            changed = True
            # Empty target renders as a non-navigating anchor while keeping the label.
            return f"[{label}]()"
        return match.group(0)

    def _replace_definition(match: "re.Match[str]") -> str:
        nonlocal changed
        prefix, target = match.group(1), match.group(2)
        if _url_is_unsafe(target):
            changed = True
            return prefix
        return match.group(0)

    def _replace_autolink(match: "re.Match[str]") -> str:
        nonlocal changed
        changed = True
        return html.escape(match.group(1))

    text = _MD_LINK_RE.sub(_replace_link, text)
    text = _MD_REFERENCE_DEF_RE.sub(_replace_definition, text)
    text = _MD_AUTOLINK_RE.sub(_replace_autolink, text)
    return text, changed


def _sanitize_prose_line(line: str) -> Tuple[str, bool]:
    """Sanitise one non-code Markdown line, protecting inline code spans."""
    protected: List[str] = []

    def _stash(match: "re.Match[str]") -> str:
        protected.append(match.group(0))
        return f"\x00{len(protected) - 1}\x00"

    stashed = _CODE_SPAN_RE.sub(_stash, line)
    sanitized, changed = sanitize_markdown_urls(stashed)
    sanitized, html_changed = sanitize_html_fragment(sanitized)
    changed = changed or html_changed
    if not changed:
        return line, False

    def _restore(match: "re.Match[str]") -> str:
        return protected[int(match.group(1))]

    return re.sub(r"\x00(\d+)\x00", _restore, sanitized), True


def sanitize_markdown(text: str) -> str:
    """
    Sanitise raw HTML in a Markdown document while leaving code untouched.

    Fenced code blocks, indented code blocks and inline code spans (i.e. literal
    code examples) are preserved exactly, including any HTML/JS shown as text.
    """
    output: List[str] = []
    fence: Optional[Tuple[str, int]] = None
    for line in _splitlines_keepends(text):
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        fence_match = _FENCE_RE.match(stripped) if indent <= 3 else None
        if fence is None and fence_match:
            fence = (fence_match.group(1)[0], len(fence_match.group(1)))
            output.append(line)
        elif fence is not None and indent <= 3 and re.match(rf"^{re.escape(fence[0])}{{{fence[1]},}}\s*$", stripped):
            fence = None
            output.append(line)
        elif fence is not None:
            # Inside a fenced code block: literal content, never rewritten.
            output.append(line)
        elif _INDENTED_CODE_RE.match(line):
            # Indented code block: literal content.
            output.append(line)
        else:
            output.append(_sanitize_prose_line(line)[0])
    return "".join(output)


def find_unsafe_markdown(text: str) -> Optional[str]:
    """
    Audit rendered Markdown for active content outside of code regions.

    Returns a short excerpt of the first offending region, or ``None`` when the
    document is clean. Used as a final verification pass after export.
    """
    fence: Optional[Tuple[str, int]] = None
    for line in _splitlines_keepends(text):
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        fence_match = _FENCE_RE.match(stripped) if indent <= 3 else None
        if fence is None and fence_match:
            fence = (fence_match.group(1)[0], len(fence_match.group(1)))
            continue
        if fence is not None:
            if indent <= 3 and re.match(rf"^{re.escape(fence[0])}{{{fence[1]},}}\s*$", stripped):
                fence = None
            continue
        if _INDENTED_CODE_RE.match(line):
            continue
        if _sanitize_prose_line(line)[1]:
            return line.strip()[:120]
    return None


def _splitlines_keepends(text: str) -> List[str]:
    return re.split(r"(?<=\n)", text)


def svg_is_active(svg: str) -> bool:
    """Return whether an SVG document embeds scripts, handlers or other activity."""
    return any(pattern.search(svg) for pattern in _SVG_ACTIVE_PATTERNS)


def sanitize_svg(svg: str) -> Tuple[str, bool]:
    """Strip active content from an SVG document, returning the static SVG."""
    sanitized, changed = sanitize_html_fragment(svg)
    # ``sanitize_html_fragment`` already removes scripts, foreignObject, PIs,
    # declarations and unsafe attributes; comments alone are not active.
    return sanitized, changed


def _decode_maybe_base64(value: Any) -> Tuple[str, bool]:
    """Decode an attachment payload which may be base64-encoded UTF-8 text."""
    text = _to_text(value).strip()
    if re.fullmatch(r"[A-Za-z0-9+/\s]+={0,2}", text or "") and not text.lstrip().startswith("<"):
        try:
            return base64.b64decode(text, validate=True).decode("utf-8"), True
        except (binascii.Error, UnicodeDecodeError, ValueError):
            pass
    return _to_text(value), False


def _encode_maybe_base64(text: str, was_base64: bool) -> str:
    if was_base64:
        return base64.b64encode(text.encode("utf-8")).decode("ascii")
    return text


class SafeNotebookPreprocessor(Preprocessor):
    """
    Apply the default-safe publishing policy to a notebook before Markdown export.

    Set ``trusted=True`` to disable all sanitisation and preserve the notebook's
    original rich media and active content.
    """

    trusted = Bool(False, help="Trust notebook content and preserve active/rich output").tag(config=True)

    def preprocess(self, nb, resources):
        if self.trusted:
            return nb, resources
        for cell in nb.cells:
            if cell.cell_type == "markdown":
                cell.source = sanitize_markdown(_to_text(cell.get("source", "")))
                self._sanitize_attachments(cell)
            elif cell.cell_type == "raw":
                cell_format = str(cell.get("metadata", {}).get("format", ""))
                source = _to_text(cell.get("source", ""))
                if cell_format.lower() in {"html", "text/html"}:
                    cell.source = sanitize_html_fragment(source)[0]
                else:
                    cell.source = sanitize_markdown(source)
                self._sanitize_attachments(cell)
            elif cell.cell_type == "code":
                # Code cell sources are literal examples and must be preserved.
                for output in cell.get("outputs", []):
                    self._sanitize_output(output)
        return nb, resources

    def _sanitize_output(self, output) -> None:
        if output.get("output_type") not in {"display_data", "execute_result"}:
            return
        data = output.get("data", {})
        if not data:
            return

        for mime in list(data.keys()):
            if mime in ACTIVE_MIMES:
                self.log.info("Dropping active output MIME type `%s`", mime)
                del data[mime]

        if "text/markdown" in data:
            data["text/markdown"] = sanitize_markdown(_to_text(data["text/markdown"]))

        if "text/html" in data:
            html_text = _to_text(data["text/html"])
            sanitized, active = sanitize_html_fragment(html_text)
            if active:
                if self._has_safe_alternative(data):
                    self.log.info("Unsafe HTML output detected; falling back to static image or text")
                    del data["text/html"]
                else:
                    self.log.info("Unsafe HTML output sanitised in place (no static alternative)")
                    data["text/html"] = sanitized

        if "image/svg+xml" in data:
            svg_text = _to_text(data["image/svg+xml"])
            if svg_is_active(svg_text):
                if self._has_raster_alternative(data):
                    self.log.info("Unsafe SVG output detected; falling back to static raster image")
                    del data["image/svg+xml"]
                else:
                    sanitized, _ = sanitize_svg(svg_text)
                    if svg_is_active(sanitized):
                        self.log.warning("Could not sanitise SVG output; replacing with a static placeholder")
                        del data["image/svg+xml"]
                        data.setdefault("text/plain", "[SVG output removed for safety]")
                    else:
                        self.log.info("Unsafe SVG output sanitised in place (no static alternative)")
                        data["image/svg+xml"] = sanitized

    @staticmethod
    def _has_safe_alternative(data: Dict[str, Any]) -> bool:
        return any(mime in data for mime in STATIC_IMAGE_MIMES + SAFE_TEXT_MIMES)

    @staticmethod
    def _has_raster_alternative(data: Dict[str, Any]) -> bool:
        return any(mime in data for mime in STATIC_IMAGE_MIMES)

    def _sanitize_attachments(self, cell) -> None:
        attachments = cell.get("attachments")
        if not attachments:
            return
        for filename in list(attachments.keys()):
            bundle = attachments[filename]
            for mime in list(bundle.keys()):
                if mime in ACTIVE_MIMES:
                    del bundle[mime]
            if "image/svg+xml" not in bundle:
                continue
            svg_text, was_base64 = _decode_maybe_base64(bundle["image/svg+xml"])
            if not svg_is_active(svg_text):
                continue
            if self._has_raster_alternative(bundle):
                self.log.info("Unsafe SVG attachment `%s`; using raster alternative", filename)
                del bundle["image/svg+xml"]
                continue
            sanitized, _ = sanitize_svg(svg_text)
            if svg_is_active(sanitized):
                self.log.warning("Removing unsafe SVG attachment `%s`", filename)
                del attachments[filename]
            else:
                bundle["image/svg+xml"] = _encode_maybe_base64(sanitized, was_base64)


def sanitize_exported_resources(body: str, resources: Dict[str, Any], trusted: bool, logger: Optional[logging.Logger] = None) -> None:
    """
    Final guard over resource files extracted into the page bundle.

    Extraction preprocessors run before user preprocessors, so an SVG removed
    from an output bundle can still be present among ``resources['outputs']``.
    Neutralise or block dangerous files in place. No-op under ``trusted`` mode.
    """
    if trusted:
        return
    logger = logger or log
    outputs = resources.get("outputs", {})
    for name in list(outputs.keys()):
        lowered_name = name.lower()
        extension = os.path.splitext(lowered_name)[1]

        if extension in BLOCKED_RESOURCE_EXTENSIONS:
            logger.warning("Blocking active resource file `%s` from page bundle", name)
            del outputs[name]
            continue

        if extension != ".svg":
            continue

        svg_text = outputs[name].decode("utf-8", errors="replace")
        if not svg_is_active(svg_text):
            continue
        referenced = os.path.basename(name) in body
        if not referenced:
            # The dangerous MIME was dropped in favour of a static alternative;
            # the extracted file is an orphan and must not reach the bundle.
            logger.warning("Removing unsafe, unreferenced SVG resource `%s`", name)
            del outputs[name]
            continue

        sanitized, _ = sanitize_svg(svg_text)
        if svg_is_active(sanitized):
            logger.warning("Replacing unsanitizable SVG resource `%s` with a placeholder", name)
            outputs[name] = _BLOCKED_SVG.encode("utf-8")
        else:
            logger.warning("Sanitised active content in SVG resource `%s`", name)
            outputs[name] = sanitized.encode("utf-8")
