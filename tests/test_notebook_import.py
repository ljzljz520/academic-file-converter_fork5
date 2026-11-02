import base64
import logging

import nbformat as nbf

from academic import cli
from academic.import_notebook import import_notebook
from academic.notebook_sanitizer import (
    find_unsafe_markdown,
    sanitize_html_fragment,
    sanitize_markdown,
    sanitize_svg,
    svg_is_active,
)

# A valid 1x1 PNG used as the static fallback representation of rich outputs.
PNG_1X1 = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c6360000002000100ffff03000006000557bfabd40000000049454e44ae426082"
    )
).decode()

ACTIVE_SVG = "<svg xmlns='http://www.w3.org/2000/svg' onload='alert(2)'>" "<script>alert(3)</script><linearGradient id='g'/><circle r='1'/></svg>"
SAFE_SVG = "<svg xmlns='http://www.w3.org/2000/svg'><linearGradient id='g'/><circle r='1'/></svg>"


def _run_import(tmp_path, notebook, trusted=False):
    """Write ``notebook`` into a temp folder, import it and return (markdown, bundle_path)."""
    nbf.write(notebook, str(tmp_path / "post.ipynb"))
    import_notebook(
        str(tmp_path / "*.ipynb"),
        output_dir=str(tmp_path / "out"),
        overwrite=True,
        trusted=trusted,
    )
    bundle_path = tmp_path / "out" / "post"
    return (bundle_path / "index.md").read_text(), bundle_path


def test_notebook_import_no_output(capfd):
    """
    The importer does not have a flag to determine what formats the user intends to convert.
    Instead, it relies on the file extension in the input string.
    If there is no file extension (e.g. just a wildcard), then expect no output.
    """
    cli.parse_args(["import", "--dry-run", "--verbose", "tests/data/notebooks/*", "content/post/"])
    out, err = capfd.readouterr()
    assert out == ""
    assert err == ""


def test_notebook_import_info_level(caplog):
    caplog.set_level(logging.INFO)

    cli.parse_args(
        [
            "import",
            "tests/data/notebooks/*.ipynb",
            "content/post/",
            "--dry-run",
        ]
    )
    # assert "Found notebook `test.ipynb`" in out
    assert "Searching for Jupyter notebooks in `tests/data/notebooks/*.ipynb`" in caplog.text


def test_notebook_import_debug_level(caplog):
    caplog.set_level(logging.DEBUG)

    cli.parse_args(
        [
            "import",
            "tests/data/notebooks/*.ipynb",
            "content/post/",
            "--dry-run",
        ]
    )

    # Note: this logging output should only be shown at DEBUG log level, so we set the corresponding level above
    assert "Found notebook `tests/data/notebooks/test.ipynb`" in caplog.text
    assert "Found notebook `tests/data/notebooks/blog-with-jupyter.ipynb`" in caplog.text


def test_cli_trusted_flag_is_accepted():
    # The explicit trust opt-in must parse without error and must not execute the import in dry-run mode.
    cli.parse_args(["import", "--trusted", "tests/data/notebooks/*.ipynb", "content/post/", "--dry-run"])


def test_markdown_active_html_stripped_but_code_examples_preserved(tmp_path):
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            "# Demo\n\n"
            "<script>alert('md')</script>\n\n"
            '<img src="x" onerror="alert(1)">\n\n'
            "<b>bold is safe</b>\n\n"
            "```html\n<script>alert('fenced example')</script>\n```\n\n"
            "inline `<script>span_example()</script>` code\n"
        ),
        # Code cell source is a literal example and must survive unchanged.
        nbf.v4.new_code_cell('code_example = "<script>source_example()</script>"'),
    ]

    markdown, _ = _run_import(tmp_path, notebook)

    assert "<script>alert('md')</script>" not in markdown
    assert "onerror" not in markdown
    assert "<b>bold is safe</b>" in markdown
    # Literal code examples are preserved verbatim ...
    assert "<script>alert('fenced example')</script>" in markdown
    assert "<script>span_example()</script>" in markdown
    assert '"<script>source_example()</script>"' in markdown
    # ... and the final audit finds nothing outside code regions.
    assert find_unsafe_markdown(markdown) is None


def test_markdown_javascript_url_is_neutralised(tmp_path):
    notebook = nbf.v4.new_notebook()
    notebook.cells = [nbf.v4.new_markdown_cell('# Demo\n\n<a href="javascript:alert(1)">click</a>\n')]

    markdown, _ = _run_import(tmp_path, notebook)

    assert "javascript:alert" not in markdown
    assert ">click</a>" in markdown


def test_html_output_falls_back_to_static_png(tmp_path):
    output = nbf.v4.new_output(
        output_type="display_data",
        data={
            "text/html": "<table></table><script>alert('html')</script>",
            "image/png": PNG_1X1,
            "text/plain": "DataFrame()",
        },
    )
    notebook = nbf.v4.new_notebook()
    notebook.cells = [nbf.v4.new_code_cell("df", outputs=[output])]

    markdown, bundle = _run_import(tmp_path, notebook)

    assert "alert('html')" not in markdown
    assert "![png](output_" in markdown
    assert not list(bundle.glob("*.html"))
    png_files = list(bundle.glob("*.png"))
    assert len(png_files) == 1
    assert png_files[0].read_bytes() == base64.b64decode(PNG_1X1)


def test_html_output_without_alternative_is_sanitized(tmp_path):
    output = nbf.v4.new_output(
        output_type="display_data",
        data={"text/html": "<p>safe text</p><script>alert('html')</script>"},
    )
    notebook = nbf.v4.new_notebook()
    notebook.cells = [nbf.v4.new_code_cell("widget()", outputs=[output])]

    markdown, _ = _run_import(tmp_path, notebook)

    assert "<p>safe text</p>" in markdown
    assert "<script" not in markdown


def test_active_svg_output_falls_back_to_static_png(tmp_path):
    output = nbf.v4.new_output(
        output_type="display_data",
        data={"image/svg+xml": ACTIVE_SVG, "image/png": PNG_1X1},
    )
    notebook = nbf.v4.new_notebook()
    notebook.cells = [nbf.v4.new_code_cell("plot()", outputs=[output])]

    markdown, bundle = _run_import(tmp_path, notebook)

    assert "![png](output_" in markdown
    assert "![svg]" not in markdown
    assert not list(bundle.glob("*.svg"))


def test_active_svg_output_without_raster_is_sanitized_in_place(tmp_path):
    output = nbf.v4.new_output(output_type="display_data", data={"image/svg+xml": ACTIVE_SVG})
    notebook = nbf.v4.new_notebook()
    notebook.cells = [nbf.v4.new_code_cell("plot()", outputs=[output])]

    markdown, bundle = _run_import(tmp_path, notebook)

    assert "![svg](output_" in markdown
    svg_files = list(bundle.glob("*.svg"))
    assert len(svg_files) == 1
    svg_text = svg_files[0].read_text()
    assert not svg_is_active(svg_text)
    assert "<script" not in svg_text
    assert "onload" not in svg_text
    # Static SVG structure and camelCase element names must survive sanitization.
    assert "linearGradient" in svg_text
    assert "<circle" in svg_text


def test_safe_svg_and_safe_html_are_preserved(tmp_path):
    outputs = [
        nbf.v4.new_output(output_type="display_data", data={"image/svg+xml": SAFE_SVG}),
        nbf.v4.new_output(output_type="display_data", data={"text/html": "<table><tr><td>ok</td></tr></table>"}),
    ]
    notebook = nbf.v4.new_notebook()
    notebook.cells = [nbf.v4.new_code_cell("plot(); table()", outputs=outputs)]

    markdown, bundle = _run_import(tmp_path, notebook)

    assert "<table><tr><td>ok</td></tr></table>" in markdown
    svg_files = list(bundle.glob("*.svg"))
    assert len(svg_files) == 1
    assert svg_files[0].read_text() == SAFE_SVG


def test_javascript_and_widget_mimes_are_dropped(tmp_path):
    outputs = [
        nbf.v4.new_output(
            output_type="display_data",
            data={"application/javascript": "alert(1)", "text/plain": "javascript output"},
        ),
        nbf.v4.new_output(
            output_type="display_data",
            data={
                "application/vnd.jupyter.widget-view+json": {"model_id": "x"},
                "text/plain": "widget representation",
            },
        ),
    ]
    notebook = nbf.v4.new_notebook()
    notebook.cells = [nbf.v4.new_code_cell("js(); w()", outputs=outputs)]

    markdown, _ = _run_import(tmp_path, notebook)

    assert "alert(1)" not in markdown
    assert "model_id" not in markdown
    assert "javascript output" in markdown
    assert "widget representation" in markdown


def test_active_svg_attachment_is_sanitized(tmp_path):
    cell = nbf.v4.new_markdown_cell("![pic](attachment:pic.svg)")
    cell["attachments"] = {"pic.svg": {"image/svg+xml": base64.b64encode(ACTIVE_SVG.encode()).decode()}}
    notebook = nbf.v4.new_notebook()
    notebook.cells = [cell]

    markdown, bundle = _run_import(tmp_path, notebook)

    assert "pic.svg" in markdown
    attachment = bundle / "pic.svg"
    assert attachment.exists()
    assert not svg_is_active(attachment.read_text())


def test_trusted_mode_preserves_active_content(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    outputs = [
        nbf.v4.new_output(
            output_type="display_data",
            data={"text/html": "<script>alert('html')</script>", "image/png": PNG_1X1},
        ),
        nbf.v4.new_output(
            output_type="display_data",
            data={"image/svg+xml": ACTIVE_SVG, "image/png": PNG_1X1},
        ),
    ]
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell("# Demo\n\n<script>alert('md')</script>\n"),
        nbf.v4.new_code_cell("df; plot()", outputs=outputs),
    ]

    markdown, bundle = _run_import(tmp_path, notebook, trusted=True)

    assert "<script>alert('md')</script>" in markdown
    assert "<script>alert('html')</script>" in markdown
    svg_files = list(bundle.glob("*.svg"))
    assert len(svg_files) == 1
    assert "onload='alert(2)'" in svg_files[0].read_text()
    assert "Trusted mode" in caplog.text


def test_shipped_fixture_notebooks_import_without_active_content(tmp_path):
    import_notebook(
        "tests/data/notebooks/*.ipynb",
        output_dir=str(tmp_path / "out"),
        overwrite=True,
    )
    index_files = list((tmp_path / "out").glob("*/index.md"))
    assert {path.parent.name for path in index_files} == {"test", "blog-with-jupyter"}
    for index_file in index_files:
        assert find_unsafe_markdown(index_file.read_text()) is None


def test_sanitize_markdown_protects_code_regions():
    text = (
        "<script>alert(1)</script>\n\n"
        "~~~\n<a href='javascript:alert(2)'>x</a>\n~~~\n\n"
        "    <script>indented()</script>\n\n"
        "use `<script>inline_guard()</script>` here\n"
    )

    sanitized = sanitize_markdown(text)

    assert "<script>alert(1)</script>" not in sanitized
    assert "<a href='javascript:alert(2)'>x</a>" in sanitized
    assert "<script>indented()</script>" in sanitized
    assert "<script>inline_guard()</script>" in sanitized


def test_sanitize_html_fragment_handles_parser_obfuscation():
    assert sanitize_html_fragment("<img/src=x/onerror=alert(1)>")[1]
    assert sanitize_html_fragment("<svg/onload=alert(1)>")[1]
    assert sanitize_html_fragment('<a href="&#106;avascript:alert(1)">x</a>')[1]
    assert sanitize_html_fragment("<meta http-equiv='refresh' content='0'>")[1]
    # Prose containing comparison operators is not markup and must stay untouched.
    assert sanitize_html_fragment("if a < b and 2 < 3: pass")[1] is False
    assert sanitize_html_fragment("if a < b and 2 < 3: pass")[0] == "if a < b and 2 < 3: pass"


def test_markdown_active_urls_are_neutralised(tmp_path):
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            "# Demo\n\n"
            "[bad](javascript:alert(1))\n\n"
            "[titled](vbscript:msgbox(1) 'tip')\n\n"
            "see <javascript:alert(1)> here\n\n"
            "[indirect][ref]\n\n[ref]: javascript:alert(1)\n\n"
            "[good](https://en.wikipedia.org/wiki/Foo_(bar))\n"
        )
    ]

    markdown, _ = _run_import(tmp_path, notebook)

    # No navigable Markdown construct may retain the active URL ...
    assert "(javascript:alert(1)" not in markdown
    assert "(vbscript:msgbox(1)" not in markdown
    assert "<javascript:alert(1)>" not in markdown
    assert not any("javascript" in line and line.strip().startswith("[ref]") for line in markdown.splitlines())
    # ... while visible text and safe links are preserved.
    assert "[bad]()" in markdown
    assert "indirect" in markdown
    assert "https://en.wikipedia.org/wiki/Foo_(bar)" in markdown
    assert find_unsafe_markdown(markdown) is None


def test_raw_cells_and_markdown_outputs_are_sanitised(tmp_path):
    raw_html_cell = nbf.v4.new_raw_cell("<script>alert('raw')</script><b>ok</b>")
    raw_html_cell.metadata = {"format": "text/html"}
    md_output = nbf.v4.new_output(
        output_type="display_data",
        data={"text/markdown": "result <img src=x onerror=alert(1)>\n", "text/plain": "fallback"},
    )
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        raw_html_cell,
        nbf.v4.new_code_cell("rich()", outputs=[md_output]),
    ]

    markdown, _ = _run_import(tmp_path, notebook)

    assert "alert('raw')" not in markdown
    assert "<b>ok</b>" in markdown
    assert "onerror" not in markdown
    assert find_unsafe_markdown(markdown) is None


def test_sanitize_svg_removes_active_vectors():
    svg = (
        '<?xml version="1.0"?><!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
        "<svg xmlns='http://www.w3.org/2000/svg' onload='alert(1)'>"
        "<foreignObject><body><script>alert(2)</script></body></foreignObject>"
        "<a xlink:href='javascript:alert(3)'><text>x</text></a></svg>"
    )
    assert svg_is_active(svg)
    sanitized, changed = sanitize_svg(svg)
    assert changed
    assert not svg_is_active(sanitized)
    assert "<text>x</text>" in sanitized
