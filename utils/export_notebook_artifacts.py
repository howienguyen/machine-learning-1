#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import markdown


MIME_EXTENSION_MAP = {
    "text/html": ".html",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "application/json": ".json",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/svg+xml": ".svg",
}

PRIMARY_MIME_ORDER = [
    "text/html",
    "image/png",
    "image/jpeg",
    "image/svg+xml",
    "text/markdown",
    "text/plain",
    "application/json",
]


@dataclass
class ExportPaths:
    output_dir: Path
    assets_dir: Path
    manifest_path: Path
    summary_path: Path
    html_path: Path
    notebook_copy_path: Path


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    return str(value)


def _safe_slug(value: str) -> str:
    slug = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            slug.append(ch)
        else:
            slug.append("_")
    return "".join(slug).strip("_") or "artifact"


def _find_melcnn_dir(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if candidate.name == "MelCNN-MGR" and (candidate / "settings.json").exists():
            return candidate
    raise FileNotFoundError("Could not locate MelCNN-MGR root from notebook path.")


def _default_output_dir(notebook_path: Path) -> Path:
    melcnn_dir = _find_melcnn_dir(notebook_path.parent)
    return melcnn_dir / "data" / "eda" / "notebook_exports" / notebook_path.stem


def _make_export_paths(output_dir: Path, notebook_path: Path) -> ExportPaths:
    return ExportPaths(
        output_dir=output_dir,
        assets_dir=output_dir / "assets",
        manifest_path=output_dir / "notebook_export.json",
        summary_path=output_dir / "export_summary.json",
        html_path=output_dir / "index.html",
        notebook_copy_path=output_dir / notebook_path.name,
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _pick_primary_mime(mime_entries: list[dict[str, Any]]) -> str | None:
    present = {entry["mime"] for entry in mime_entries}
    for mime in PRIMARY_MIME_ORDER:
        if mime in present:
            return mime
    return mime_entries[0]["mime"] if mime_entries else None


def _export_stream_output(output: dict[str, Any], cell_number: int, output_number: int, assets_dir: Path) -> dict[str, Any]:
    text = _normalize_text(output.get("text"))
    rel_path = Path("assets") / f"cell_{cell_number:03d}_output_{output_number:02d}_stream.txt"
    _write_text(assets_dir.parent / rel_path, text)
    return {
        "output_type": "stream",
        "name": output.get("name", "stdout"),
        "representations": [
            {
                "mime": "text/plain",
                "path": rel_path.as_posix(),
                "chars": len(text),
            }
        ],
        "primary_mime": "text/plain",
        "preview": text[:400],
    }


def _export_error_output(output: dict[str, Any], cell_number: int, output_number: int, assets_dir: Path) -> dict[str, Any]:
    traceback_text = _normalize_text(output.get("traceback"))
    if not traceback_text:
        traceback_text = f"{output.get('ename', 'Error')}: {output.get('evalue', '')}"
    rel_path = Path("assets") / f"cell_{cell_number:03d}_output_{output_number:02d}_error.txt"
    _write_text(assets_dir.parent / rel_path, traceback_text)
    return {
        "output_type": "error",
        "ename": output.get("ename"),
        "evalue": output.get("evalue"),
        "representations": [
            {
                "mime": "text/plain",
                "path": rel_path.as_posix(),
                "chars": len(traceback_text),
            }
        ],
        "primary_mime": "text/plain",
        "preview": traceback_text[:400],
    }


def _export_rich_output(output: dict[str, Any], cell_number: int, output_number: int, assets_dir: Path) -> dict[str, Any]:
    data = output.get("data") or {}
    metadata = output.get("metadata") or {}
    mime_entries: list[dict[str, Any]] = []

    for mime, value in data.items():
        ext = MIME_EXTENSION_MAP.get(mime, ".txt")
        slug = _safe_slug(mime.replace("/", "_"))
        rel_path = Path("assets") / f"cell_{cell_number:03d}_output_{output_number:02d}_{slug}{ext}"
        abs_path = assets_dir.parent / rel_path

        if mime.startswith("image/") and mime != "image/svg+xml":
            raw_bytes = base64.b64decode(_normalize_text(value))
            _write_bytes(abs_path, raw_bytes)
            mime_entries.append(
                {
                    "mime": mime,
                    "path": rel_path.as_posix(),
                    "bytes": len(raw_bytes),
                }
            )
            continue

        text = _normalize_text(value)
        _write_text(abs_path, text)
        mime_entries.append(
            {
                "mime": mime,
                "path": rel_path.as_posix(),
                "chars": len(text),
            }
        )

    return {
        "output_type": output.get("output_type", "display_data"),
        "execution_count": output.get("execution_count"),
        "metadata": metadata,
        "representations": mime_entries,
        "primary_mime": _pick_primary_mime(mime_entries),
    }


def _render_source(cell_type: str, source: str) -> str:
    if cell_type == "markdown":
        return markdown.markdown(source, extensions=["tables", "fenced_code"])
    return f"<pre><code>{html.escape(source)}</code></pre>"


def _render_output(output: dict[str, Any], output_dir: Path) -> str:
    representations = output.get("representations") or []
    primary_mime = output.get("primary_mime")
    primary = next((item for item in representations if item.get("mime") == primary_mime), None)
    if primary is None and representations:
        primary = representations[0]

    rendered = []
    if primary is not None:
        mime = primary["mime"]
        rel_path = primary["path"]
        abs_path = output_dir / rel_path
        if mime == "text/html":
            rendered.append(abs_path.read_text(encoding="utf-8"))
        elif mime == "text/markdown":
            rendered.append(markdown.markdown(abs_path.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"]))
        elif mime in {"text/plain", "application/json"}:
            rendered.append(f"<pre>{html.escape(abs_path.read_text(encoding='utf-8'))}</pre>")
        elif mime.startswith("image/"):
            rendered.append(f'<img src="{html.escape(rel_path)}" alt="{html.escape(rel_path)}" loading="lazy">')
        else:
            rendered.append(f"<pre>{html.escape(abs_path.read_text(encoding='utf-8'))}</pre>")

    if len(representations) > 1:
        links = []
        for item in representations:
            links.append(
                f'<a href="{html.escape(item["path"])}" target="_blank" rel="noopener">{html.escape(item["mime"])}</a>'
            )
        rendered.append(f"<div class=\"output-links\">Representations: {' | '.join(links)}</div>")

    return "\n".join(rendered)


def _build_html(notebook_title: str, cells: list[dict[str, Any]], output_dir: Path) -> str:
    body_parts = []
    for cell in cells:
        cell_number = cell["cell_number"]
        cell_type = cell["cell_type"]
        source_html = _render_source(cell_type, cell["source"])
        output_html_parts = []
        for output in cell.get("outputs", []):
            output_html_parts.append(_render_output(output, output_dir))

        outputs_html = "".join(f'<div class="cell-output">{chunk}</div>' for chunk in output_html_parts if chunk)
        body_parts.append(
            f"""
<section class=\"cell cell-{cell_type}\">
  <div class=\"cell-header\">Cell {cell_number} · {cell_type}</div>
  <div class=\"cell-source\">{source_html}</div>
  {outputs_html}
</section>
"""
        )

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{html.escape(notebook_title)}</title>
  <style>
    :root {{
      --bg: #f7f7f8;
      --panel: #ffffff;
      --text: #1f2328;
      --muted: #59636e;
      --border: #d0d7de;
      --accent: #0f766e;
      --code-bg: #f6f8fa;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .page {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
    .hero {{ margin-bottom: 24px; }}
    .hero h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .hero p {{ margin: 0; color: var(--muted); }}
    .cell {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; margin-bottom: 20px; overflow: hidden; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }}
    .cell-header {{ padding: 10px 14px; background: #eef6f5; color: var(--accent); font-weight: 600; border-bottom: 1px solid var(--border); }}
    .cell-source {{ padding: 16px; border-bottom: 1px solid var(--border); }}
    .cell-output {{ padding: 16px; border-top: 1px solid var(--border); overflow-x: auto; }}
    .cell-output:first-of-type {{ border-top: 0; }}
    pre {{ margin: 0; padding: 12px; background: var(--code-bg); border: 1px solid var(--border); border-radius: 8px; overflow-x: auto; white-space: pre-wrap; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid var(--border); padding: 6px 8px; text-align: left; }}
    th {{ background: #f6f8fa; }}
    img {{ max-width: 100%; height: auto; border: 1px solid var(--border); border-radius: 8px; background: white; }}
    .output-links {{ margin-top: 10px; font-size: 13px; color: var(--muted); }}
    .output-links a {{ color: var(--accent); text-decoration: none; }}
    .output-links a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <main class=\"page\">
    <header class=\"hero\">
      <h1>{html.escape(notebook_title)}</h1>
      <p>Static export of notebook cells, inline outputs, figures, HTML tables, markdown blocks, and text streams.</p>
    </header>
    {''.join(body_parts)}
  </main>
</body>
</html>
"""


def export_notebook(notebook_path: Path, output_dir: Path, clear_output_dir: bool = False) -> ExportPaths:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    paths = _make_export_paths(output_dir, notebook_path)

    if clear_output_dir and paths.output_dir.exists():
        shutil.rmtree(paths.output_dir)

    paths.assets_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(notebook_path, paths.notebook_copy_path)

    cells_manifest: list[dict[str, Any]] = []
    total_outputs = 0
    total_images = 0

    for idx, cell in enumerate(notebook.get("cells", []), start=1):
        cell_outputs: list[dict[str, Any]] = []
        for output_index, output in enumerate(cell.get("outputs", []), start=1):
            output_type = output.get("output_type")
            if output_type == "stream":
                exported = _export_stream_output(output, idx, output_index, paths.assets_dir)
            elif output_type == "error":
                exported = _export_error_output(output, idx, output_index, paths.assets_dir)
            else:
                exported = _export_rich_output(output, idx, output_index, paths.assets_dir)
            cell_outputs.append(exported)
            total_outputs += 1
            total_images += sum(1 for item in exported.get("representations", []) if str(item.get("mime", "")).startswith("image/"))

        cell_source = _normalize_text(cell.get("source"))
        cell_type = str(cell.get("cell_type", "unknown"))
        cell_language = str((cell.get("metadata") or {}).get("language", cell_type))
        cells_manifest.append(
            {
                "cell_number": idx,
                "cell_type": cell_type,
                "language": cell_language,
                "source": cell_source,
                "source_lines": len(cell_source.splitlines()),
                "outputs": cell_outputs,
                "output_count": len(cell_outputs),
                "execution_count": cell.get("execution_count"),
            }
        )

    notebook_title = notebook_path.stem
    if cells_manifest:
        first_markdown = next((cell for cell in cells_manifest if cell["cell_type"] == "markdown" and cell["source"].strip()), None)
        if first_markdown:
            first_line = first_markdown["source"].strip().splitlines()[0].lstrip("# ").strip()
            if first_line:
                notebook_title = first_line

    manifest = {
        "notebook_path": str(notebook_path),
        "notebook_name": notebook_path.name,
        "notebook_title": notebook_title,
        "export_dir": str(paths.output_dir),
        "cell_count": len(cells_manifest),
        "output_count": total_outputs,
        "image_output_count": total_images,
        "cells": cells_manifest,
    }

    summary = {
        "notebook_name": notebook_path.name,
        "notebook_title": notebook_title,
        "cell_count": len(cells_manifest),
        "output_count": total_outputs,
        "image_output_count": total_images,
        "export_dir": str(paths.output_dir),
        "html": paths.html_path.name,
        "manifest": paths.manifest_path.name,
    }

    _write_text(paths.manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False))
    _write_text(paths.summary_path, json.dumps(summary, indent=2, ensure_ascii=False))
    _write_text(paths.html_path, _build_html(notebook_title, cells_manifest, paths.output_dir))
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a notebook's stored inline outputs into a web-app-friendly artifact bundle.",
    )
    parser.add_argument("notebook", type=str, help="Path to the source .ipynb file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for the artifact bundle. Defaults to MelCNN-MGR/data/eda/notebook_exports/<notebook_stem>.",
    )
    parser.add_argument(
        "--clear-output-dir",
        action="store_true",
        help="Delete the output directory before exporting.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    notebook_path = Path(args.notebook).expanduser().resolve()
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _default_output_dir(notebook_path)
    paths = export_notebook(notebook_path, output_dir, clear_output_dir=bool(args.clear_output_dir))
    print(f"Notebook export complete: {paths.output_dir}")
    print(f"HTML page       : {paths.html_path}")
    print(f"JSON manifest   : {paths.manifest_path}")
    print(f"Summary         : {paths.summary_path}")
    print(f"Notebook copy   : {paths.notebook_copy_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())