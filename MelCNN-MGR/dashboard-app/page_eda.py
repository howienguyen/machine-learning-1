from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components


DASHBOARD_DIR = Path(__file__).resolve().parent
MELCNN_DIR = DASHBOARD_DIR.parent
NOTEBOOK_EXPORTS_ROOT = MELCNN_DIR / "data" / "eda" / "notebook_exports"

SECTION_ORDER = [
	"Executive summary & readiness",
	"Final split quality & leakage safety",
	"Log-mel cache readiness",
	"File discovery & sampling audit",
	"Schema & configuration",
	"Small-split supplementation audit",
	"Notebook overview",
	"Supporting methods & appendix",
]


def _list_export_dirs(root: Path) -> list[Path]:
	if not root.exists():
		return []
	candidates = [path for path in root.iterdir() if path.is_dir() and (path / "notebook_export.json").exists()]
	return sorted(candidates, key=lambda path: path.name, reverse=True)


@st.cache_data(show_spinner=False)
def _read_json(path_str: str) -> dict[str, Any]:
	return json.loads(Path(path_str).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _read_text(path_str: str) -> str:
	return Path(path_str).read_text(encoding="utf-8")


def _extract_heading(source: str) -> str | None:
	for line in source.splitlines():
		text = line.strip()
		if text.startswith("#"):
			return text.lstrip("# ").strip() or None
	return None


def _classify_section(cell: dict[str, Any], current_heading: str | None) -> str:
	source = str(cell.get("source", ""))
	text = f"{current_heading or ''}\n{source}".casefold()

	if any(token in text for token in ["summarize readiness", "remaining gaps", "summary_markdown", "readiness", "divergence"]):
		return "Executive summary & readiness"
	if any(token in text for token in ["final split", "leakage", "additional-source contribution", "genre and split", "final segment rows"]):
		return "Final split quality & leakage safety"
	if any(token in text for token in ["log-mel", "logmel", "alignment", "logmel cache", "split_dir", "logmel_status"]):
		return "Log-mel cache readiness"
	if any(token in text for token in ["file-level", "reason code", "sampling eligibility", "duration summary", "discovered audio candidates"]):
		return "File discovery & sampling audit"
	if any(token in text for token in ["schema", "configuration", "artifact presence", "manifest builder report preview"]):
		return "Schema & configuration"
	if any(token in text for token in ["small-split", "supplementation", "exact-small", "payload allocation", "split-distance metrics"]):
		return "Small-split supplementation audit"
	if any(token in text for token in ["notebook map", "goal is not just", "continuous audit", "pipeline as one continuous audit"]):
		return "Notebook overview"
	return "Supporting methods & appendix"


def _group_cells(cells: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
	grouped = {section: [] for section in SECTION_ORDER}
	current_heading: str | None = None
	for cell in cells:
		if str(cell.get("cell_type")) == "markdown":
			heading = _extract_heading(str(cell.get("source", "")))
			if heading:
				current_heading = heading
		section = _classify_section(cell, current_heading)
		grouped.setdefault(section, []).append(cell)

	return {section: grouped.get(section, []) for section in SECTION_ORDER if grouped.get(section)}


def _render_output(output: dict[str, Any], export_dir: Path, cell_number: int, output_number: int) -> None:
	representations = output.get("representations") or []
	if not representations:
		return

	primary_mime = output.get("primary_mime")
	primary = next((item for item in representations if item.get("mime") == primary_mime), None)
	if primary is None:
		primary = representations[0]

	mime = str(primary.get("mime", "text/plain"))
	rel_path = str(primary.get("path", ""))
	asset_path = export_dir / rel_path if rel_path else None
	if asset_path is None or not asset_path.exists():
		st.warning(f"Missing asset for Cell {cell_number}, output {output_number}: {rel_path}")
		return

	if mime == "text/markdown":
		st.markdown(_read_text(str(asset_path)))
	elif mime in {"text/plain", "application/json"}:
		if mime == "application/json":
			try:
				st.json(_read_json(str(asset_path)), expanded=False)
			except Exception:
				st.code(_read_text(str(asset_path)), language="json")
		else:
			st.code(_read_text(str(asset_path)), language="text")
	elif mime == "text/html":
		html_payload = _read_text(str(asset_path))
		default_height = 420 if "<table" in html_payload.lower() else 640
		components.html(html_payload, height=default_height, scrolling=True)
	elif mime.startswith("image/"):
		st.image(str(asset_path), use_container_width=True)
	else:
		st.code(_read_text(str(asset_path)), language="text")

	if len(representations) > 1:
		labels = [item.get("mime", "unknown") for item in representations]
		st.caption("Available representations: " + ", ".join(labels))


def _render_cell(cell: dict[str, Any], export_dir: Path, show_code: bool) -> None:
	cell_number = int(cell.get("cell_number", 0))
	cell_type = str(cell.get("cell_type", "unknown"))
	output_count = int(cell.get("output_count", 0))
	title = f"Cell {cell_number} · {cell_type} · outputs={output_count}"

	expanded = cell_type == "markdown" or output_count > 0
	with st.expander(title, expanded=expanded):
		source = str(cell.get("source", ""))

		if cell_type == "markdown":
			st.markdown(source)
		elif show_code:
			st.code(source, language="python")

		outputs = cell.get("outputs") or []
		for output_number, output in enumerate(outputs, start=1):
			st.markdown(f"**Output {output_number}**")
			_render_output(output, export_dir, cell_number, output_number)


def render() -> None:
	st.markdown(
		"""
		<style>
		.page2-subheader {
			display: inline-block;
			padding: 0.28rem 0.0rem;
			margin-top: 1rem;
			margin-bottom: 1rem;
			border-bottom: 2px solid rgba(99, 179, 237, 0.8);
			font-size: 1.25rem;
			font-weight: 600;
			line-height: 1.2;
		}

		div[data-testid="stMetricValue"],
		div[data-testid="stMetricValue"] > div {
			font-size: 1.25rem;
			line-height: 1.1;
		}
		</style>
		<div class="page2-subheader">🔅 Exploratory Data Analysis: Notebook Export Viewer</div>
		""",
		unsafe_allow_html=True,
	)

	export_dirs = _list_export_dirs(NOTEBOOK_EXPORTS_ROOT)
	if not export_dirs:
		st.error(f"No notebook exports found under {NOTEBOOK_EXPORTS_ROOT}")
		st.info(
			"Run the notebook export step first so the dashboard can load the generated JSON bundle and assets."
		)
		return

	export_labels: list[str] = []
	export_map: dict[str, Path] = {}
	for export_dir in export_dirs:
		try:
			summary = _read_json(str(export_dir / "export_summary.json"))
			title = str(summary.get("notebook_title") or export_dir.name)
			label = f"{title} · {export_dir.name}"
		except Exception:
			label = export_dir.name
		export_labels.append(label)
		export_map[label] = export_dir

	label_col, select_col, cap_col = st.columns([2, 3, 5])
	with label_col:
		st.markdown(
			'<div style="white-space: nowrap; font-weight: 600;">Select notebook export:</div>',
			unsafe_allow_html=True,
		)
	with select_col:
		selected_label = st.selectbox(
			"Notebook export",
			options=export_labels,
			index=0,
			help="Latest export bundle is selected by default.",
			label_visibility="collapsed",
		)
	with cap_col:
		st.caption("This page re-renders notebook outputs from the exported JSON bundle and extracted assets.")

	selected_dir = export_map[selected_label]
	manifest = _read_json(str(selected_dir / "notebook_export.json"))
	notebook_title = str(manifest.get("notebook_title") or selected_dir.name)

	top_cols = st.columns(4)
	top_cols[0].metric("Cells", int(manifest.get("cell_count", 0)))
	top_cols[1].metric("Outputs", int(manifest.get("output_count", 0)))
	top_cols[2].metric("Images", int(manifest.get("image_output_count", 0)))
	top_cols[3].metric("Bundle", selected_dir.name)

	control_left, control_mid, control_right = st.columns([5, 2, 2])
	with control_left:
		st.markdown(f"**Notebook:** {notebook_title}")
	with control_mid:
		show_code = st.toggle("Show code cells", value=False)
	with control_right:
		show_markdown_only = st.toggle("Markdown only", value=False)

	with st.expander("1) Bundle metadata", expanded=False):
		st.json(
			{
				"notebook_title": manifest.get("notebook_title"),
				"notebook_name": manifest.get("notebook_name"),
				"export_dir": manifest.get("export_dir"),
				"cell_count": manifest.get("cell_count"),
				"output_count": manifest.get("output_count"),
				"image_output_count": manifest.get("image_output_count"),
			},
			expanded=False,
		)

	cells = manifest.get("cells") or []
	if show_markdown_only:
		cells = [cell for cell in cells if cell.get("cell_type") == "markdown"]
	grouped_cells = _group_cells(cells)

	with st.expander("2) Section overview", expanded=False):
		overview_rows = []
		for section_name, section_cells in grouped_cells.items():
			overview_rows.append(
				{
					"section": section_name,
					"cells": len(section_cells),
					"outputs": sum(int(cell.get("output_count", 0)) for cell in section_cells),
				}
			)
		st.dataframe(overview_rows, width="stretch", hide_index=True)

	section_number = 3
	for section_name, section_cells in grouped_cells.items():
		total_outputs = sum(int(cell.get("output_count", 0)) for cell in section_cells)
		title = f"{section_number}) {section_name}"
		with st.expander(title, expanded=section_number == 3):
			st.caption(f"Cells: {len(section_cells)} · Outputs: {total_outputs}")
			for cell in section_cells:
				_render_cell(cell, selected_dir, show_code=show_code)
		section_number += 1

	st.caption(f"Rendering from {selected_dir / 'notebook_export.json'}")
