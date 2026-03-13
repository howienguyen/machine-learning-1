from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ── Paths ────────────────────────────────────────────────────────────────────
# data/eda/ is scanned for feature_* run directories.
# Legacy: older notebooks wrote to eda/output/feature_{RUN_ID}/.
from _paths import EDA_OUTPUT_ROOT, LEGACY_EDA_OUTPUT_ROOT

EDA_RUNS_ROOT = EDA_OUTPUT_ROOT
EDA_RUNS_ROOT_FALLBACK = LEGACY_EDA_OUTPUT_ROOT

# Canonical plot groups (filename → display label), exactly matching what
# Method_Ready_Data_EDA.ipynb saves plots under feature_{RUN_ID}/ (canonical: data/eda/).
_PLOT_GROUPS: dict[str, list[tuple[str, str]]] = {
    "Overview": [
        ("boxplots_method_ready_core.png", "Box plots (method-ready core variables)"),
    ],
    "User Features": [
        ("user_numeric_dist.png",    "Numeric feature distributions"),
        ("boxplots_user_features.png","Box plots (user features)"),
        ("user_genre_pref_mean.png", "Mean genre preference per genre"),
        ("user_corr.png",            "Pearson correlation heatmap"),
    ],
    "Movie Features": [
        ("movie_nratings_log.png",      "Rating-count distribution (log-scale)"),
        ("movie_numeric_dist.png",      "Numeric feature distributions"),
        ("boxplots_movie_features.png", "Box plots (movie features)"),
        ("movie_genre_flags.png",       "Genre flag frequency"),
        ("movie_entropy_vs_polar.png",  "Rating entropy vs polarisation score"),
        ("movie_corr.png",              "Pearson correlation heatmap"),
    ],
    "Transactions": [
        ("txn_basket_size.png",   "Basket-size distribution (log-scale)"),
        ("txn_token_types.png",   "Token type breakdown"),
        ("boxplots_transactions.png", "Box plots (transactions)"),
        ("txn_top30_tokens.png",  "Top-30 most frequent tokens"),
    ],
    "Similarity & Distance": [
        ("similarity_cosine_genre_pref.png", "Cosine similarity: users in genre_pref space"),
        ("similarity_jaccard_genre_flag.png", "Jaccard similarity: movies in genre_flag space"),
        ("similarity_euclidean_raw_vs_zscore.png", "Euclidean distance: raw vs z-scored activity features"),
    ],
}

_DEFAULT_FINDINGS_MD = """\
### Key Results & Interpretation

This dashboard page renders *precomputed* EDA artifacts.

Generate `data/eda/feature_<run_id>/` (older runs may exist under `eda/output/feature_<run_id>/`).
"""

_DEFAULT_DQ_MD = """\
## Data Quality Report

Generate the run artifacts under `data/eda/feature_<run_id>/` (or legacy `eda/output/feature_<run_id>/`).

Expected files for this section:
- `16_data_quality_report.md`
- `Method_Ready_Data_EDA_metrics.json`
"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _list_run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    runs = [p for p in root.glob("feature_*") if p.is_dir()]
    return sorted(runs, key=lambda p: p.name, reverse=True)


@st.cache_data(show_spinner=False)
def _read_csv(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


@st.cache_data(show_spinner=False)
def _read_json(path_str: str) -> dict[str, Any]:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _read_text(path_str: str) -> str:
    return Path(path_str).read_text(encoding="utf-8")


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


_FULL_WIDTH_PLOTS = {
    # Explicitly excluded from half-width rendering by request.
    "txn_top30_tokens.png",
    "user_genre_pref_mean.png",  # Mean genre preference per genre
    "movie_genre_flags.png",     # Genre flag frequency
    "txn_basket_size.png",
    "movie_numeric_dist.png"
    ,
    "boxplots_method_ready_core.png",
    "boxplots_user_features.png",
    "boxplots_movie_features.png",
    "boxplots_transactions.png",
    "similarity_cosine_genre_pref.png",
    "similarity_jaccard_genre_flag.png",
    "similarity_euclidean_raw_vs_zscore.png",
}


def _render_plot(path: Path, label: str, *, width_mode: str = "stretch") -> None:
    if path.exists():
        # Streamlit: `use_container_width` is deprecated (removal after 2025-12-31).
        # Use `width='stretch'` for the same behavior.
        st.image(str(path), caption=label, width=width_mode)
    else:
        st.warning(f"Plot not found: `{path.name}`")


def _render_plot_single_row(run_dir: Path, fname: str, label: str) -> None:
    path = run_dir / fname
    if not path.exists():
        st.warning(f"Plot not found: `{fname}`")
        return

    # Render most plots at half-width (centered). Keep selected ones full-width.
    if fname in _FULL_WIDTH_PLOTS:
        _render_plot(path, label, width_mode="stretch")
    else:
        left, mid, right = st.columns([2, 6, 2])
        with mid:
            _render_plot(path, label, width_mode="stretch")


def _render_plot_grid(run_dir: Path, plots: list[tuple[str, str]], columns: int = 2) -> None:
    available = [(f, lbl) for f, lbl in plots if (run_dir / f).exists()]
    missing = [(f, lbl) for f, lbl in plots if not (run_dir / f).exists()]

    if not available:
        st.info("No plot files found for this group.")
        return

    # Render every plot full-width, one per row (no extra section labels).
    for fname, lbl in available:
        _render_plot_single_row(run_dir, fname, lbl)

    if missing:
        st.caption("Missing: " + ", ".join(f"`{f}`" for f, _ in missing))


# ── Page entry-point ──────────────────────────────────────────────────────────

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

        /* Reduce st.metric VALUE font size (Hero mini-band). */
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] > div {
            font-size: 1.25rem;
            line-height: 1.1;
        }
        </style>
        <div class="page2-subheader">🔅 Exploratory Data Analysis: Mining-Ready Data</div>
        """,
        unsafe_allow_html=True,
    )

    # ── Run selector ─────────────────────────────────────────────────────────
    runs_root = EDA_RUNS_ROOT
    run_dirs = _list_run_dirs(runs_root)
    if not run_dirs and EDA_RUNS_ROOT_FALLBACK != runs_root:
        runs_root = EDA_RUNS_ROOT_FALLBACK
        run_dirs = _list_run_dirs(runs_root)

    if not run_dirs:
        st.error(f"No `feature_*` directories found under: {runs_root}")
        st.info(
            "This dashboard page renders *precomputed* EDA artifacts. "
            "Generate `data/eda/feature_<run_id>/` (older runs may exist under `eda/output/feature_<run_id>/`)."
        )
        return

    run_names = [r.name for r in run_dirs]
    label_col, select_col, cap_col = st.columns([2, 3, 5])
    with label_col:
        st.markdown('<div style="white-space: nowrap; font-weight: 600;">Select EDA run:</div>',
                    unsafe_allow_html=True)
    with select_col:
        selected_run = st.selectbox(
            "Feature EDA run",
            options=run_names,
            index=0,
            help="Latest run is selected by default",
            label_visibility="collapsed",
        )
    with cap_col:
        st.caption("This page displays outputs generated by `Method_Ready_Data_EDA.ipynb`.")

    run_dir = runs_root / selected_run
    summary_csv = run_dir / "Method_Ready_Data_EDA_summary.csv"
    metrics_json = run_dir / "Method_Ready_Data_EDA_metrics.json"
    findings_md = run_dir / "15_key_results_and_interpretation.md"
    dq_md = run_dir / "16_data_quality_report.md"
    skewness_csv = run_dir / "skewness.csv"
    similarity_json = run_dir / "similarity_distance_foundation.json"
    similarity_readme = run_dir / "similarity_distance_README.md"

    metrics: dict[str, Any] = {}
    if metrics_json.exists():
        try:
            metrics = _read_json(str(metrics_json))
        except Exception as e:
            st.warning(f"Could not read metrics JSON: `{metrics_json.name}` ({e})")

    users_val = metrics.get("user_features_train", {}).get("rows")
    movies_val = metrics.get("movie_features_train", {}).get("rows")
    baskets_val = metrics.get("transactions_train", {}).get("rows")

    neg_pol = metrics.get("polarization_score_negatives")
    total_tokens = metrics.get("transactions_total_tokens")
    token_counts = metrics.get("token_counts") if isinstance(metrics.get("token_counts"), list) else []

    # ── Hero mini-band ────────────────────────────────────────────────────────
    metric_cols = st.columns(4)
    metric_cols[0].metric("Users",   _fmt_int(users_val) if users_val is not None else "—", help="Rows in user_features_train")
    metric_cols[1].metric("Movies",  _fmt_int(movies_val) if movies_val is not None else "—", help="Rows in movie_features_train")
    metric_cols[2].metric("Baskets", _fmt_int(baskets_val) if baskets_val is not None else "—", help="Rows in transactions_train")
    if neg_pol is None:
        metric_cols[3].metric("DQ Issues", "—", help="Populate by running the notebook export step")
    else:
        metric_cols[3].metric("DQ Issues", f"polarization<0: {neg_pol}", help="Key consistency check")

    mini_cols = st.columns(3)
    mini_specs = [
        ("user_activity_lorenz.png",  "User activity inequality (Lorenz)"),
        ("movie_controversy_map.png", "Movie controversy map"),
        ("rules_support_vs_lift.png", "Rules galaxy: support vs lift"),
    ]
    for col, (fname, label) in zip(mini_cols, mini_specs):
        with col:
            st.markdown(f"**{label}**")
            _render_plot(run_dir / fname, label, width_mode="stretch")

    # ── 1) Inventory tables (expanded by default) ─────────────────────────────
    with st.expander("1) Inventory tables", expanded=True):
        if summary_csv.exists():
            st.dataframe(_read_csv(str(summary_csv)), width="stretch")
        else:
            st.warning(f"`{summary_csv.name}` not found in `{run_dir.name}/` — run the notebook first.")

        if metrics:
            st.markdown("**Cross-table integrity**")
            user_cov = metrics.get("user_coverage") if isinstance(metrics.get("user_coverage"), dict) else {}
            unresolved = metrics.get("unresolved_movie_tokens")
            uniq_movie_tokens = metrics.get("unique_movie_tokens_in_transactions")
            cross_rows: list[dict[str, Any]] = []
            if user_cov:
                cross_rows.extend(
                    [
                        {"Check": "Users in both (features ∩ transactions)", "Result": _fmt_int(user_cov.get("users_in_both", "—"))},
                        {"Check": "Users in features only (no basket)", "Result": _fmt_int(user_cov.get("users_in_features_only", "—"))},
                        {"Check": "Users in transactions only", "Result": _fmt_int(user_cov.get("users_in_transactions_only", "—"))},
                    ]
                )
            if uniq_movie_tokens is not None:
                cross_rows.append({"Check": "Unique movie: tokens in transactions", "Result": _fmt_int(uniq_movie_tokens)})
            if unresolved is not None:
                cross_rows.append({"Check": "Unresolved movie: tokens", "Result": _fmt_int(unresolved)})
            if cross_rows:
                st.dataframe(pd.DataFrame(cross_rows), width="stretch", hide_index=True)

            st.markdown("**Token composition in `transactions_train`**")
            if token_counts:
                df_tokens = pd.DataFrame(token_counts)
                if total_tokens:
                    st.caption(f"Total tokens: {_fmt_int(total_tokens)}")
                st.dataframe(df_tokens, width="stretch", hide_index=True)
            else:
                st.info("Token composition not available for this run (missing metrics export).")
        else:
            st.info("Run the notebook export step to populate integrity + token-composition tables.")

    # ── 2) Sample files (collapsed by default) ────────────────────────────────
    with st.expander("2) Sample files (6-row CSV exports)", expanded=False):
        samples_dir = run_dir / "samples"
        if not samples_dir.exists():
            st.info("No `samples/` directory found for this run. Re-run the notebook to generate sample CSVs.")
        else:
            sample_paths = sorted(samples_dir.glob("sample_*.csv"))
            if not sample_paths:
                st.info("No `sample_*.csv` files found in `samples/`.")
            else:
                st.caption("Showing first 6 rows for each exported sample CSV.")
                for path in sample_paths:
                    st.markdown(f"**{path.name}**")
                    try:
                        df = _read_csv(str(path))
                        st.dataframe(df.head(6), width="stretch")
                    except Exception as e:
                        st.warning(f"Could not read `{path.name}`: {e}")

    # ── 3) Plots by table (expanded by default) ───────────────────────────────
    for group_name, plots in _PLOT_GROUPS.items():
        with st.expander(f"3) {group_name} — plots", expanded=True):
            _render_plot_grid(run_dir, plots, columns=2)

    # ── 4) Findings ───────────────────────────────────────────────────────────
    with st.expander("4) Findings & interpretation", expanded=False):
        if findings_md.exists():
            st.markdown(_read_text(str(findings_md)))
        else:
            st.markdown(_DEFAULT_FINDINGS_MD)

    # ── 5) DQ Report ─────────────────────────────────────────────────────────
    with st.expander("5) Data Quality Report (§16)", expanded=False):
        if dq_md.exists():
            st.markdown(_read_text(str(dq_md)))
        else:
            st.markdown(_DEFAULT_DQ_MD)

    # ── 6) Skewness & similarity/distance (new notebook outputs) ─────────────
    with st.expander("6) Skewness & similarity/distance (exports)", expanded=False):
        left, right = st.columns(2)

        with left:
            st.markdown("**Skewness (CSV)**")
            if skewness_csv.exists():
                try:
                    st.dataframe(_read_csv(str(skewness_csv)), width="stretch", hide_index=True)
                except Exception as e:
                    st.warning(f"Could not read `{skewness_csv.name}`: {e}")
            else:
                st.info("No `skewness.csv` found for this run.")

        with right:
            st.markdown("**Similarity / distance baseline (JSON)**")
            if similarity_json.exists():
                try:
                    st.json(_read_json(str(similarity_json)), expanded=False)
                except Exception as e:
                    st.warning(f"Could not read `{similarity_json.name}`: {e}")
            else:
                st.info("No `similarity_distance_foundation.json` found for this run.")

        if similarity_readme.exists():
            st.markdown("**How to interpret these similarity outputs**")
            st.markdown(_read_text(str(similarity_readme)))
        else:
            st.caption("No `similarity_distance_README.md` found for this run.")

    st.caption(f"Page 2 — run: `{selected_run}` · outputs from `{runs_root / selected_run}`")
