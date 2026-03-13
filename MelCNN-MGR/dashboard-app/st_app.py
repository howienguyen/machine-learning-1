from pathlib import Path

import streamlit as st

import demo_apps.eda_dashboard.page_raw_data_eda as page_raw_data_eda
import page_preprocessed_eda
import demo_apps.eda_dashboard.page_mining_ready_eda as page_mining_ready_eda
import demo_apps.eda_dashboard.page_storyb1_eval as page_storyb1_eval


ICON_PATH = Path(__file__).parent / "app_icon.png"

st.set_page_config(
    page_title="Music Genre Prediction Dashboard",
    # page_icon=str(ICON_PATH) if ICON_PATH.exists() else "🎬",
    page_icon = "🎬",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stAppViewBlockContainer {
            padding-top: 0rem;
            padding-bottom: 0rem;
        }
        header.stAppHeader {
            background-color: transparent;
        }
        section.stMain .block-container {
            padding-top: 0.35rem;
            padding-bottom: 0.35rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        section.stMain .block-container > div {
            row-gap: 0.9rem;
        }
        h1, h2, h3, p {
            margin-top: 0.2rem !important;
            margin-bottom: 0.35rem !important;
        }
        div[data-testid="stMetric"] {
            padding-top: 0.15rem;
            padding-bottom: 0.15rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Music Genre Prediction Dashboard", text_alignment="center")

pages = [
        st.Page(mfcc_vs_logmel.render, title="MFCC vs Log-Mel", icon="🔅", url_path="mfcc-vs-logmel"),
        st.Page(page_eda.render, title="EDA", icon="🔅", url_path="eda"),
        st.Page(page_model_training_n_evaluation_reports.render, title="Training Report & Evaluation", icon="🔅", url_path="report"),
    ]

navigation = st.navigation(pages)

st.sidebar.markdown(
    """<small><span style="color:#c9c9c9;">🪶 Nguyễn Sỹ Hùng, 2026
    <br/>(Đây chưa phải Dashboard chính thức của nhóm. Trang này hiện chỉ mang tính hỗ trợ giúp hiểu về bộ dataset gốc)
    </span></small>""", 
    unsafe_allow_html=True
)

with st.spinner("Please wait..."):
    navigation.run()


