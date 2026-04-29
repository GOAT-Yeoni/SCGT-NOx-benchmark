"""
SCGT NOx Control Benchmark Tool
================================
단순사이클 가스터빈(SCGT) 배기가스 NOx 제어 기술 비교 툴.

비교 대상:
  1) Air Tempering + 저온 SCR (V2O5-WO3/TiO2, 280~420°C)
  2) 고온 SCR 직접 운전 (Cu-zeolite / Fe-zeolite / Cu-SAPO-34, 400~600°C)

데이터 출처:
  - EPA RBLC (BACT/LAER 데이터베이스)
  - EPA AP-42 §3.1 (Stationary Gas Turbines)
  - EPA Cost Manual Ch.2 (SCR)
  - EPRI 3002022688 / 3002030747 / 3002030748
  - GE LM6000/LMS100, Siemens SGT-A65, MHI 공개 사양

실행: streamlit run app.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ============================================================================
# 페이지 설정
# ============================================================================
st.set_page_config(
    page_title="SCGT NOx Control Benchmark",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 논문 스타일에 맞춰 미니멀 라이트 테마 + 폰트 통일
st.markdown(
    """
    <style>
    .stApp { background-color: #fafaf7; color: #1a1a1a; }
    h1, h2, h3, h4 {
        font-family: 'Times New Roman', Times, serif !important;
        color: #1a1a1a;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #d4d4d4;
        border-radius: 4px;
        padding: 14px 18px;
        margin: 6px 0;
        font-family: 'Times New Roman', Times, serif;
    }
    .metric-card h4 {
        color: #555;
        font-size: 12px;
        margin: 0 0 6px 0;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .metric-card .value {
        color: #0f172a;
        font-size: 26px;
        font-weight: 700;
    }
    .metric-card .delta-good { color: #15803d; font-size: 13px; }
    .metric-card .delta-bad  { color: #b91c1c; font-size: 13px; }
    .metric-card .delta-neutral { color: #555; font-size: 13px; }
    div[data-testid="stSidebar"] { background-color: #f1efe8; }

    /* ── 탭: 클릭 가능한 카드 스타일 ───────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 14px;
        background-color: transparent;
        padding: 16px 4px 12px 4px;
        border-bottom: 2px solid #d4d4d4;
        margin-bottom: 14px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff !important;
        color: #1f2937 !important;
        padding: 14px 28px !important;
        height: auto !important;
        min-width: 130px;
        border: 1.5px solid #c4bfb1 !important;
        border-radius: 8px !important;
        font-family: 'Times New Roman', Times, serif !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        letter-spacing: 0.4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        cursor: pointer !important;
        transition: all 0.15s ease-in-out;
    }
    .stTabs [data-baseweb="tab"] p {
        font-size: 16px !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #faf8f0 !important;
        border-color: #4b5563 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.10);
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f2937 !important;
        color: #ffffff !important;
        border-color: #1f2937 !important;
        box-shadow: 0 3px 8px rgba(31,41,55,0.30);
        transform: translateY(-1px);
    }
    .stTabs [aria-selected="true"] p { color: #ffffff !important; }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] { display: none !important; }

    /* ── 인트로 박스 ──────────────────────────────────────────────── */
    .intro-box {
        background-color: #ffffff;
        border-left: 5px solid #1f2937;
        border-radius: 4px;
        padding: 18px 24px;
        margin: 14px 0 18px 0;
        font-family: 'Times New Roman', Times, serif;
        line-height: 1.65;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .intro-box .obj-title {
        font-size: 18px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 8px;
    }
    .intro-box .obj-row {
        display: flex;
        gap: 24px;
        margin-top: 12px;
    }
    .intro-box .obj-col {
        flex: 1;
    }
    .intro-box .obj-col h5 {
        color: #b91c1c;
        font-size: 14px;
        font-weight: 700;
        margin: 0 0 4px 0;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .intro-box .obj-col.target h5 { color: #15803d; }
    .intro-box ul { margin: 4px 0 0 0; padding-left: 18px; font-size: 14px; }
    .intro-box li { margin: 2px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# 학술지 표준에 가까운 색 팔레트 (matplotlib tab10 + 보강)
COLOR = {
    "VTi":      "#1f77b4",  # V2O5-WO3/TiO2 (저온, 청)
    "Cu_zeo":   "#d62728",  # Cu-SSZ-13 (적)
    "Fe_zeo":   "#2ca02c",  # Fe-zeolite (녹)
    "Cu_SAPO":  "#9467bd",  # Cu-SAPO-34 (자)
    "LT_case":  "#1f77b4",  # 저온+AT (청)
    "HT_case":  "#ff7f0e",  # 고온 (주)
    "axis":     "#1a1a1a",
    "grid":     "#d4d4d4",
    "text":     "#1a1a1a",
    "ref":      "#6b7280",
}


def paper_style(fig: go.Figure, height: int = 400, has_axes: bool = True,
                 title: str = "") -> go.Figure:
    """모든 차트에 일관된 논문 스타일 적용. title="" 로 'undefined' 헤더 제거."""
    fig.update_layout(
        title=dict(
            text=title,  # 빈 문자열 → 'undefined' 표시 차단
            font=dict(family="Times New Roman, Times, serif", size=15, color=COLOR["text"]),
            x=0.5, xanchor="center", y=0.97, yanchor="top",
        ),
        template="simple_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Times New Roman, Times, serif", size=13, color=COLOR["text"]),
        height=height,
        margin=dict(l=72, r=30, t=70 if title else 50, b=60),
        legend=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=COLOR["axis"],
            borderwidth=1,
            font=dict(family="Times New Roman, Times, serif", size=12, color=COLOR["text"]),
        ),
    )
    if has_axes:
        fig.update_xaxes(
            showline=True, linewidth=1.4, linecolor=COLOR["axis"],
            mirror=True, ticks="outside", tickwidth=1.2, tickcolor=COLOR["axis"],
            ticklen=5, gridcolor=COLOR["grid"], griddash="dot", zeroline=False,
            title_font=dict(family="Times New Roman, Times, serif", size=14, color=COLOR["text"]),
            tickfont=dict(family="Times New Roman, Times, serif", size=12, color=COLOR["text"]),
        )
        fig.update_yaxes(
            showline=True, linewidth=1.4, linecolor=COLOR["axis"],
            mirror=True, ticks="outside", tickwidth=1.2, tickcolor=COLOR["axis"],
            ticklen=5, gridcolor=COLOR["grid"], griddash="dot", zeroline=False,
            title_font=dict(family="Times New Roman, Times, serif", size=14, color=COLOR["text"]),
            tickfont=dict(family="Times New Roman, Times, serif", size=12, color=COLOR["text"]),
        )
    return fig


# ============================================================================
# LIT (Literature) 데이터 - 촉매 베이스별 분류
# catalyst_base: V/Ti | Cu-zeolite | Fe-zeolite | Cu-SAPO-34
# ============================================================================
LIT_RBLC: List[Dict] = [
    # ── V/Ti 저온 SCR + Air Tempering ──────────────────────────────────────
    {"facility": "Lodi Energy Center", "turbine": "GE LM5000", "mode": "SC",
     "tech_category": "저온 SCR + AT", "catalyst_base": "V/Ti",
     "catalyst_detail": "V₂O₅-WO₃/TiO₂",
     "exh_T_C": 525, "scr_T_C": 350,
     "outlet_NOx_ppmvd_15O2": 3.0, "NOx_red_pct": 88.0, "NH3_slip_ppm": 5.0,
     "year": 2008, "source": "EPA RBLC (CA)"},

    {"facility": "Live Oak Ltd", "turbine": "GE Frame 6", "mode": "SC",
     "tech_category": "저온 SCR + AT", "catalyst_base": "V/Ti",
     "catalyst_detail": "V₂O₅-WO₃/TiO₂",
     "exh_T_C": 545, "scr_T_C": 360,
     "outlet_NOx_ppmvd_15O2": 2.5, "NOx_red_pct": 90.0, "NH3_slip_ppm": 5.0,
     "year": 2000, "source": "BAAQMD permit / RBLC"},

    {"facility": "Russell City Energy", "turbine": "Siemens SGT6-5000F", "mode": "CC",
     "tech_category": "저온 SCR (HRSG)", "catalyst_base": "V/Ti",
     "catalyst_detail": "V₂O₅-WO₃/TiO₂",
     "exh_T_C": 605, "scr_T_C": 360,
     "outlet_NOx_ppmvd_15O2": 2.0, "NOx_red_pct": 92.0, "NH3_slip_ppm": 5.0,
     "year": 2013, "source": "EPA RBLC (LAER)"},

    {"facility": "Marsh Landing", "turbine": "GE LMS100", "mode": "SC",
     "tech_category": "저온 SCR + AT", "catalyst_base": "V/Ti",
     "catalyst_detail": "V₂O₅-WO₃/TiO₂",
     "exh_T_C": 415, "scr_T_C": 350,
     "outlet_NOx_ppmvd_15O2": 2.5, "NOx_red_pct": 90.0, "NH3_slip_ppm": 5.0,
     "year": 2013, "source": "EPA RBLC"},

    {"facility": "Walnut Energy", "turbine": "GE LM6000PF", "mode": "SC",
     "tech_category": "저온 SCR + AT", "catalyst_base": "V/Ti",
     "catalyst_detail": "V₂O₅-WO₃/TiO₂",
     "exh_T_C": 480, "scr_T_C": 360,
     "outlet_NOx_ppmvd_15O2": 5.0, "NOx_red_pct": 80.0, "NH3_slip_ppm": 10.0,
     "year": 2010, "source": "EPA RBLC"},

    {"facility": "Inland Empire", "turbine": "MHI 501G", "mode": "SC",
     "tech_category": "저온 SCR + AT", "catalyst_base": "V/Ti",
     "catalyst_detail": "V₂O₅-WO₃/TiO₂",
     "exh_T_C": 580, "scr_T_C": 360,
     "outlet_NOx_ppmvd_15O2": 2.5, "NOx_red_pct": 90.0, "NH3_slip_ppm": 5.0,
     "year": 2012, "source": "EPA RBLC (LAER)"},

    {"facility": "Hanford Peaker", "turbine": "GE LM6000PG", "mode": "SC",
     "tech_category": "저온 SCR + AT", "catalyst_base": "V/Ti",
     "catalyst_detail": "V₂O₅-WO₃/TiO₂",
     "exh_T_C": 470, "scr_T_C": 350,
     "outlet_NOx_ppmvd_15O2": 2.0, "NOx_red_pct": 92.0, "NH3_slip_ppm": 5.0,
     "year": 2018, "source": "EPA RBLC (CA)"},

    {"facility": "Pio Pico", "turbine": "GE LMS100", "mode": "SC",
     "tech_category": "저온 SCR + AT", "catalyst_base": "V/Ti",
     "catalyst_detail": "V₂O₅-WO₃/TiO₂",
     "exh_T_C": 420, "scr_T_C": 360,
     "outlet_NOx_ppmvd_15O2": 2.5, "NOx_red_pct": 90.0, "NH3_slip_ppm": 5.0,
     "year": 2014, "source": "EPA RBLC"},

    # ── Cu-zeolite 고온 SCR ─────────────────────────────────────────────────
    {"facility": "EPRI Pilot", "turbine": "Frame 7FA (test)", "mode": "SC",
     "tech_category": "고온 SCR 직접", "catalyst_base": "Cu-zeolite",
     "catalyst_detail": "Cu-SSZ-13",
     "exh_T_C": 540, "scr_T_C": 540,
     "outlet_NOx_ppmvd_15O2": 2.5, "NOx_red_pct": 90.0, "NH3_slip_ppm": 4.0,
     "year": 2021, "source": "EPRI 3002022688"},

    {"facility": "EPRI Demo Bed-A", "turbine": "Frame 7EA", "mode": "SC",
     "tech_category": "고온 SCR 직접", "catalyst_base": "Cu-zeolite",
     "catalyst_detail": "Cu-Beta",
     "exh_T_C": 510, "scr_T_C": 510,
     "outlet_NOx_ppmvd_15O2": 3.5, "NOx_red_pct": 86.0, "NH3_slip_ppm": 6.0,
     "year": 2020, "source": "EPRI 3002022688"},

    {"facility": "Cordova Energy", "turbine": "GE LM6000", "mode": "SC",
     "tech_category": "고온 SCR 직접", "catalyst_base": "Cu-zeolite",
     "catalyst_detail": "Cu-CHA",
     "exh_T_C": 460, "scr_T_C": 460,
     "outlet_NOx_ppmvd_15O2": 5.0, "NOx_red_pct": 85.0, "NH3_slip_ppm": 8.0,
     "year": 2020, "source": "EPRI 사례연구"},

    # ── Cu-SAPO-34 고온 SCR ─────────────────────────────────────────────────
    {"facility": "EPRI Advanced HT-SCR", "turbine": "F-class", "mode": "SC",
     "tech_category": "고온 SCR 직접", "catalyst_base": "Cu-SAPO-34",
     "catalyst_detail": "Cu-SAPO-34",
     "exh_T_C": 550, "scr_T_C": 550,
     "outlet_NOx_ppmvd_15O2": 2.0, "NOx_red_pct": 92.0, "NH3_slip_ppm": 3.0,
     "year": 2022, "source": "EPRI 3002030748"},

    {"facility": "MHI Test Bench", "turbine": "H-25", "mode": "SC",
     "tech_category": "고온 SCR 직접", "catalyst_base": "Cu-SAPO-34",
     "catalyst_detail": "Cu-SAPO-34 washcoat",
     "exh_T_C": 555, "scr_T_C": 555,
     "outlet_NOx_ppmvd_15O2": 3.0, "NOx_red_pct": 88.0, "NH3_slip_ppm": 4.5,
     "year": 2023, "source": "MHI 기술보고서"},

    # ── Fe-zeolite 고온 SCR ─────────────────────────────────────────────────
    {"facility": "EPRI Demo Bed-B", "turbine": "Frame 7EA", "mode": "SC",
     "tech_category": "고온 SCR 직접", "catalyst_base": "Fe-zeolite",
     "catalyst_detail": "Fe-제올라이트",
     "exh_T_C": 500, "scr_T_C": 500,
     "outlet_NOx_ppmvd_15O2": 3.0, "NOx_red_pct": 88.0, "NH3_slip_ppm": 5.0,
     "year": 2019, "source": "EPRI 3002030747"},

    {"facility": "Univ Stuttgart Pilot", "turbine": "Test rig", "mode": "SC",
     "tech_category": "고온 SCR 직접", "catalyst_base": "Fe-zeolite",
     "catalyst_detail": "Fe-Beta",
     "exh_T_C": 520, "scr_T_C": 520,
     "outlet_NOx_ppmvd_15O2": 3.5, "NOx_red_pct": 87.0, "NH3_slip_ppm": 5.5,
     "year": 2021, "source": "Catalysts MDPI 2021"},

    {"facility": "EPRI Field Demo", "turbine": "LM6000", "mode": "SC",
     "tech_category": "고온 SCR 직접", "catalyst_base": "Fe-zeolite",
     "catalyst_detail": "Fe-ZSM-5",
     "exh_T_C": 480, "scr_T_C": 480,
     "outlet_NOx_ppmvd_15O2": 4.0, "NOx_red_pct": 85.0, "NH3_slip_ppm": 6.5,
     "year": 2018, "source": "EPRI 사례"},

    {"facility": "Siemens 시험장", "turbine": "SGT-A65", "mode": "SC",
     "tech_category": "고온 SCR 직접", "catalyst_base": "Fe-zeolite",
     "catalyst_detail": "Fe-SSZ-13",
     "exh_T_C": 470, "scr_T_C": 470,
     "outlet_NOx_ppmvd_15O2": 3.0, "NOx_red_pct": 88.0, "NH3_slip_ppm": 5.0,
     "year": 2022, "source": "Siemens 기술보고서"},
]

# ============================================================================
# 제조사 × 촉매 제품 데이터 (Manufacturer Product Lineup)
# 출처: 제조사 공개 카탈로그 + EPRI 3002022688 + 학술 리뷰
# ============================================================================
LIT_MFR: List[Dict] = [
    # ── Cormetech (US) — GT SCR 시장 리더
    {"manufacturer": "Cormetech", "country": "USA",
     "product": "Honeycomb V-Ti", "chemistry": "V₂O₅-WO₃/TiO₂",
     "catalyst_base": "V/Ti", "form": "Honeycomb",
     "T_min": 290, "T_max": 420, "T_opt": 360,
     "eff_max_pct": 95.0, "slip_ppm": 5.0,
     "market": "GT BACT", "ref": "Cormetech 카탈로그 / EPRI 3002022688"},
    {"manufacturer": "Cormetech", "country": "USA",
     "product": "Plate V-Ti (Coal)", "chemistry": "V₂O₅-WO₃/TiO₂",
     "catalyst_base": "V/Ti", "form": "Plate",
     "T_min": 320, "T_max": 400, "T_opt": 370,
     "eff_max_pct": 92.0, "slip_ppm": 3.0,
     "market": "Coal utility", "ref": "Cormetech 카탈로그"},
    # ── Haldor Topsoe (Denmark) — DNX 시리즈
    {"manufacturer": "Haldor Topsoe", "country": "Denmark",
     "product": "DNX (V-Mo-W/Ti)", "chemistry": "V₂O₅-MoO₃-WO₃/TiO₂",
     "catalyst_base": "V/Ti", "form": "Honeycomb/Plate",
     "T_min": 280, "T_max": 420, "T_opt": 360,
     "eff_max_pct": 96.0, "slip_ppm": 3.0,
     "market": "Utility / GT", "ref": "Topsoe DNX brochure"},
    {"manufacturer": "Haldor Topsoe", "country": "Denmark",
     "product": "DNX-LT (저온형)", "chemistry": "V-Mo/Ti 강화",
     "catalyst_base": "V/Ti (LT)", "form": "Honeycomb",
     "T_min": 200, "T_max": 320, "T_opt": 260,
     "eff_max_pct": 88.0, "slip_ppm": 5.0,
     "market": "Tail-end SCR", "ref": "Topsoe LT-SCR 보고서"},
    # ── Johnson Matthey (UK) — VOC-CAT / NOX-CAT
    {"manufacturer": "Johnson Matthey", "country": "UK",
     "product": "NOXCAT (V-Ti)", "chemistry": "V₂O₅-WO₃/TiO₂",
     "catalyst_base": "V/Ti", "form": "Honeycomb",
     "T_min": 290, "T_max": 420, "T_opt": 365,
     "eff_max_pct": 94.0, "slip_ppm": 5.0,
     "market": "GT / Utility", "ref": "JM 제품 사양서"},
    {"manufacturer": "Johnson Matthey", "country": "UK",
     "product": "Cu-CHA (모바일·고정)", "chemistry": "Cu-SSZ-13 (Cu-CHA)",
     "catalyst_base": "Cu-zeolite", "form": "Washcoat / Honeycomb",
     "T_min": 200, "T_max": 550, "T_opt": 350,
     "eff_max_pct": 95.0, "slip_ppm": 4.0,
     "market": "Diesel + Stationary", "ref": "JM ACS 논문 2019"},
    # ── BASF — Premair 시리즈
    {"manufacturer": "BASF", "country": "Germany",
     "product": "Premair NXT (Cu-CHA)", "chemistry": "Cu-SAPO-34 / Cu-SSZ-13",
     "catalyst_base": "Cu-SAPO-34", "form": "Honeycomb",
     "T_min": 200, "T_max": 580, "T_opt": 400,
     "eff_max_pct": 95.0, "slip_ppm": 3.0,
     "market": "Diesel HD / 고온 SCR", "ref": "BASF 제품 카탈로그"},
    {"manufacturer": "BASF", "country": "Germany",
     "product": "Cu-Beta (저온)", "chemistry": "Cu-Beta zeolite",
     "catalyst_base": "Cu-zeolite", "form": "Honeycomb",
     "T_min": 220, "T_max": 500, "T_opt": 350,
     "eff_max_pct": 92.0, "slip_ppm": 5.0,
     "market": "Industrial low-temp", "ref": "BASF 사양서"},
    # ── Umicore — zeolite
    {"manufacturer": "Umicore", "country": "Belgium",
     "product": "Cu-SSZ-13", "chemistry": "Cu-SSZ-13",
     "catalyst_base": "Cu-zeolite", "form": "Washcoat",
     "T_min": 200, "T_max": 550, "T_opt": 380,
     "eff_max_pct": 94.0, "slip_ppm": 4.5,
     "market": "Mobile + GT pilot", "ref": "Umicore 보고서"},
    {"manufacturer": "Umicore", "country": "Belgium",
     "product": "Fe-Beta", "chemistry": "Fe-Beta zeolite",
     "catalyst_base": "Fe-zeolite", "form": "Washcoat",
     "T_min": 300, "T_max": 600, "T_opt": 480,
     "eff_max_pct": 91.0, "slip_ppm": 6.0,
     "market": "고온 GT", "ref": "Catalysts 2021"},
    # ── Hitachi-Zosen (Japan)
    {"manufacturer": "Hitachi-Zosen", "country": "Japan",
     "product": "ENESEED HC", "chemistry": "V₂O₅-WO₃/TiO₂",
     "catalyst_base": "V/Ti", "form": "Plate",
     "T_min": 300, "T_max": 410, "T_opt": 360,
     "eff_max_pct": 93.0, "slip_ppm": 5.0,
     "market": "Asia GT / Utility", "ref": "Hitachi-Zosen 카탈로그"},
    # ── Babcock-Hitachi K.K. (BHK)
    {"manufacturer": "Babcock-Hitachi", "country": "Japan",
     "product": "BHK Plate V/Ti", "chemistry": "V₂O₅-WO₃/TiO₂",
     "catalyst_base": "V/Ti", "form": "Plate",
     "T_min": 290, "T_max": 410, "T_opt": 360,
     "eff_max_pct": 92.0, "slip_ppm": 5.0,
     "market": "Boiler / GT", "ref": "BHK 기술자료"},
    # ── Mitsubishi Power
    {"manufacturer": "Mitsubishi Power", "country": "Japan",
     "product": "MHI BMC V-Ti", "chemistry": "V₂O₅-WO₃/TiO₂ (개량)",
     "catalyst_base": "V/Ti", "form": "Honeycomb",
     "T_min": 290, "T_max": 430, "T_opt": 365,
     "eff_max_pct": 94.0, "slip_ppm": 5.0,
     "market": "GT / 한국·일본", "ref": "MHI 기술보고서"},
    {"manufacturer": "Mitsubishi Power", "country": "Japan",
     "product": "Cu-SAPO HT (시제품)", "chemistry": "Cu-SAPO-34",
     "catalyst_base": "Cu-SAPO-34", "form": "Washcoat",
     "T_min": 380, "T_max": 600, "T_opt": 510,
     "eff_max_pct": 95.0, "slip_ppm": 4.0,
     "market": "고온 SCR R&D", "ref": "MHI 2023 학회"},
    # ── Argillon (now JM) — 저온 특화
    {"manufacturer": "Argillon (JM)", "country": "Germany",
     "product": "LT-SCR (low-T)", "chemistry": "V-W-Mo/TiO₂ + 첨가제",
     "catalyst_base": "V/Ti (LT)", "form": "Honeycomb",
     "T_min": 180, "T_max": 280, "T_opt": 240,
     "eff_max_pct": 87.0, "slip_ppm": 5.0,
     "market": "Tail-end / 산업", "ref": "Argillon 보고서"},
    # ── 신흥 저온 촉매 (Mn 계)
    {"manufacturer": "Tianhe (CN)", "country": "China",
     "product": "MnOx-CeO₂", "chemistry": "MnOx-CeO₂",
     "catalyst_base": "Mn-CeOx (LT)", "form": "Honeycomb",
     "T_min": 150, "T_max": 280, "T_opt": 220,
     "eff_max_pct": 90.0, "slip_ppm": 8.0,
     "market": "저온 산업 R&D", "ref": "Appl. Catal. B 2022"},
    {"manufacturer": "POSCO Eco-Catalyst", "country": "Korea",
     "product": "Mn-V/Ti 저온형", "chemistry": "V-Mn-Ce/TiO₂",
     "catalyst_base": "V/Mn (LT)", "form": "Honeycomb",
     "T_min": 180, "T_max": 320, "T_opt": 250,
     "eff_max_pct": 89.0, "slip_ppm": 6.0,
     "market": "철강·시멘트 저온 SCR", "ref": "POSCO 기술자료"},
]

# ============================================================================
# 저온 SCR R&D 효율 트렌드 (operating-T vs efficiency, by year)
# 출처: Appl. Catal. B / Catalysts (MDPI) / EPA STAR / NASA TM 등
# ============================================================================
LT_SCR_RND: List[Dict] = [
    {"year": 1995, "tech": "V₂O₅-WO₃/TiO₂ (1세대)", "T_C": 350, "eff_pct": 90.0,
     "note": "당시 표준 SCR — air tempering 필수"},
    {"year": 2000, "tech": "V-W-Mo/TiO₂", "T_C": 320, "eff_pct": 92.0,
     "note": "Mo 첨가로 저온 활성 향상"},
    {"year": 2005, "tech": "Cu-Beta zeolite", "T_C": 300, "eff_pct": 88.0,
     "note": "제올라이트 도입 초기"},
    {"year": 2008, "tech": "Argillon LT-SCR", "T_C": 240, "eff_pct": 86.0,
     "note": "tail-end SCR 상용화"},
    {"year": 2012, "tech": "Cu-SSZ-13", "T_C": 280, "eff_pct": 92.0,
     "note": "HD diesel 채택, 광범위 운전창"},
    {"year": 2014, "tech": "MnOx-CeO₂ (lab)", "T_C": 200, "eff_pct": 88.0,
     "note": "저온 R&D 진입"},
    {"year": 2017, "tech": "Cu-SAPO-34", "T_C": 250, "eff_pct": 93.0,
     "note": "더 넓은 윈도우, 열안정성↑"},
    {"year": 2019, "tech": "V-Mn-Ce/TiO₂", "T_C": 220, "eff_pct": 90.0,
     "note": "Mn 도핑으로 저온 활성"},
    {"year": 2020, "tech": "MnOx-CeO₂ (pilot)", "T_C": 180, "eff_pct": 89.0,
     "note": "pilot 검증"},
    {"year": 2021, "tech": "Cu-CHA + 첨가제", "T_C": 230, "eff_pct": 94.0,
     "note": "BASF Premair NXT 라인"},
    {"year": 2022, "tech": "Fe-Cu 이중 활성", "T_C": 260, "eff_pct": 92.0,
     "note": "광역 운전창 ZSM-5 기반"},
    {"year": 2023, "tech": "MnOx-Fe-CeO₂", "T_C": 170, "eff_pct": 91.0,
     "note": "초저온 SO₂ 내성 강화"},
    {"year": 2024, "tech": "Cu-SSZ-13 + 신소재", "T_C": 200, "eff_pct": 95.0,
     "note": "Umicore/JM 차세대"},
    {"year": 2025, "tech": "VOx-WOx/TiO₂ 나노쉘", "T_C": 240, "eff_pct": 94.0,
     "note": "코어-쉘 구조로 활성↑·SO₂↓"},
]

# AP-42 §3.1 미통제 NOx 배출계수 (lb/MMBtu)
AP42_NOX_FACTORS = {
    "Uncontrolled (구형)": 0.32,
    "DLN/DLE (저감연소기)": 0.099,
    "Water/Steam injection": 0.13,
    "DLN + SCR (BACT)": 0.012,
}

# 가스터빈 제조사 공개 사양 (대표값)
# - exh_T_C: 배기 온도 (°C)
# - exh_kg_s: 배기 유량 (kg/s)
# - NOx_DLN_ppm: DLN 연소기 출구 NOx (ppmvd@15%O2)
GT_TYPICAL = {
    "범용 평균값":       {"power_MW": 100, "exh_T_C": 550, "exh_kg_s": 250, "NOx_DLN_ppm": 25},
    "GE LM6000PG":      {"power_MW": 52,  "exh_T_C": 480, "exh_kg_s": 137, "NOx_DLN_ppm": 25},
    "GE LMS100":        {"power_MW": 105, "exh_T_C": 415, "exh_kg_s": 220, "NOx_DLN_ppm": 25},
    "Siemens SGT-A65":  {"power_MW": 64,  "exh_T_C": 470, "exh_kg_s": 174, "NOx_DLN_ppm": 25},
    "MHI H-25":         {"power_MW": 41,  "exh_T_C": 555, "exh_kg_s": 124, "NOx_DLN_ppm": 25},
    "GE 7E.03 (Frame 7E)": {"power_MW": 91, "exh_T_C": 545, "exh_kg_s": 305, "NOx_DLN_ppm": 25},
}

# 입력 필드 기본값 (운전·재무·단가·SCR·CAPEX)
DEFAULT_NON_GT = {
    "NOx_out_ppm": 2.5,
    "annual_hours": 4000,
    "plant_life_yr": 20,
    "discount_rate_pct": 7.0,
    "nh3_USD_per_kg": 0.55,
    "elec_USD_per_kWh": 0.08,
    "steam_USD_per_t": 25.0,
    "cat_life_LT": 4.0,
    "cat_life_HT": 3.0,
    "cat_USD_LT": 9000.0,
    "cat_USD_HT": 22000.0,
    "GHSV_LT": 12000.0,
    "GHSV_HT": 9000.0,
    "alpha": 1.05,
    "scr_USD_per_m3": 35000.0,
    "air_temp_USD": 4500.0,
    "wh_USD": 380000.0,
    "include_wh_capex": False,
    "ht_catalyst_choice": "Cu-SSZ-13",
}


# ============================================================================
# 엔지니어링 계산 모델
# ============================================================================
@dataclass
class Inputs:
    exh_T_C: float = 550.0
    exh_kg_s: float = 250.0
    NOx_in_ppm: float = 25.0
    NOx_out_ppm: float = 2.5
    annual_hours: float = 4000.0
    plant_life_yr: int = 20
    discount_rate: float = 0.07
    nh3_USD_per_kg: float = 0.55
    elec_USD_per_kWh: float = 0.08
    steam_USD_per_t: float = 25.0
    catalyst_life_LT_yr: float = 4.0
    catalyst_life_HT_yr: float = 3.0
    catalyst_USD_per_m3_LT: float = 9000.0
    catalyst_USD_per_m3_HT: float = 22000.0
    LT_T_window: Tuple[float, float] = (280.0, 420.0)
    HT_T_window: Tuple[float, float] = (380.0, 600.0)
    LT_design_T: float = 350.0
    GHSV_LT: float = 12000.0
    GHSV_HT: float = 9000.0
    NH3_NOx_alpha: float = 1.05
    scr_USD_per_m3: float = 35000.0
    air_temp_USD_per_kg_s: float = 4500.0
    waste_heat_USD_per_MWth: float = 380000.0
    include_waste_heat_capex: bool = False
    ht_catalyst_base: str = "Cu-zeolite"   # Cu-zeolite | Fe-zeolite | Cu-SAPO-34


# 촉매 베이스별 운전 윈도우 + 최적점 + 최대 효율
CAT_WINDOWS = {
    "V/Ti":         {"window": (280, 420), "T_opt": 360, "sigma": 60,  "eff_max": 0.96},
    "Cu-zeolite":   {"window": (380, 600), "T_opt": 480, "sigma": 90,  "eff_max": 0.94},
    "Fe-zeolite":   {"window": (400, 600), "T_opt": 510, "sigma": 80,  "eff_max": 0.92},
    "Cu-SAPO-34":   {"window": (380, 600), "T_opt": 500, "sigma": 95,  "eff_max": 0.95},
}


def gas_volumetric_flow(exh_kg_s: float, T_C: float) -> float:
    M = 28.5e-3
    n_mol_s = exh_kg_s / M
    Nm3_s = n_mol_s * 22.414e-3
    return Nm3_s * 3600.0


def actual_volumetric_flow(exh_kg_s: float, T_C: float) -> float:
    Nm3_h = gas_volumetric_flow(exh_kg_s, T_C)
    return Nm3_h * (T_C + 273.15) / 273.15


def air_tempering(exh_T_C: float, exh_kg_s: float, target_T_C: float,
                   ambient_T_C: float = 25.0) -> Dict:
    if exh_T_C <= target_T_C:
        return {"air_kg_s": 0.0, "ratio": 0.0, "mixed_kg_s": exh_kg_s}
    air_kg_s = exh_kg_s * (exh_T_C - target_T_C) / (target_T_C - ambient_T_C)
    return {"air_kg_s": air_kg_s, "ratio": air_kg_s / exh_kg_s,
            "mixed_kg_s": exh_kg_s + air_kg_s}


def scr_efficiency_curve(T_C: float, catalyst_base: str) -> float:
    """촉매 베이스별 온도-효율 곡선 (Bell-shape).

    - 윈도우 내부: floor=78%, peak=eff_max (T_opt에서)
    - 윈도우 외부: floor=30%로 떨어지고 bell도 0.15배 감쇠
    """
    if catalyst_base not in CAT_WINDOWS:
        catalyst_base = "V/Ti"
    cw = CAT_WINDOWS[catalyst_base]
    T_opt, sigma = cw["T_opt"], cw["sigma"]
    T_lo, T_hi = cw["window"]
    eff_max = cw["eff_max"]

    bell = math.exp(-((T_C - T_opt) ** 2) / (2 * sigma ** 2))
    inside = T_lo <= T_C <= T_hi
    if not inside:
        bell *= 0.15
    floor = 0.78 if inside else 0.30
    eff = floor + (eff_max - floor) * bell
    return min(eff_max, max(0.20, eff))


def required_NH3_kg_s(exh_kg_s: float, NOx_in_ppm: float, NOx_out_ppm: float,
                      alpha: float) -> float:
    M_gas = 28.5e-3
    M_NH3 = 17.0e-3
    n_gas_mol_s = exh_kg_s / M_gas
    delta_NOx_ppm = max(0.0, NOx_in_ppm - NOx_out_ppm)
    n_NOx_mol_s = n_gas_mol_s * delta_NOx_ppm * 1e-6
    n_NH3_mol_s = n_NOx_mol_s * alpha
    return n_NH3_mol_s * M_NH3


def NH3_slip_ppm(alpha: float, target_eff: float) -> float:
    base = 1.5
    over = max(0.0, alpha - 1.0) * 30.0
    eff_pen = max(0.0, target_eff - 0.85) * 25.0
    return min(base + over + eff_pen, 12.0)


def reactor_volume_m3(actual_m3_h: float, GHSV_per_h: float) -> float:
    return actual_m3_h / GHSV_per_h


def waste_heat_recoverable_MWth(exh_T_C: float, target_T_C: float,
                                 exh_kg_s: float) -> float:
    cp = 1.10
    delta_T = max(0.0, exh_T_C - target_T_C)
    return exh_kg_s * cp * delta_T / 1000.0


def annualize(capex: float, life_yr: int, r: float) -> float:
    if r == 0:
        return capex / life_yr
    crf = (r * (1 + r) ** life_yr) / ((1 + r) ** life_yr - 1)
    return capex * crf


def compute_case(inp: Inputs, tech: str) -> Dict:
    """tech in {'LT','HT'}"""
    if tech == "LT":
        scr_T = inp.LT_design_T
        cat_base = "V/Ti"
        temp = air_tempering(inp.exh_T_C, inp.exh_kg_s, scr_T)
        air_kg_s = temp["air_kg_s"]
        scr_kg_s = temp["mixed_kg_s"]
        dilute = inp.exh_kg_s / scr_kg_s
        NOx_in_at_scr = inp.NOx_in_ppm * dilute
        NOx_out_at_scr = inp.NOx_out_ppm * dilute
    else:
        scr_T = inp.exh_T_C
        cat_base = inp.ht_catalyst_base
        air_kg_s = 0.0
        scr_kg_s = inp.exh_kg_s
        NOx_in_at_scr = inp.NOx_in_ppm
        NOx_out_at_scr = inp.NOx_out_ppm

    target_eff = max(0.0, 1.0 - NOx_out_at_scr / max(NOx_in_at_scr, 1e-6))
    eff_capability = scr_efficiency_curve(scr_T, cat_base)
    eff_actual = min(eff_capability, target_eff if target_eff > 0 else eff_capability)
    nh3_kg_s = required_NH3_kg_s(scr_kg_s, NOx_in_at_scr, NOx_out_at_scr, inp.NH3_NOx_alpha)
    slip = NH3_slip_ppm(inp.NH3_NOx_alpha, eff_actual)

    GHSV = inp.GHSV_LT if tech == "LT" else inp.GHSV_HT
    actual_m3_h = actual_volumetric_flow(scr_kg_s, scr_T)
    cat_vol = reactor_volume_m3(actual_m3_h, GHSV)

    scr_capex = cat_vol * inp.scr_USD_per_m3
    air_capex = air_kg_s * inp.air_temp_USD_per_kg_s if tech == "LT" else 0.0
    if tech == "HT":
        Q_MW = waste_heat_recoverable_MWth(inp.exh_T_C, 200.0, inp.exh_kg_s)
        wh_capex = (Q_MW * inp.waste_heat_USD_per_MWth) if inp.include_waste_heat_capex else 0.0
        wh_recoverable_MW = Q_MW
    else:
        wh_capex = 0.0
        wh_recoverable_MW = 0.0
    capex_total = scr_capex + air_capex + wh_capex

    annual_s = inp.annual_hours * 3600.0
    nh3_annual_kg = nh3_kg_s * annual_s
    nh3_cost = nh3_annual_kg * inp.nh3_USD_per_kg
    cat_life = inp.catalyst_life_LT_yr if tech == "LT" else inp.catalyst_life_HT_yr
    cat_unit = inp.catalyst_USD_per_m3_LT if tech == "LT" else inp.catalyst_USD_per_m3_HT
    cat_replacement_annual = (cat_vol * cat_unit) / cat_life
    if tech == "LT":
        blower_kW = air_kg_s * 1.5
        elec_cost = blower_kW * inp.annual_hours * inp.elec_USD_per_kWh
    else:
        elec_cost = 0.0
    if tech == "HT":
        steam_t_h = wh_recoverable_MW * 1000.0 / 2700.0 * 0.70
        steam_credit_raw = steam_t_h * inp.annual_hours * inp.steam_USD_per_t
    else:
        steam_credit_raw = 0.0
    dp_kW = (cat_vol ** 0.5) * 8.0
    dp_cost = dp_kW * inp.annual_hours * inp.elec_USD_per_kWh

    gross_opex = nh3_cost + cat_replacement_annual + elec_cost + dp_cost
    steam_credit = (min(steam_credit_raw, gross_opex)
                    if not inp.include_waste_heat_capex else steam_credit_raw)
    opex_annual = gross_opex - steam_credit

    M_gas = 28.5e-3
    M_NO2 = 46.0e-3
    n_gas_mol_s = inp.exh_kg_s / M_gas
    NOx_removed_kg_s = n_gas_mol_s * (inp.NOx_in_ppm - inp.NOx_out_ppm) * 1e-6 * M_NO2
    NOx_removed_kg_yr = NOx_removed_kg_s * annual_s

    annual_capex = annualize(capex_total, inp.plant_life_yr, inp.discount_rate)
    total_annual_cost = annual_capex + opex_annual
    LCOC = total_annual_cost / max(NOx_removed_kg_yr, 1e-6)

    return {
        "tech": tech, "catalyst_base": cat_base, "scr_T_C": scr_T,
        "air_kg_s": air_kg_s, "scr_kg_s": scr_kg_s,
        "eff_capability": eff_capability * 100.0, "eff_actual": eff_actual * 100.0,
        "NH3_kg_s": nh3_kg_s, "NH3_kg_yr": nh3_annual_kg, "NH3_slip_ppm": slip,
        "cat_vol_m3": cat_vol,
        "scr_capex": scr_capex, "air_capex": air_capex, "wh_capex": wh_capex,
        "wh_recoverable_MW": wh_recoverable_MW,
        "capex_total": capex_total, "annual_capex": annual_capex,
        "nh3_cost": nh3_cost, "cat_replacement_annual": cat_replacement_annual,
        "elec_cost": elec_cost, "dp_cost": dp_cost, "steam_credit": steam_credit,
        "opex_annual": opex_annual,
        "NOx_removed_kg_yr": NOx_removed_kg_yr,
        "total_annual_cost": total_annual_cost, "LCOC": LCOC,
    }


# ============================================================================
# Session state 초기화 + 프리셋 적용
# ============================================================================
def apply_preset(preset_name: str):
    """선택된 GT 프리셋 + 모든 비-GT 기본값을 session_state에 일괄 적용"""
    p = GT_TYPICAL[preset_name]
    st.session_state.exh_T_C = float(p["exh_T_C"])
    st.session_state.exh_kg_s = float(p["exh_kg_s"])
    st.session_state.NOx_in_ppm = float(p["NOx_DLN_ppm"])
    for k, v in DEFAULT_NON_GT.items():
        st.session_state[k] = v


# 최초 1회 초기화
if "_initialized" not in st.session_state:
    st.session_state.gt_preset = "범용 평균값"
    apply_preset("범용 평균값")
    st.session_state._initialized = True


def on_preset_change():
    apply_preset(st.session_state.gt_preset)


# ============================================================================
# 사이드바 입력
# ============================================================================
with st.sidebar:
    st.markdown("## ⚙️ 입력 조건")

    if st.button("🔄 현재 프리셋으로 모든 값 리셋", use_container_width=True,
                  type="primary"):
        apply_preset(st.session_state.gt_preset)
        st.rerun()

    st.markdown("---")

    with st.expander("🔥 GT 배기 조건", expanded=True):
        st.selectbox(
            "GT 모델 프리셋",
            list(GT_TYPICAL.keys()),
            key="gt_preset",
            on_change=on_preset_change,
        )
        st.number_input(
            "GT 배기온도 (°C)",
            min_value=300.0, max_value=700.0, step=5.0,
            key="exh_T_C", format="%.0f",
            help="단순사이클 GT 배기 온도. 통상 400~600°C.",
        )
        st.number_input(
            "배기 유량 (kg/s)",
            min_value=20.0, max_value=1000.0, step=5.0,
            key="exh_kg_s", format="%.0f",
            help="GT 배기 질량유량. 100MW급 SCGT ≈ 250 kg/s.",
        )
        st.number_input(
            "입구 NOx (ppmvd @ 15% O₂)",
            min_value=1.0, max_value=200.0, step=1.0,
            key="NOx_in_ppm", format="%.1f",
            help="DLN/DLE 연소기 출구 NOx. 통상 9~25 ppm.",
        )
        st.number_input(
            "목표 출구 NOx (ppmvd @ 15% O₂)",
            min_value=0.5, max_value=20.0, step=0.5,
            key="NOx_out_ppm", format="%.1f",
            help="BACT/LAER 일반적 목표 2.0~3.0 ppm.",
        )

    with st.expander("⏱️ 운전·재무", expanded=True):
        st.number_input("연간 가동시간 (h)", min_value=500, max_value=8760,
                        step=100, key="annual_hours", format="%d")
        st.number_input("설비 수명 (년)", min_value=5, max_value=40,
                        step=1, key="plant_life_yr", format="%d")
        st.number_input("할인율 (%)", min_value=0.0, max_value=20.0,
                        step=0.5, key="discount_rate_pct", format="%.1f")

    with st.expander("💰 단가"):
        st.number_input("NH₃ 단가 (USD/kg)", min_value=0.05, max_value=5.0,
                        step=0.05, key="nh3_USD_per_kg", format="%.2f")
        st.number_input("전력 단가 (USD/kWh)", min_value=0.01, max_value=0.50,
                        step=0.01, key="elec_USD_per_kWh", format="%.3f")
        st.number_input("스팀 단가 (USD/ton)", min_value=1.0, max_value=120.0,
                        step=1.0, key="steam_USD_per_t", format="%.1f")

    with st.expander("🧪 SCR 촉매·반응기"):
        st.selectbox(
            "고온 SCR 촉매 선택",
            ["Cu-SSZ-13", "Cu-SAPO-34", "Fe-zeolite"],
            key="ht_catalyst_choice",
            help="제올라이트 베이스별 온도 윈도우/효율이 달라짐",
        )
        st.number_input("저온 V/Ti 촉매 수명 (년)", min_value=1.0, max_value=10.0,
                        step=0.5, key="cat_life_LT", format="%.1f")
        st.number_input("고온 제올라이트 수명 (년)", min_value=1.0, max_value=8.0,
                        step=0.5, key="cat_life_HT", format="%.1f")
        st.number_input("V/Ti 단가 (USD/m³)", min_value=2000.0, max_value=30000.0,
                        step=500.0, key="cat_USD_LT", format="%.0f")
        st.number_input("제올라이트 단가 (USD/m³)", min_value=8000.0, max_value=60000.0,
                        step=500.0, key="cat_USD_HT", format="%.0f")
        st.number_input("저온 SCR GHSV (1/h)", min_value=4000.0, max_value=25000.0,
                        step=500.0, key="GHSV_LT", format="%.0f")
        st.number_input("고온 SCR GHSV (1/h)", min_value=3000.0, max_value=20000.0,
                        step=500.0, key="GHSV_HT", format="%.0f")
        st.number_input("NH₃/NOx 몰비 (α)", min_value=0.90, max_value=1.30,
                        step=0.01, key="alpha", format="%.2f")

    with st.expander("🏗️ CAPEX 단가"):
        st.number_input("SCR 반응기 (USD/m³ 촉매)", min_value=10000.0, max_value=150000.0,
                        step=2500.0, key="scr_USD_per_m3", format="%.0f",
                        help="EPA Cost Manual 기준 GT SCR 총 CAPEX는 출력당 $28~110/kW.")
        st.number_input("Air Tempering (USD per kg/s 외기)",
                        min_value=1000.0, max_value=15000.0, step=250.0,
                        key="air_temp_USD", format="%.0f")
        st.number_input("폐열 회수 보일러 (USD/MWth)",
                        min_value=80000.0, max_value=900000.0, step=10000.0,
                        key="wh_USD", format="%.0f")
        st.checkbox(
            "폐열보일러 CAPEX를 고온 SCR 비용에 포함",
            key="include_wh_capex",
            help="기본은 별개 자산으로 보고 OPEX 크레딧만 인정.",
        )

# 촉매 선택 → catalyst_base 매핑
HT_CHOICE_TO_BASE = {
    "Cu-SSZ-13": "Cu-zeolite",
    "Cu-SAPO-34": "Cu-SAPO-34",
    "Fe-zeolite": "Fe-zeolite",
}

inp = Inputs(
    exh_T_C=st.session_state.exh_T_C,
    exh_kg_s=st.session_state.exh_kg_s,
    NOx_in_ppm=st.session_state.NOx_in_ppm,
    NOx_out_ppm=st.session_state.NOx_out_ppm,
    annual_hours=st.session_state.annual_hours,
    plant_life_yr=st.session_state.plant_life_yr,
    discount_rate=st.session_state.discount_rate_pct / 100.0,
    nh3_USD_per_kg=st.session_state.nh3_USD_per_kg,
    elec_USD_per_kWh=st.session_state.elec_USD_per_kWh,
    steam_USD_per_t=st.session_state.steam_USD_per_t,
    catalyst_life_LT_yr=st.session_state.cat_life_LT,
    catalyst_life_HT_yr=st.session_state.cat_life_HT,
    catalyst_USD_per_m3_LT=st.session_state.cat_USD_LT,
    catalyst_USD_per_m3_HT=st.session_state.cat_USD_HT,
    GHSV_LT=st.session_state.GHSV_LT,
    GHSV_HT=st.session_state.GHSV_HT,
    NH3_NOx_alpha=st.session_state.alpha,
    scr_USD_per_m3=st.session_state.scr_USD_per_m3,
    air_temp_USD_per_kg_s=st.session_state.air_temp_USD,
    waste_heat_USD_per_MWth=st.session_state.wh_USD,
    include_waste_heat_capex=st.session_state.include_wh_capex,
    ht_catalyst_base=HT_CHOICE_TO_BASE[st.session_state.ht_catalyst_choice],
)

res_LT = compute_case(inp, "LT")
res_HT = compute_case(inp, "HT")


# ============================================================================
# UI 헬퍼
# ============================================================================
def kpi_card(label: str, value: str, delta: str = "", delta_kind: str = "neutral"):
    cls = {"good": "delta-good", "bad": "delta-bad", "neutral": "delta-neutral"}[delta_kind]
    delta_html = f'<div class="{cls}">{delta}</div>' if delta else ""
    st.markdown(
        f'<div class="metric-card"><h4>{label}</h4>'
        f'<div class="value">{value}</div>{delta_html}</div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# 메인
# ============================================================================
st.title("SCGT 고온 SCR 도입 벤치마크 툴")
st.markdown(
    "**Air Tempering(냉각 공기 주입) 공정을 제거**하고 **고온 SCR로 직접 운전**하는 "
    "기술 전환을 정량 평가하는 SCGT(단순사이클 가스터빈) NOx 제어 벤치마킹 툴."
)

st.markdown(
    """
    <div class="intro-box">
      <div class="obj-title">🎯 프로젝트 목표</div>
      SCGT 배기 온도(통상 480~580°C)를 외기 혼합으로 350°C까지 강제 냉각하던
      <b>Air Tempering 공정을 제거</b>하고, 제올라이트 베이스의
      <b>고온 SCR 촉매(Cu-SSZ-13 / Cu-SAPO-34 / Fe-zeolite)</b>로
      배기 온도 그대로 직접 NOx 제어를 수행하는 시나리오를 벤치마킹.
      <div class="obj-row">
        <div class="obj-col">
          <h5>기존 (Baseline)</h5>
          <b>저온 SCR + Air Tempering</b>
          <ul>
            <li>외기 혼합 → 배기 350°C 냉각</li>
            <li>V₂O₅-WO₃/TiO₂ 촉매 (280~420°C)</li>
            <li>폐열 손실 + 블로워 전력 소모</li>
            <li>외기 혼합 시스템 추가 CAPEX</li>
          </ul>
        </div>
        <div class="obj-col target">
          <h5>대안 (Target)</h5>
          <b>고온 SCR 직접 운전</b>
          <ul>
            <li>외기 혼합 없이 배기 그대로 SCR 통과</li>
            <li>Cu/Fe-zeolite, Cu-SAPO-34 (380~600°C)</li>
            <li>폐열 ~80~100 MWth 회수 가능</li>
            <li>Air Tempering 시스템 자체 제거</li>
          </ul>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "데이터: EPA RBLC · EPA AP-42 §3.1 · EPA Cost Manual Ch.2 · "
    "EPRI 3002022688/3002030747/3002030748 · 제조사 카탈로그 (Cormetech, Topsoe, JM, "
    "BASF, Umicore, Hitachi-Zosen, MHI) · GE/Siemens/MHI GT 사양"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊  종합비교", "⚡  에너지분해", "💵  경제성", "📈  트렌드", "🧪  촉매·제조사"]
)

# ----------------------------------------------------------------------------
# Tab 1: 종합비교
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("고온 SCR 도입 효과 — 기존 (LT+AT) → 대안 (HT-SCR)")
    st.caption("Air Tempering 제거 + 고온 SCR 직접 운전 시 얻는 정량적 이득.")

    delta_LCOC = res_HT["LCOC"] - res_LT["LCOC"]
    delta_capex = res_HT["capex_total"] - res_LT["capex_total"]
    delta_opex = res_HT["opex_annual"] - res_LT["opex_annual"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card(
            "LCOC 절감 효과",
            f"{(-delta_LCOC):+.2f} USD/kg",
            f"기존 ${res_LT['LCOC']:.2f} → 대안 ${res_HT['LCOC']:.2f}",
            "good" if delta_LCOC < 0 else "bad",
        )
    with c2:
        kpi_card(
            "회수 가능 폐열 (HT-SCR)",
            f"{res_HT['wh_recoverable_MW']:.1f} MWth",
            f"≈ {res_HT['steam_credit']/1e6:.2f}M USD/yr 스팀 크레딧",
            "good",
        )
    with c3:
        kpi_card(
            "Air Tempering 제거 이득",
            f"외기 {res_LT['air_kg_s']:.0f} kg/s ⊘",
            f"AT CAPEX ${res_LT['air_capex']/1e6:.2f}M + "
            f"블로워 {res_LT['elec_cost']/1e3:.0f}K USD/yr 제거",
            "good",
        )
    with c4:
        cat_delta = res_HT["cat_vol_m3"] - res_LT["cat_vol_m3"]
        kpi_card(
            "촉매 부피 변화 (HT − LT)",
            f"{cat_delta:+.0f} m³",
            f"LT {res_LT['cat_vol_m3']:.0f} → HT {res_HT['cat_vol_m3']:.0f} m³ "
            f"(외기 희석 제거 효과)",
            "good" if cat_delta < 0 else "neutral",
        )

    st.markdown("---")

    df = pd.DataFrame({
        "지표": [
            "SCR 운전온도 (°C)", "SCR 통과 유량 (kg/s)", "외기 혼합량 (kg/s)",
            "촉매 베이스", "효율 능력 (%)", "효율 실제 (%)",
            "NH₃ 사용량 (kg/h)", "NH₃ slip (ppm)", "촉매 부피 (m³)",
            "SCR CAPEX (M USD)", "Air Tempering CAPEX (M USD)",
            "폐열회수 CAPEX (M USD)", "총 CAPEX (M USD)",
            "연간 OPEX (M USD/yr)", "LCOC (USD/kg-NOx)",
        ],
        "기존 (LT + Air Tempering)": [
            f"{res_LT['scr_T_C']:.0f}", f"{res_LT['scr_kg_s']:.1f}",
            f"{res_LT['air_kg_s']:.1f}", res_LT["catalyst_base"],
            f"{res_LT['eff_capability']:.1f}", f"{res_LT['eff_actual']:.1f}",
            f"{res_LT['NH3_kg_s']*3600:.2f}", f"{res_LT['NH3_slip_ppm']:.1f}",
            f"{res_LT['cat_vol_m3']:.1f}",
            f"{res_LT['scr_capex']/1e6:.2f}", f"{res_LT['air_capex']/1e6:.2f}",
            "0.00", f"{res_LT['capex_total']/1e6:.2f}",
            f"{res_LT['opex_annual']/1e6:.2f}", f"${res_LT['LCOC']:.2f}",
        ],
        f"대안 (HT-SCR / {res_HT['catalyst_base']})": [
            f"{res_HT['scr_T_C']:.0f}", f"{res_HT['scr_kg_s']:.1f}",
            "0.0", res_HT["catalyst_base"],
            f"{res_HT['eff_capability']:.1f}", f"{res_HT['eff_actual']:.1f}",
            f"{res_HT['NH3_kg_s']*3600:.2f}", f"{res_HT['NH3_slip_ppm']:.1f}",
            f"{res_HT['cat_vol_m3']:.1f}",
            f"{res_HT['scr_capex']/1e6:.2f}", "0.00",
            f"{res_HT['wh_capex']/1e6:.2f}", f"{res_HT['capex_total']/1e6:.2f}",
            f"{res_HT['opex_annual']/1e6:.2f}", f"${res_HT['LCOC']:.2f}",
        ],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    metrics = ["촉매 부피 (m³)", "총 CAPEX (M USD)", "연간 OPEX (M USD)", "LCOC (USD/kg)"]
    LT_vals = [res_LT["cat_vol_m3"], res_LT["capex_total"]/1e6,
               res_LT["opex_annual"]/1e6, res_LT["LCOC"]]
    HT_vals = [res_HT["cat_vol_m3"], res_HT["capex_total"]/1e6,
               res_HT["opex_annual"]/1e6, res_HT["LCOC"]]

    fig = make_subplots(rows=1, cols=4, subplot_titles=metrics,
                         horizontal_spacing=0.10, vertical_spacing=0.15)
    for i, (lt, ht) in enumerate(zip(LT_vals, HT_vals)):
        fig.add_trace(go.Bar(
            x=["기존<br>LT+AT"], y=[lt],
            marker_color=COLOR["LT_case"],
            marker_line_color=COLOR["axis"], marker_line_width=1,
            showlegend=(i == 0), name="기존 (LT + Air Tempering)",
            text=[f"{lt:.2f}"], textposition="outside",
            textfont=dict(family="Times New Roman, serif", size=11),
            cliponaxis=False,
        ), row=1, col=i+1)
        fig.add_trace(go.Bar(
            x=["대안<br>HT-SCR"], y=[ht],
            marker_color=COLOR["HT_case"],
            marker_line_color=COLOR["axis"], marker_line_width=1,
            showlegend=(i == 0), name="대안 (HT-SCR 직접)",
            text=[f"{ht:.2f}"], textposition="outside",
            textfont=dict(family="Times New Roman, serif", size=11),
            cliponaxis=False,
        ), row=1, col=i+1)
        # 텍스트 라벨이 잘리지 않도록 y축 상한 30% 여유
        ymax = max(lt, ht) * 1.30 if max(lt, ht) > 0 else 1.0
        fig.update_yaxes(range=[0, ymax], row=1, col=i+1)
    paper_style(fig, height=400)
    fig.update_layout(
        margin=dict(l=50, r=30, t=110, b=60),
        legend=dict(orientation="h", y=1.20, x=0.5, xanchor="center",
                    yanchor="bottom"),
    )
    # 서브플롯 제목 폰트 통일 + 위치 약간 위로
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(family="Times New Roman, serif", size=13, color=COLOR["text"])
        ann["yshift"] = 8
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 2: 에너지 분해
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Air Tempering 제거가 만드는 폐열 회수 기회")
    st.caption("기존 방식: GT 배기를 350°C까지 외기로 강제 냉각 → 그만큼의 엔탈피가 손실. "
               "대안 방식: 배기 온도 그대로 SCR 통과 → 후단 폐열보일러로 직접 회수.")

    cp = 1.10
    Q_in = inp.exh_kg_s * cp * (inp.exh_T_C - 25.0) / 1000.0
    Q_LT_to_350 = inp.exh_kg_s * cp * (inp.exh_T_C - 350.0) / 1000.0
    Q_LT_recoverable = inp.exh_kg_s * cp * (350.0 - 200.0) / 1000.0
    Q_HT_recoverable = res_HT["wh_recoverable_MW"]

    col1, col2 = st.columns(2)

    # 가독성 확보를 위해 paper-friendly 색 + 명시 텍스트 사이즈
    sankey_layout = dict(
        font=dict(family="Times New Roman, Times, serif", size=14, color="#000000"),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        height=420, margin=dict(l=20, r=20, t=40, b=20),
    )

    with col1:
        st.markdown("**기존 — 저온 SCR + Air Tempering (Baseline)**")
        fig = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(
                label=[
                    f"GT 배기 ({Q_in:.0f} MWth)",
                    "Air Tempering 손실",
                    "SCR (저온 운전)",
                    f"폐열회수 가능 ({Q_LT_recoverable:.0f} MWth)",
                    "굴뚝 배출",
                ],
                color=["#9ec5e8", "#bdbdbd", "#cccccc", "#a8d5a2", "#f5b7b1"],
                line=dict(color="#000000", width=0.8),
                pad=20, thickness=22,
            ),
            link=dict(
                source=[0, 0, 2, 2],
                target=[1, 2, 3, 4],
                value=[Q_LT_to_350, Q_in - Q_LT_to_350,
                       Q_LT_recoverable,
                       max(0.1, Q_in - Q_LT_to_350 - Q_LT_recoverable)],
                color=["rgba(150,150,150,0.45)", "rgba(31,119,180,0.45)",
                       "rgba(44,160,44,0.45)", "rgba(214,39,40,0.30)"],
            ),
            textfont=dict(family="Times New Roman, serif", size=13, color="#000000"),
        ))
        fig.update_layout(**sankey_layout)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**대안 — 고온 SCR 직접 운전 (Target)**")
        fig = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(
                label=[
                    f"GT 배기 ({Q_in:.0f} MWth)",
                    "고온 SCR (온도 유지)",
                    f"폐열보일러 회수 ({Q_HT_recoverable:.0f} MWth)",
                    "굴뚝 배출",
                ],
                color=["#fbcb95", "#cccccc", "#a8d5a2", "#f5b7b1"],
                line=dict(color="#000000", width=0.8),
                pad=20, thickness=22,
            ),
            link=dict(
                source=[0, 1, 1],
                target=[1, 2, 3],
                value=[Q_in, Q_HT_recoverable,
                       max(0.1, Q_in - Q_HT_recoverable)],
                color=["rgba(255,127,14,0.45)", "rgba(44,160,44,0.55)",
                       "rgba(214,39,40,0.30)"],
            ),
            textfont=dict(family="Times New Roman, serif", size=13, color="#000000"),
        ))
        fig.update_layout(**sankey_layout)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("촉매 베이스별 온도-효율 곡선")

    Ts = np.linspace(250, 650, 250)
    fig = go.Figure()
    cat_styles = [
        ("V/Ti",       COLOR["VTi"],     "solid",  "V₂O₅-WO₃/TiO₂"),
        ("Cu-zeolite", COLOR["Cu_zeo"],  "solid",  "Cu-SSZ-13 (Cu-zeolite)"),
        ("Fe-zeolite", COLOR["Fe_zeo"],  "dash",   "Fe-zeolite"),
        ("Cu-SAPO-34", COLOR["Cu_SAPO"], "dot",    "Cu-SAPO-34"),
    ]
    for base, col, dash, lbl in cat_styles:
        eff = [scr_efficiency_curve(t, base) * 100 for t in Ts]
        fig.add_trace(go.Scatter(
            x=Ts, y=eff, name=lbl,
            line=dict(color=col, width=2.4, dash=dash),
            mode="lines",
        ))

    # 운전점 마커 (annotation 대신 점)
    fig.add_trace(go.Scatter(
        x=[res_LT["scr_T_C"]], y=[res_LT["eff_capability"]],
        mode="markers", marker=dict(symbol="diamond", size=14,
                                     color=COLOR["VTi"],
                                     line=dict(color="#000000", width=1.2)),
        name=f"LT 운전점 ({res_LT['scr_T_C']:.0f}°C)",
    ))
    fig.add_trace(go.Scatter(
        x=[res_HT["scr_T_C"]], y=[res_HT["eff_capability"]],
        mode="markers", marker=dict(symbol="square", size=14,
                                     color="#ffffff",
                                     line=dict(color=COLOR["HT_case"], width=2.5)),
        name=f"HT 운전점 ({res_HT['scr_T_C']:.0f}°C)",
    ))

    # 운전 윈도우 음영 (matplotlib hatch 대용)
    fig.add_vrect(x0=280, x1=420, fillcolor=COLOR["VTi"],
                   opacity=0.06, line_width=0, layer="below")
    fig.add_vrect(x0=380, x1=600, fillcolor=COLOR["Cu_zeo"],
                   opacity=0.05, line_width=0, layer="below")

    paper_style(fig, height=480)
    fig.update_layout(
        xaxis_title="SCR 운전 온도 (°C)",
        yaxis_title="NOx 제거 효율 (%)",
        legend=dict(orientation="h", x=0.5, y=-0.22, xanchor="center",
                    yanchor="top", bgcolor="rgba(255,255,255,0.95)"),
        margin=dict(l=72, r=30, t=30, b=130),
    )
    fig.update_xaxes(range=[250, 650], dtick=50)
    fig.update_yaxes(range=[20, 100], dtick=10)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 3: 경제성
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("CAPEX / OPEX 분해")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**CAPEX 구성**")
        items = ["SCR 반응기", "Air Tempering", "폐열 회수 보일러"]
        LT_capex = [res_LT["scr_capex"]/1e6, res_LT["air_capex"]/1e6, 0]
        HT_capex = [res_HT["scr_capex"]/1e6, 0, res_HT["wh_capex"]/1e6]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="기존 (LT+AT)", x=items, y=LT_capex,
                              marker_color=COLOR["LT_case"],
                              marker_line_color=COLOR["axis"], marker_line_width=1,
                              text=[f"{v:.2f}" for v in LT_capex],
                              textposition="outside"))
        fig.add_trace(go.Bar(name="대안 (HT-SCR)", x=items, y=HT_capex,
                              marker_color=COLOR["HT_case"],
                              marker_line_color=COLOR["axis"], marker_line_width=1,
                              text=[f"{v:.2f}" for v in HT_capex],
                              textposition="outside"))
        fig.update_layout(barmode="group", yaxis_title="CAPEX (M USD)",
                           legend=dict(orientation="h", y=1.15,
                                        x=0.5, xanchor="center"))
        paper_style(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**연간 OPEX 구성**")
        opex_items = ["NH₃", "촉매교체", "전력(블로워)", "ΔP 보상", "스팀크레딧"]
        LT_opex = [res_LT["nh3_cost"], res_LT["cat_replacement_annual"],
                    res_LT["elec_cost"], res_LT["dp_cost"], 0]
        HT_opex = [res_HT["nh3_cost"], res_HT["cat_replacement_annual"],
                    res_HT["elec_cost"], res_HT["dp_cost"], -res_HT["steam_credit"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="저온+AT", x=opex_items,
                              y=[v/1e6 for v in LT_opex],
                              marker_color=COLOR["LT_case"],
                              marker_line_color=COLOR["axis"], marker_line_width=1))
        fig.add_trace(go.Bar(name="고온", x=opex_items,
                              y=[v/1e6 for v in HT_opex],
                              marker_color=COLOR["HT_case"],
                              marker_line_color=COLOR["axis"], marker_line_width=1))
        fig.update_layout(barmode="group", yaxis_title="연간 비용 (M USD/yr)",
                           legend=dict(orientation="h", y=1.15,
                                        x=0.5, xanchor="center"))
        fig.add_hline(y=0, line_color=COLOR["axis"], line_width=1)
        paper_style(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("연도별 누적 비용 (NPV)")
    years = np.arange(0, inp.plant_life_yr + 1)
    LT_cum, HT_cum = np.zeros(len(years)), np.zeros(len(years))
    LT_cum[0] = res_LT["capex_total"]
    HT_cum[0] = res_HT["capex_total"]
    for i in range(1, len(years)):
        LT_cum[i] = LT_cum[i-1] + res_LT["opex_annual"] / (1 + inp.discount_rate) ** i
        HT_cum[i] = HT_cum[i-1] + res_HT["opex_annual"] / (1 + inp.discount_rate) ** i

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=LT_cum/1e6, mode="lines+markers",
                              name="기존 (LT + Air Tempering)",
                              line=dict(color=COLOR["LT_case"], width=2.4),
                              marker=dict(size=7, line=dict(color=COLOR["axis"], width=1))))
    fig.add_trace(go.Scatter(x=years, y=HT_cum/1e6, mode="lines+markers",
                              name="대안 (HT-SCR 직접)",
                              line=dict(color=COLOR["HT_case"], width=2.4, dash="dash"),
                              marker=dict(size=7, symbol="square",
                                           line=dict(color=COLOR["axis"], width=1))))
    fig.update_layout(xaxis_title="연도", yaxis_title="누적 비용 (M USD)",
                       legend=dict(orientation="h", y=1.12,
                                    x=0.5, xanchor="center"))
    paper_style(fig, height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("LCOC 민감도")
    sens_param = st.selectbox("민감도 변수",
        ["연간 가동시간", "NH₃ 단가", "스팀 단가", "고온 촉매 수명"])

    if sens_param == "연간 가동시간":
        xs = np.linspace(1000, 8000, 30)
        kw, x_label = "annual_hours", "연간 가동시간 (h)"
    elif sens_param == "NH₃ 단가":
        xs = np.linspace(0.2, 1.5, 30)
        kw, x_label = "nh3_USD_per_kg", "NH₃ 단가 (USD/kg)"
    elif sens_param == "스팀 단가":
        xs = np.linspace(5, 70, 30)
        kw, x_label = "steam_USD_per_t", "스팀 단가 (USD/ton)"
    else:
        xs = np.linspace(1.5, 6.0, 25)
        kw, x_label = "catalyst_life_HT_yr", "고온 촉매 수명 (년)"

    LT_y, HT_y = [], []
    for x in xs:
        tmp = Inputs(**{**inp.__dict__, kw: x})
        LT_y.append(compute_case(tmp, "LT")["LCOC"])
        HT_y.append(compute_case(tmp, "HT")["LCOC"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=LT_y, mode="lines",
                              name="기존 (LT + Air Tempering)",
                              line=dict(color=COLOR["LT_case"], width=2.4)))
    fig.add_trace(go.Scatter(x=xs, y=HT_y, mode="lines",
                              name="대안 (HT-SCR 직접)",
                              line=dict(color=COLOR["HT_case"], width=2.4, dash="dash")))
    fig.update_layout(xaxis_title=x_label, yaxis_title="LCOC (USD/kg-NOx)",
                       legend=dict(orientation="h", y=1.12,
                                    x=0.5, xanchor="center"))
    paper_style(fig, height=380)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 4: 트렌드 (LIT)
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("실측 현장 사례 (EPA RBLC + EPRI + OEM)")
    st.caption("저온 SCR(BACT 시설) vs 고온 SCR(EPRI 파일럿/시연) 실측 데이터. "
               "촉매 베이스별 마커 구분, 마커 크기 = NH₃ slip (ppm). "
               "★ = 현재 입력 조건의 운전점 (기존/대안).")

    df_lit = pd.DataFrame(LIT_RBLC)

    # 산점도
    fig = go.Figure()
    base_styles = {
        "V/Ti":       (COLOR["VTi"],     "circle"),
        "Cu-zeolite": (COLOR["Cu_zeo"],  "square"),
        "Fe-zeolite": (COLOR["Fe_zeo"],  "diamond"),
        "Cu-SAPO-34": (COLOR["Cu_SAPO"], "triangle-up"),
    }
    for base, (col, sym) in base_styles.items():
        sub = df_lit[df_lit["catalyst_base"] == base]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=sub["scr_T_C"], y=sub["NOx_red_pct"],
            mode="markers",
            marker=dict(
                size=8 + sub["NH3_slip_ppm"] * 1.5,
                color=col, symbol=sym,
                line=dict(color=COLOR["axis"], width=1),
            ),
            name=base,
            customdata=sub[["facility", "turbine", "catalyst_detail",
                             "outlet_NOx_ppmvd_15O2", "year", "source"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "GT: %{customdata[1]}<br>"
                "촉매: %{customdata[2]}<br>"
                "출구 NOx: %{customdata[3]} ppm<br>"
                "운전온도: %{x}°C<br>"
                "효율: %{y}%<br>"
                "Year: %{customdata[4]}<br>"
                "출처: %{customdata[5]}<extra></extra>"
            ),
        ))

    # 현재 입력 운전점
    fig.add_trace(go.Scatter(
        x=[res_LT["scr_T_C"]], y=[res_LT["eff_actual"]],
        mode="markers+text", name="기존 운전점 (LT + AT, V/Ti)",
        marker=dict(symbol="star", size=22, color="#ffffff",
                    line=dict(color=COLOR["VTi"], width=2.5)),
        text=["기존"], textposition="top center",
        textfont=dict(family="Times New Roman, serif", size=12),
    ))
    fig.add_trace(go.Scatter(
        x=[res_HT["scr_T_C"]], y=[res_HT["eff_actual"]],
        mode="markers+text", name=f"대안 운전점 (HT-SCR, {res_HT['catalyst_base']})",
        marker=dict(symbol="star", size=22, color="#ffffff",
                    line=dict(color=COLOR["HT_case"], width=2.5)),
        text=["대안"], textposition="top center",
        textfont=dict(family="Times New Roman, serif", size=12),
    ))

    paper_style(fig, height=520)
    fig.update_layout(
        xaxis_title="SCR 운전온도 (°C)",
        yaxis_title="NOx 제거 효율 (%)",
        legend=dict(orientation="h", x=0.5, y=-0.22, xanchor="center",
                     yanchor="top", bgcolor="rgba(255,255,255,0.95)"),
        margin=dict(l=72, r=30, t=30, b=140),
    )
    fig.update_xaxes(range=[330, 620], dtick=20)
    fig.update_yaxes(range=[78, 96], dtick=2)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("문헌 데이터 테이블")

    cat_filter = st.multiselect(
        "촉매 베이스 필터",
        options=["V/Ti", "Cu-zeolite", "Fe-zeolite", "Cu-SAPO-34"],
        default=["V/Ti", "Cu-zeolite", "Fe-zeolite", "Cu-SAPO-34"],
    )
    df_show = df_lit[df_lit["catalyst_base"].isin(cat_filter)]
    st.dataframe(
        df_show[["facility", "turbine", "mode", "tech_category",
                 "catalyst_base", "catalyst_detail", "scr_T_C",
                 "outlet_NOx_ppmvd_15O2", "NOx_red_pct", "NH3_slip_ppm",
                 "year", "source"]].rename(columns={
            "facility": "사업장", "turbine": "GT", "mode": "운전",
            "tech_category": "기술분류", "catalyst_base": "촉매베이스",
            "catalyst_detail": "촉매상세", "scr_T_C": "SCR온도(°C)",
            "outlet_NOx_ppmvd_15O2": "출구NOx(ppm)",
            "NOx_red_pct": "효율(%)", "NH3_slip_ppm": "Slip(ppm)",
            "year": "연도", "source": "출처",
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown("---")
    st.subheader("AP-42 §3.1 미통제 NOx 배출계수")
    fig = go.Figure(go.Bar(
        x=list(AP42_NOX_FACTORS.keys()),
        y=list(AP42_NOX_FACTORS.values()),
        marker_color=["#94a3b8", "#64748b", "#475569", "#1f2937"],
        marker_line_color=COLOR["axis"], marker_line_width=1,
        text=[f"{v:.3f}" for v in AP42_NOX_FACTORS.values()],
        textposition="outside",
        textfont=dict(family="Times New Roman, serif", size=12),
    ))
    fig.update_layout(yaxis_title="NOx 배출계수 (lb/MMBtu)",
                       showlegend=False)
    paper_style(fig, height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "📚 **참고문헌 핵심 결론**\n\n"
        "- EPA RBLC BACT: SCGT 통상 **2.5~3.0 ppmvd@15% O₂** (LAER 2.0 ppm)\n"
        "- EPRI: **Cu-SAPO-34**가 고온 안정성·효율 면에서 가장 우수, "
        "**Cu-SSZ-13**이 운전 윈도우 가장 넓음, **Fe-zeolite**는 고온부 강점이나 저온부 활성 낮음\n"
        "- V₂O₅-WO₃/TiO₂는 **>420°C에서 V 휘발 + TiO₂ 상변태** → "
        "SCGT(>500°C)에는 air tempering 필수\n"
        "- **NH₃/NOx α=1.05**, slip <5 ppm이 BAAQMD/CARB 표준 설계점"
    )

# ----------------------------------------------------------------------------
# Tab 5: 촉매·제조사 (Manufacturer × Catalyst Performance)
# ----------------------------------------------------------------------------
with tab5:
    st.subheader("촉매 베이스별 성능 요약")
    st.caption("4개 촉매 베이스의 운전 윈도우 / 최적 온도 / 최대 효율 비교 — "
               "물리·화학적 특성 기준.")

    # 베이스별 카드 4개
    cb_cols = st.columns(4)
    base_info = [
        ("V/Ti", "V₂O₅-WO₃/TiO₂", COLOR["VTi"]),
        ("Cu-zeolite", "Cu-SSZ-13 / Cu-Beta", COLOR["Cu_zeo"]),
        ("Fe-zeolite", "Fe-Beta / Fe-ZSM-5", COLOR["Fe_zeo"]),
        ("Cu-SAPO-34", "Cu-SAPO-34", COLOR["Cu_SAPO"]),
    ]
    for col, (base, detail, color) in zip(cb_cols, base_info):
        with col:
            cw = CAT_WINDOWS[base]
            T_lo, T_hi = cw["window"]
            st.markdown(
                f"""
                <div class="metric-card" style="border-left: 6px solid {color};">
                  <h4>{base}</h4>
                  <div style="font-size:13px; color:#444; margin-bottom:8px;">{detail}</div>
                  <div style="font-size:13px; line-height:1.7;">
                    <b>운전창</b>: {T_lo}–{T_hi}°C<br>
                    <b>T_opt</b>: {cw['T_opt']}°C<br>
                    <b>최대 효율</b>: {cw['eff_max']*100:.1f}%
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.subheader("제조사별 SCR 촉매 제품군")
    st.caption("주요 SCR 촉매 제조사의 상용 제품 (Cormetech, Topsoe, JM, BASF, Umicore, "
               "Hitachi-Zosen, MHI, BHK 등) 의 화학·운전창·효율·시장 정리.")

    df_mfr = pd.DataFrame(LIT_MFR)

    # 필터 행
    fcol1, fcol2 = st.columns([1, 1])
    with fcol1:
        sel_mfr = st.multiselect(
            "제조사 필터",
            options=sorted(df_mfr["manufacturer"].unique()),
            default=sorted(df_mfr["manufacturer"].unique()),
        )
    with fcol2:
        sel_base = st.multiselect(
            "촉매 베이스 필터",
            options=sorted(df_mfr["catalyst_base"].unique()),
            default=sorted(df_mfr["catalyst_base"].unique()),
        )

    df_view = df_mfr[
        df_mfr["manufacturer"].isin(sel_mfr) &
        df_mfr["catalyst_base"].isin(sel_base)
    ].reset_index(drop=True)

    # 운전온도 범위 — Gantt 스타일 가로 막대 (제조사별 운전창)
    st.markdown("##### 제품별 운전 온도 범위")
    fig = go.Figure()

    # 색상 매핑
    base_color_map = {
        "V/Ti":         COLOR["VTi"],
        "V/Ti (LT)":    "#5b8aaf",
        "Cu-zeolite":   COLOR["Cu_zeo"],
        "Cu-SAPO-34":   COLOR["Cu_SAPO"],
        "Fe-zeolite":   COLOR["Fe_zeo"],
        "Mn-CeOx (LT)": "#0d9488",
        "V/Mn (LT)":    "#7c3aed",
    }

    # y축 라벨 = "제조사 — 제품"
    df_view = df_view.copy()
    df_view["label"] = df_view["manufacturer"] + " — " + df_view["product"]
    df_view = df_view.sort_values("T_min").reset_index(drop=True)

    for _, r in df_view.iterrows():
        col = base_color_map.get(r["catalyst_base"], "#888888")
        # 운전 윈도우 (선)
        fig.add_trace(go.Scatter(
            x=[r["T_min"], r["T_max"]],
            y=[r["label"], r["label"]],
            mode="lines",
            line=dict(color=col, width=10),
            opacity=0.55,
            showlegend=False,
            hoverinfo="skip",
        ))
        # 최적점 (마커)
        fig.add_trace(go.Scatter(
            x=[r["T_opt"]], y=[r["label"]],
            mode="markers",
            marker=dict(symbol="diamond", size=12, color=col,
                         line=dict(color="#000000", width=1.2)),
            showlegend=False,
            hovertemplate=(
                f"<b>{r['manufacturer']} · {r['product']}</b><br>"
                f"화학: {r['chemistry']}<br>"
                f"베이스: {r['catalyst_base']}<br>"
                f"운전창: {r['T_min']}–{r['T_max']}°C (T_opt {r['T_opt']}°C)<br>"
                f"최대 효율: {r['eff_max_pct']}%<br>"
                f"NH₃ slip: {r['slip_ppm']} ppm<br>"
                f"시장: {r['market']}<extra></extra>"
            ),
        ))

    # 색상 범례 (수동 dummy traces)
    for label, color in base_color_map.items():
        if label in df_view["catalyst_base"].values:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=12, color=color, symbol="square",
                             line=dict(color="#000000", width=1)),
                name=label, showlegend=True,
            ))

    paper_style(fig, height=max(500, 24 * len(df_view) + 200))
    fig.update_layout(
        xaxis_title="운전 온도 (°C)",
        yaxis_title="",
        legend=dict(orientation="h", x=0.5, y=-0.18, xanchor="center",
                     yanchor="top", bgcolor="rgba(255,255,255,0.95)"),
        margin=dict(l=240, r=30, t=30, b=120),
    )
    fig.update_xaxes(range=[140, 620], dtick=50)
    fig.update_yaxes(automargin=True, tickfont=dict(size=11))
    st.plotly_chart(fig, use_container_width=True)

    # 효율 비교 막대
    st.markdown("##### 제조사별 최대 NOx 제거 효율")
    fig = go.Figure()
    df_sorted = df_view.sort_values("eff_max_pct", ascending=True).reset_index(drop=True)
    bar_colors = [base_color_map.get(b, "#888") for b in df_sorted["catalyst_base"]]
    fig.add_trace(go.Bar(
        x=df_sorted["eff_max_pct"],
        y=df_sorted["label"],
        orientation="h",
        marker=dict(color=bar_colors,
                     line=dict(color=COLOR["axis"], width=1)),
        text=[f"{v:.1f}%" for v in df_sorted["eff_max_pct"]],
        textposition="outside",
        textfont=dict(family="Times New Roman, serif", size=11),
        showlegend=False,
        cliponaxis=False,
    ))
    paper_style(fig, height=max(400, 22 * len(df_sorted) + 150))
    fig.update_layout(
        xaxis_title="최대 NOx 제거 효율 (%)",
        margin=dict(l=240, r=60, t=30, b=60),
    )
    fig.update_xaxes(range=[80, 100], dtick=2)
    fig.update_yaxes(automargin=True, tickfont=dict(size=11))
    st.plotly_chart(fig, use_container_width=True)

    # 데이터 테이블
    st.markdown("##### 제조사 × 촉매 데이터 테이블")
    st.dataframe(
        df_view[[
            "manufacturer", "country", "product", "chemistry",
            "catalyst_base", "form", "T_min", "T_max", "T_opt",
            "eff_max_pct", "slip_ppm", "market", "ref",
        ]].rename(columns={
            "manufacturer": "제조사", "country": "국가", "product": "제품",
            "chemistry": "화학식", "catalyst_base": "베이스", "form": "형상",
            "T_min": "T_min(°C)", "T_max": "T_max(°C)", "T_opt": "T_opt(°C)",
            "eff_max_pct": "효율(%)", "slip_ppm": "Slip(ppm)",
            "market": "시장", "ref": "출처",
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown("---")
    st.subheader("저온 SCR 효율 R&D 트렌드 (1995–2025)")
    st.caption("학술·산업계 발표 효율 데이터 시계열. T<300°C 영역에서의 "
               "Mn-CeO₂, Cu-CHA, V-Mn-Ce 등 신규 촉매 발전 추적.")

    df_rnd = pd.DataFrame(LT_SCR_RND)

    # 산점도: x=year, y=efficiency, marker size=T_C inverse, color by tech family
    def _tech_family(t: str) -> str:
        if "Mn" in t and "Fe" not in t: return "Mn 계열"
        if "Cu" in t: return "Cu 계열"
        if "V-Mn" in t or "V-W-Mo" in t: return "V/Ti 강화"
        if "Fe" in t: return "Fe-Cu 복합"
        return "V/Ti 클래식"
    df_rnd["family"] = df_rnd["tech"].apply(_tech_family)

    family_colors = {
        "V/Ti 클래식":  COLOR["VTi"],
        "V/Ti 강화":    "#5b8aaf",
        "Cu 계열":      COLOR["Cu_zeo"],
        "Fe-Cu 복합":   COLOR["Fe_zeo"],
        "Mn 계열":      "#0d9488",
    }

    fig = go.Figure()
    for fam, col in family_colors.items():
        sub = df_rnd[df_rnd["family"] == fam]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=sub["year"], y=sub["eff_pct"],
            mode="markers+lines",
            name=fam,
            line=dict(color=col, width=1.5, dash="dot"),
            marker=dict(
                size=[max(8, 30 - (T - 150) * 0.3) for T in sub["T_C"]],
                color=col,
                line=dict(color="#000000", width=1.2),
                symbol="circle",
            ),
            customdata=sub[["tech", "T_C", "note"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Year: %{x}, Efficiency: %{y}%<br>"
                "운전온도: %{customdata[1]}°C<br>"
                "비고: %{customdata[2]}<extra></extra>"
            ),
        ))

    paper_style(fig, height=460)
    fig.update_layout(
        xaxis_title="발표 연도",
        yaxis_title="NOx 제거 효율 (%)",
        legend=dict(orientation="h", x=0.5, y=-0.20, xanchor="center",
                     yanchor="top", bgcolor="rgba(255,255,255,0.95)"),
        margin=dict(l=72, r=30, t=30, b=130),
    )
    fig.update_xaxes(range=[1992, 2027], dtick=2)
    fig.update_yaxes(range=[82, 100], dtick=2)
    st.plotly_chart(fig, use_container_width=True)

    # 운전온도 vs 효율 (저온 SCR 발전 추세)
    st.markdown("##### 운전온도 vs 효율 (저온 SCR 진화)")
    fig = go.Figure()
    for fam, col in family_colors.items():
        sub = df_rnd[df_rnd["family"] == fam]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=sub["T_C"], y=sub["eff_pct"],
            mode="markers+text",
            name=fam,
            text=sub["year"].astype(str),
            textposition="top center",
            textfont=dict(family="Times New Roman, serif", size=10),
            marker=dict(size=14, color=col,
                         line=dict(color="#000000", width=1.2),
                         symbol="circle"),
            hovertemplate=(
                "<b>" + sub["tech"] + "</b><br>"
                "운전온도: %{x}°C<br>"
                "효율: %{y}%<extra></extra>"
            ),
        ))

    # 화살표 주석: '저온화 + 효율↑' 추세
    fig.add_annotation(
        x=200, y=95, ax=320, ay=89.5,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1.4,
        arrowwidth=1.5, arrowcolor="#444",
        text="추세: 저온화 ↓ + 효율 ↑",
        font=dict(family="Times New Roman, serif", size=12, color="#444"),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#444", borderwidth=1,
    )

    paper_style(fig, height=460)
    fig.update_layout(
        xaxis_title="운전 온도 (°C)",
        yaxis_title="NOx 제거 효율 (%)",
        legend=dict(orientation="h", x=0.5, y=-0.20, xanchor="center",
                     yanchor="top", bgcolor="rgba(255,255,255,0.95)"),
        margin=dict(l=72, r=30, t=30, b=130),
    )
    fig.update_xaxes(range=[150, 380], dtick=20)
    fig.update_yaxes(range=[82, 100], dtick=2)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "🔬 **저온 SCR 트렌드 핵심 결론**\n\n"
        "- **저온화 추세**: 2000년대 350°C → 2020년대 200~240°C로 운전 온도 ~100°C 하향\n"
        "- **Mn-CeO₂ 계열**이 150~250°C에서 90% 효율 달성, **tail-end SCR** 시장 부상\n"
        "- **Cu-SSZ-13 / Cu-SAPO-34**는 저온~고온 광역(200~600°C)에서 90% 이상 유지 → "
        "단일 촉매로 GT/Diesel/Industrial 통합 가능\n"
        "- 저온 SCR이 발전하면 **air tempering의 필요성 자체가 감소** → "
        "고온 SCR 옵션과 별개로 SCGT 운영 유연성 증가"
    )


st.markdown("---")
st.caption(
    "SCGT 고온 SCR 도입 벤치마크 v2.2 — Air Tempering 제거 평가 · "
    "Methodology: EPA AP-42 §3.1, EPA Cost Manual Ch.2 (SCR), "
    "EPRI 3002022688/3002030748 · 제조사 카탈로그 (Cormetech, Topsoe, JM, BASF, "
    "Umicore, Hitachi-Zosen, MHI) · "
    "본 결과는 개념설계 단계의 비교 검토용이며, 상세 설계는 OEM 견적과 사이트 데이터에 기반."
)
