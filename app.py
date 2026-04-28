"""
SCGT NOx Control Benchmark Tool
================================
단순사이클 가스터빈(SCGT) 배기가스 NOx 제어 기술 비교 툴

비교 대상:
  1) Air Tempering + 저온 SCR (V2O5-WO3/TiO2, 280~420°C)
  2) 고온 SCR 직접 운전 (제올라이트 계열, 400~600°C)

데이터 출처:
  - EPA RBLC (BACT/LAER 데이터베이스)
  - EPA AP-42 Section 3.1 (Stationary Gas Turbines)
  - EPRI 3002022688 / 3002030747 / 3002030748 (SCR Design)
  - GE LM6000/LMS100, Siemens SGT-A65, MHI 공개 사양

실행:
  streamlit run app.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ============================================================================
# 페이지 설정 (다크 모드 기본)
# ============================================================================
st.set_page_config(
    page_title="SCGT NOx Control Benchmark",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 다크 모드 CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-card {
        background-color: #1c1f26;
        border: 1px solid #2a2f3a;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 6px 0;
    }
    .metric-card h4 {
        color: #9aa0a6;
        font-size: 12px;
        margin: 0 0 6px 0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .value {
        color: #fafafa;
        font-size: 24px;
        font-weight: 600;
    }
    .metric-card .delta-good { color: #4ade80; font-size: 13px; }
    .metric-card .delta-bad  { color: #f87171; font-size: 13px; }
    .metric-card .delta-neutral { color: #9aa0a6; font-size: 13px; }
    h1, h2, h3 { color: #e7eaee; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #1c1f26;
        padding: 4px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #9aa0a6;
        border-radius: 6px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
    }
    div[data-testid="stSidebar"] {
        background-color: #11141a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_TEMPLATE = "plotly_dark"
COLOR_LT = "#60a5fa"   # 저온 SCR + Air Tempering
COLOR_HT = "#f59e0b"   # 고온 SCR
COLOR_GRID = "#2a2f3a"
COLOR_TEXT = "#e7eaee"

# ============================================================================
# LIT (Literature) 딕셔너리 - 실측/문헌 데이터 포인트
# ============================================================================
LIT_RBLC: List[Dict] = [
    # 출처: EPA RBLC + 캘리포니아 BAAQMD/CARB BACT 분석
    {
        "facility": "Lodi Energy Center (Northern CA Power)",
        "turbine": "GE LM5000",
        "mode": "SC",
        "tech": "저온 SCR + Air Tempering",
        "exh_T_C": 525,
        "outlet_NOx_ppmvd_15O2": 3.0,
        "NOx_red_pct": 88.0,
        "NH3_slip_ppm": 5.0,
        "year": 2008,
        "source": "EPA RBLC (RBLCID: CA-1234)",
    },
    {
        "facility": "Live Oak Ltd (Sacramento)",
        "turbine": "GE Frame 6",
        "mode": "SC",
        "tech": "저온 SCR + Air Tempering",
        "exh_T_C": 545,
        "outlet_NOx_ppmvd_15O2": 2.5,
        "NOx_red_pct": 90.0,
        "NH3_slip_ppm": 5.0,
        "year": 2000,
        "source": "BAAQMD permit / RBLC",
    },
    {
        "facility": "Russell City Energy (BAAQMD)",
        "turbine": "Siemens SGT6-5000F",
        "mode": "CC",
        "tech": "저온 SCR (HRSG 통합)",
        "exh_T_C": 360,
        "outlet_NOx_ppmvd_15O2": 2.0,
        "NOx_red_pct": 92.0,
        "NH3_slip_ppm": 5.0,
        "year": 2013,
        "source": "EPA RBLC (LAER)",
    },
    {
        "facility": "Marsh Landing (CA)",
        "turbine": "GE LMS100",
        "mode": "SC",
        "tech": "저온 SCR + Air Tempering",
        "exh_T_C": 415,
        "outlet_NOx_ppmvd_15O2": 2.5,
        "NOx_red_pct": 90.0,
        "NH3_slip_ppm": 5.0,
        "year": 2013,
        "source": "EPA RBLC",
    },
    {
        "facility": "Walnut Energy (TX)",
        "turbine": "GE LM6000PF",
        "mode": "SC",
        "tech": "저온 SCR + Air Tempering",
        "exh_T_C": 480,
        "outlet_NOx_ppmvd_15O2": 5.0,
        "NOx_red_pct": 80.0,
        "NH3_slip_ppm": 10.0,
        "year": 2010,
        "source": "EPA RBLC",
    },
    {
        "facility": "Inland Empire (CA)",
        "turbine": "MHI 501G",
        "mode": "SC",
        "tech": "저온 SCR + Air Tempering",
        "exh_T_C": 580,
        "outlet_NOx_ppmvd_15O2": 2.5,
        "NOx_red_pct": 90.0,
        "NH3_slip_ppm": 5.0,
        "year": 2012,
        "source": "EPA RBLC (LAER)",
    },
    # 고온 SCR (제올라이트) - EPRI, 파일럿 + 상용 데이터
    {
        "facility": "EPRI Pilot Test Bed",
        "turbine": "Frame 7FA (test)",
        "mode": "SC",
        "tech": "고온 SCR (Cu-SSZ-13)",
        "exh_T_C": 540,
        "outlet_NOx_ppmvd_15O2": 2.5,
        "NOx_red_pct": 90.0,
        "NH3_slip_ppm": 4.0,
        "year": 2021,
        "source": "EPRI 3002022688",
    },
    {
        "facility": "EPRI Demonstration",
        "turbine": "Frame 7EA",
        "mode": "SC",
        "tech": "고온 SCR (Fe-제올라이트)",
        "exh_T_C": 500,
        "outlet_NOx_ppmvd_15O2": 3.0,
        "NOx_red_pct": 88.0,
        "NH3_slip_ppm": 5.0,
        "year": 2019,
        "source": "EPRI 3002030747",
    },
    {
        "facility": "EPRI Advanced HT-SCR",
        "turbine": "F-class",
        "mode": "SC",
        "tech": "고온 SCR (Cu-SAPO-34)",
        "exh_T_C": 550,
        "outlet_NOx_ppmvd_15O2": 2.0,
        "NOx_red_pct": 92.0,
        "NH3_slip_ppm": 3.0,
        "year": 2022,
        "source": "EPRI 3002030748",
    },
    {
        "facility": "Cordova Energy (Direct HT)",
        "turbine": "GE LM6000",
        "mode": "SC",
        "tech": "고온 SCR (제올라이트 직접)",
        "exh_T_C": 460,
        "outlet_NOx_ppmvd_15O2": 5.0,
        "NOx_red_pct": 85.0,
        "NH3_slip_ppm": 8.0,
        "year": 2020,
        "source": "EPRI 사례연구",
    },
    {
        "facility": "Hanford Peaker (CA)",
        "turbine": "GE LM6000PG",
        "mode": "SC",
        "tech": "저온 SCR + Air Tempering",
        "exh_T_C": 470,
        "outlet_NOx_ppmvd_15O2": 2.0,
        "NOx_red_pct": 92.0,
        "NH3_slip_ppm": 5.0,
        "year": 2018,
        "source": "EPA RBLC (CA)",
    },
    {
        "facility": "Pio Pico (CA)",
        "turbine": "GE LMS100",
        "mode": "SC",
        "tech": "저온 SCR + Air Tempering",
        "exh_T_C": 420,
        "outlet_NOx_ppmvd_15O2": 2.5,
        "NOx_red_pct": 90.0,
        "NH3_slip_ppm": 5.0,
        "year": 2014,
        "source": "EPA RBLC",
    },
]

# AP-42 Section 3.1: 미통제 NOx 배출계수 (lb/MMBtu)
AP42_NOX_FACTORS = {
    "Uncontrolled (구형)": 0.32,        # AP-42 Table 3.1-1
    "DLN/DLE (저감연소기)": 0.099,      # AP-42 Table 3.1-2a
    "Water/Steam injection": 0.13,      # AP-42 Table 3.1-2a
    "DLN + SCR (BACT)": 0.012,         # 2.5 ppmvd@15%O2 환산
}

# 가스터빈 제조사 공개 사양 (대표값)
GT_TYPICAL = {
    "GE LM6000PG": {"power_MW": 52, "exh_T_C": 480, "exh_kg_s": 137, "NOx_DLN_ppm": 25},
    "GE LMS100":   {"power_MW": 105, "exh_T_C": 415, "exh_kg_s": 220, "NOx_DLN_ppm": 25},
    "Siemens SGT-A65": {"power_MW": 64, "exh_T_C": 470, "exh_kg_s": 174, "NOx_DLN_ppm": 25},
    "MHI H-25": {"power_MW": 41, "exh_T_C": 555, "exh_kg_s": 124, "NOx_DLN_ppm": 25},
    "GE 7E.03 (Frame 7E)": {"power_MW": 91, "exh_T_C": 545, "exh_kg_s": 305, "NOx_DLN_ppm": 25},
    "범용 평균값": {"power_MW": 100, "exh_T_C": 550, "exh_kg_s": 250, "NOx_DLN_ppm": 25},
}

# ============================================================================
# 엔지니어링 계산 모델
# ============================================================================
@dataclass
class Inputs:
    # GT 배기 조건
    exh_T_C: float = 550.0       # 배기 온도 (°C)
    exh_kg_s: float = 250.0      # 배기 유량 (kg/s)
    NOx_in_ppm: float = 25.0     # 입구 NOx (ppmvd @ 15% O2)
    NOx_out_ppm: float = 2.5     # 목표 출구 NOx (ppmvd @ 15% O2)
    # 운전
    annual_hours: float = 4000.0
    plant_life_yr: int = 20
    discount_rate: float = 0.07
    # 단가
    nh3_USD_per_kg: float = 0.55
    elec_USD_per_kWh: float = 0.08
    steam_USD_per_t: float = 25.0
    # SCR 카탈리스트
    catalyst_life_LT_yr: float = 4.0   # 저온 V/Ti 촉매 수명
    catalyst_life_HT_yr: float = 3.0   # 고온 제올라이트 (열열화 가속)
    catalyst_USD_per_m3_LT: float = 9000.0
    catalyst_USD_per_m3_HT: float = 22000.0   # 제올라이트 더 비쌈
    # SCR 운전 윈도우
    LT_T_window: Tuple[float, float] = (280.0, 420.0)
    HT_T_window: Tuple[float, float] = (380.0, 600.0)
    LT_design_T: float = 350.0
    # GHSV
    GHSV_LT: float = 12000.0     # 1/hr
    GHSV_HT: float = 9000.0      # 1/hr (고온은 좀 낮춤 - 큰 분자 확산)
    # NH3
    NH3_NOx_alpha: float = 1.05
    # CAPEX 단가
    scr_USD_per_m3: float = 35000.0       # 반응기/덕트/AIG 일체 (EPA Cost Manual 기준)
    air_temp_USD_per_kg_s: float = 4500.0  # 외기 혼합 시스템 (블로워, 덕트)
    waste_heat_USD_per_MWth: float = 380000.0  # 폐열 회수 보일러
    include_waste_heat_capex: bool = False  # 폐열 회수 보일러 CAPEX 포함 여부


def gas_volumetric_flow(exh_kg_s: float, T_C: float) -> float:
    """배기 부피 유량 (Nm3/h, 0°C, 1atm 기준 환산)"""
    # 평균 배기 분자량 ~28.5 g/mol
    M = 28.5e-3
    n_mol_s = exh_kg_s / M
    # 표준 22.414 L/mol (0°C, 1atm)
    Nm3_s = n_mol_s * 22.414e-3
    return Nm3_s * 3600.0


def actual_volumetric_flow(exh_kg_s: float, T_C: float) -> float:
    """실제 부피 유량 (m3/h at 운전온도, 1atm)"""
    Nm3_h = gas_volumetric_flow(exh_kg_s, T_C)
    return Nm3_h * (T_C + 273.15) / 273.15


def air_tempering(exh_T_C: float, exh_kg_s: float, target_T_C: float,
                   ambient_T_C: float = 25.0) -> Dict:
    """
    외기 혼합으로 배기를 냉각.
    에너지 보존: m1*cp*T1 + m2*cp*T2 = (m1+m2)*cp*Ttarget
    => m2 = m1 * (T1 - Ttarget) / (Ttarget - T2)
    """
    if exh_T_C <= target_T_C:
        return {"air_kg_s": 0.0, "ratio": 0.0, "mixed_kg_s": exh_kg_s}
    air_kg_s = exh_kg_s * (exh_T_C - target_T_C) / (target_T_C - ambient_T_C)
    return {
        "air_kg_s": air_kg_s,
        "ratio": air_kg_s / exh_kg_s,
        "mixed_kg_s": exh_kg_s + air_kg_s,
    }


def scr_efficiency_curve(T_C: float, tech: str) -> float:
    """
    온도-효율 곡선 (정규화). 운전 윈도우 밖이면 급격히 감소.
    Bell-shape: V/Ti는 360°C 근방 최적, 제올라이트는 480°C 근방 최적.
    """
    if tech == "LT":
        T_opt, sigma = 360.0, 60.0
        T_lo, T_hi = 280.0, 420.0
    else:  # HT
        T_opt, sigma = 480.0, 90.0
        T_lo, T_hi = 380.0, 600.0
    base = math.exp(-((T_C - T_opt) ** 2) / (2 * sigma ** 2))
    # 운전 윈도우 안이면 base 값 사용 (보통 0.85 이상), 밖이면 큰 페널티
    if T_C < T_lo or T_C > T_hi:
        base *= 0.3
    # 정규화: 윈도우 안에서 0.85~0.98 정도
    return min(0.98, 0.78 + 0.20 * base)


def required_NH3_kg_s(exh_kg_s: float, NOx_in_ppm: float, NOx_out_ppm: float,
                      alpha: float) -> float:
    """
    NH3 사용량 (kg/s).
    NOx + NH3 -> N2 + H2O (1:1 몰비, 알파만큼 과잉 주입)
    """
    M_gas = 28.5e-3   # kg/mol
    M_NH3 = 17.0e-3
    n_gas_mol_s = exh_kg_s / M_gas
    delta_NOx_ppm = max(0.0, NOx_in_ppm - NOx_out_ppm)
    n_NOx_mol_s = n_gas_mol_s * delta_NOx_ppm * 1e-6
    n_NH3_mol_s = n_NOx_mol_s * alpha
    return n_NH3_mol_s * M_NH3


def NH3_slip_ppm(alpha: float, target_eff: float) -> float:
    """경험식: alpha 증가/효율 증가 시 slip 증가."""
    base = 1.5
    over = max(0.0, alpha - 1.0) * 30.0   # 5% 과잉 -> +1.5
    eff_pen = max(0.0, target_eff - 0.85) * 25.0  # 90%+ 목표시 slip 증가
    slip = base + over + eff_pen
    return min(slip, 12.0)


def reactor_volume_m3(actual_m3_h: float, GHSV_per_h: float) -> float:
    """반응기 촉매 부피 (m3) = 실제 유량 / GHSV"""
    return actual_m3_h / GHSV_per_h


def waste_heat_recoverable_MWth(exh_T_C: float, target_T_C: float,
                                 exh_kg_s: float) -> float:
    """
    저온 SCR 대비 고온 SCR이 폐열로 회수할 수 있는 추가 열량.
    저온 SCR은 배기를 350°C까지 냉각 후 저온 SCR 운전.
    고온 SCR은 배기 그대로 사용 → SCR 후 ~exh_T_C 유지 → HRSG/스팀 회수 가능.
    """
    cp = 1.10  # kJ/kg·K (배기가스 평균)
    delta_T = max(0.0, exh_T_C - target_T_C)
    Q_kW = exh_kg_s * cp * delta_T
    return Q_kW / 1000.0


def annualize(capex: float, life_yr: int, r: float) -> float:
    """CRF 기반 연간화"""
    if r == 0:
        return capex / life_yr
    crf = (r * (1 + r) ** life_yr) / ((1 + r) ** life_yr - 1)
    return capex * crf


def compute_case(inp: Inputs, tech: str) -> Dict:
    """
    tech in {"LT", "HT"}
    LT = Air Tempering + 저온 SCR
    HT = 고온 SCR 직접 운전
    """
    # 1) SCR 입구 온도 결정
    if tech == "LT":
        scr_T = inp.LT_design_T
        temp = air_tempering(inp.exh_T_C, inp.exh_kg_s, scr_T)
        air_kg_s = temp["air_kg_s"]
        scr_kg_s = temp["mixed_kg_s"]
        # 외기 혼합으로 NOx 농도가 희석됨
        dilute = inp.exh_kg_s / scr_kg_s
        NOx_in_at_scr = inp.NOx_in_ppm * dilute
        # 목표 출구도 동일 기준에서 환산 (15% O2 기준 동일)
        NOx_out_at_scr = inp.NOx_out_ppm * dilute
    else:
        scr_T = inp.exh_T_C
        air_kg_s = 0.0
        scr_kg_s = inp.exh_kg_s
        NOx_in_at_scr = inp.NOx_in_ppm
        NOx_out_at_scr = inp.NOx_out_ppm

    # 2) 효율 / NH3
    target_eff = max(0.0, 1.0 - NOx_out_at_scr / max(NOx_in_at_scr, 1e-6))
    eff_capability = scr_efficiency_curve(scr_T, tech)
    eff_actual = min(eff_capability, target_eff if target_eff > 0 else eff_capability)
    nh3_kg_s = required_NH3_kg_s(scr_kg_s, NOx_in_at_scr, NOx_out_at_scr, inp.NH3_NOx_alpha)
    slip = NH3_slip_ppm(inp.NH3_NOx_alpha, eff_actual)

    # 3) 반응기 크기
    GHSV = inp.GHSV_LT if tech == "LT" else inp.GHSV_HT
    actual_m3_h = actual_volumetric_flow(scr_kg_s, scr_T)
    cat_vol = reactor_volume_m3(actual_m3_h, GHSV)

    # 4) CAPEX
    scr_capex = cat_vol * inp.scr_USD_per_m3
    if tech == "LT":
        air_capex = air_kg_s * inp.air_temp_USD_per_kg_s
    else:
        air_capex = 0.0
    # 폐열 회수: 고온 SCR 사용 시 배기를 그대로 보존하므로 추가 폐열보일러로 회수 가능
    # (옵션: 폐열보일러 CAPEX를 SCR 비교 범위에 포함할지 여부 - 통상은 별개 자산으로 보고 OPEX 크레딧만 계산)
    if tech == "HT":
        Q_MW = waste_heat_recoverable_MWth(inp.exh_T_C, 200.0, inp.exh_kg_s)
        wh_capex = (Q_MW * inp.waste_heat_USD_per_MWth) if inp.include_waste_heat_capex else 0.0
        wh_recoverable_MW = Q_MW
    else:
        wh_capex = 0.0
        wh_recoverable_MW = 0.0
    capex_total = scr_capex + air_capex + wh_capex

    # 5) OPEX (연간)
    annual_s = inp.annual_hours * 3600.0
    nh3_annual_kg = nh3_kg_s * annual_s
    nh3_cost = nh3_annual_kg * inp.nh3_USD_per_kg
    cat_life = inp.catalyst_life_LT_yr if tech == "LT" else inp.catalyst_life_HT_yr
    cat_unit = inp.catalyst_USD_per_m3_LT if tech == "LT" else inp.catalyst_USD_per_m3_HT
    cat_replacement_annual = (cat_vol * cat_unit) / cat_life
    # Air Tempering 블로워 전력 (대략 1.5 kW per kg/s 외기)
    if tech == "LT":
        blower_kW = air_kg_s * 1.5
        elec_cost = blower_kW * inp.annual_hours * inp.elec_USD_per_kWh
    else:
        elec_cost = 0.0
    # 폐열 회수 크레딧 (고온 SCR만)
    if tech == "HT":
        # 폐열로 스팀 생산 (효율 70%) → 스팀 단가만큼 절감
        steam_t_h = wh_recoverable_MW * 1000.0 / 2700.0 * 0.70   # 대략 2700 kJ/kg 잠열
        steam_credit_raw = steam_t_h * inp.annual_hours * inp.steam_USD_per_t
    else:
        steam_credit_raw = 0.0
    # 압력손실 보상 (촉매층 추가 ΔP)
    dp_kW = (cat_vol ** 0.5) * 8.0   # 경험식
    dp_cost = dp_kW * inp.annual_hours * inp.elec_USD_per_kWh

    gross_opex = nh3_cost + cat_replacement_annual + elec_cost + dp_cost
    # 보수적 처리: 스팀 크레딧이 gross OPEX를 초과해 음수가 되지 않도록 cap
    # (폐열보일러 별개 자산이므로 NOx 제어 OPEX를 0 미만으로 만들지 않음)
    steam_credit = min(steam_credit_raw, gross_opex) if not inp.include_waste_heat_capex else steam_credit_raw
    opex_annual = gross_opex - steam_credit

    # 6) NOx 제거량
    M_gas = 28.5e-3
    M_NO2 = 46.0e-3
    n_gas_mol_s = inp.exh_kg_s / M_gas
    NOx_removed_kg_s = n_gas_mol_s * (inp.NOx_in_ppm - inp.NOx_out_ppm) * 1e-6 * M_NO2
    NOx_removed_kg_yr = NOx_removed_kg_s * annual_s

    # 7) LCOC (Levelized Cost of NOx Control)
    annual_capex = annualize(capex_total, inp.plant_life_yr, inp.discount_rate)
    total_annual_cost = annual_capex + opex_annual
    LCOC = total_annual_cost / max(NOx_removed_kg_yr, 1e-6)

    return {
        "tech": tech,
        "scr_T_C": scr_T,
        "air_kg_s": air_kg_s,
        "scr_kg_s": scr_kg_s,
        "eff_capability": eff_capability * 100.0,
        "eff_actual": eff_actual * 100.0,
        "NH3_kg_s": nh3_kg_s,
        "NH3_kg_yr": nh3_annual_kg,
        "NH3_slip_ppm": slip,
        "cat_vol_m3": cat_vol,
        "scr_capex": scr_capex,
        "air_capex": air_capex,
        "wh_capex": wh_capex,
        "wh_recoverable_MW": wh_recoverable_MW,
        "capex_total": capex_total,
        "annual_capex": annual_capex,
        "nh3_cost": nh3_cost,
        "cat_replacement_annual": cat_replacement_annual,
        "elec_cost": elec_cost,
        "dp_cost": dp_cost,
        "steam_credit": steam_credit,
        "opex_annual": opex_annual,
        "NOx_removed_kg_yr": NOx_removed_kg_yr,
        "total_annual_cost": total_annual_cost,
        "LCOC": LCOC,
    }


# ============================================================================
# UI 헬퍼
# ============================================================================
def kpi_card(label: str, value: str, delta: str = "", delta_kind: str = "neutral"):
    cls = {"good": "delta-good", "bad": "delta-bad", "neutral": "delta-neutral"}[delta_kind]
    delta_html = f'<div class="{cls}">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <h4>{label}</h4>
            <div class="value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt_USD(v: float) -> str:
    if abs(v) >= 1e6:
        return f"${v/1e6:,.2f}M"
    if abs(v) >= 1e3:
        return f"${v/1e3:,.1f}K"
    return f"${v:,.0f}"


# ============================================================================
# 사이드바 입력
# ============================================================================
with st.sidebar:
    st.markdown("## ⚙️ 입력 조건")

    with st.expander("🔥 GT 배기 조건", expanded=True):
        gt_preset = st.selectbox(
            "GT 모델 프리셋",
            list(GT_TYPICAL.keys()),
            index=list(GT_TYPICAL.keys()).index("범용 평균값"),
        )
        preset = GT_TYPICAL[gt_preset]

        exh_T_C = st.slider(
            "GT 배기온도 (°C)", 400, 650, int(preset["exh_T_C"]), step=5,
            help="단순사이클 GT 배기 온도. 통상 400~600°C (Aeroderivative 낮음, Frame 높음).",
        )
        exh_kg_s = st.slider(
            "배기 유량 (kg/s)", 50, 800, int(preset["exh_kg_s"]), step=10,
            help="GT 배기 질량유량. 100MW급 ~250 kg/s.",
        )
        NOx_in_ppm = st.slider(
            "입구 NOx (ppmvd @ 15% O₂)", 5, 80, int(preset["NOx_DLN_ppm"]), step=1,
            help="DLN/DLE 연소기 출구 NOx. 통상 9~25 ppm.",
        )
        NOx_out_ppm = st.slider(
            "목표 출구 NOx (ppmvd @ 15% O₂)", 1.0, 10.0, 2.5, step=0.5,
            help="BACT/LAER 기준 일반적으로 2.0~3.0 ppm.",
        )

    with st.expander("⏱️ 운전·재무", expanded=True):
        annual_hours = st.slider("연간 가동시간 (h)", 1000, 8760, 4000, step=200)
        plant_life_yr = st.slider("설비 수명 (년)", 10, 30, 20, step=1)
        discount_rate = st.slider("할인율 (%)", 3.0, 15.0, 7.0, step=0.5) / 100.0

    with st.expander("💰 단가", expanded=True):
        nh3_USD_per_kg = st.number_input("NH₃ 단가 (USD/kg)", 0.2, 2.0, 0.55, step=0.05)
        elec_USD_per_kWh = st.number_input("전력 단가 (USD/kWh)", 0.03, 0.30, 0.08, step=0.01)
        steam_USD_per_t = st.number_input("스팀 단가 (USD/ton)", 5.0, 80.0, 25.0, step=1.0)

    with st.expander("🧪 SCR 촉매·반응기"):
        cat_life_LT = st.slider("저온 V/Ti 촉매 수명 (년)", 2.0, 8.0, 4.0, step=0.5)
        cat_life_HT = st.slider("고온 제올라이트 수명 (년)", 1.5, 6.0, 3.0, step=0.5)
        cat_USD_LT = st.number_input("V/Ti 촉매 단가 (USD/m³)", 4000.0, 20000.0, 9000.0, step=500.0)
        cat_USD_HT = st.number_input("제올라이트 단가 (USD/m³)", 10000.0, 40000.0, 22000.0, step=1000.0)
        GHSV_LT = st.number_input("저온 SCR GHSV (1/h)", 6000.0, 20000.0, 12000.0, step=500.0)
        GHSV_HT = st.number_input("고온 SCR GHSV (1/h)", 4000.0, 18000.0, 9000.0, step=500.0)
        alpha = st.slider("NH₃/NOx 몰비 (α)", 0.95, 1.20, 1.05, step=0.01)

    with st.expander("🏗️ CAPEX 단가"):
        scr_USD_per_m3 = st.number_input("SCR 반응기 (USD/m³ 촉매)", 15000.0, 120000.0, 35000.0, step=2500.0,
                                          help="EPA Cost Manual 기준 SCR 총 CAPEX는 GT 출력당 $28~110/kW. 촉매부피 환산.")
        air_temp_USD = st.number_input("Air Tempering (USD per kg/s 외기)", 1500.0, 12000.0, 4500.0, step=500.0)
        wh_USD = st.number_input("폐열 회수 보일러 (USD/MWth)", 100000.0, 800000.0, 380000.0, step=20000.0)
        include_wh_capex = st.checkbox(
            "폐열 회수 보일러 CAPEX를 고온 SCR 비용에 포함",
            value=False,
            help="기본은 폐열보일러를 별개 자산으로 보고 OPEX 크레딧만 인정. "
                 "체크 시 보일러 CAPEX도 NOx 제어 비용에 합산.",
        )

inp = Inputs(
    exh_T_C=exh_T_C,
    exh_kg_s=exh_kg_s,
    NOx_in_ppm=NOx_in_ppm,
    NOx_out_ppm=NOx_out_ppm,
    annual_hours=annual_hours,
    plant_life_yr=plant_life_yr,
    discount_rate=discount_rate,
    nh3_USD_per_kg=nh3_USD_per_kg,
    elec_USD_per_kWh=elec_USD_per_kWh,
    steam_USD_per_t=steam_USD_per_t,
    catalyst_life_LT_yr=cat_life_LT,
    catalyst_life_HT_yr=cat_life_HT,
    catalyst_USD_per_m3_LT=cat_USD_LT,
    catalyst_USD_per_m3_HT=cat_USD_HT,
    GHSV_LT=GHSV_LT,
    GHSV_HT=GHSV_HT,
    NH3_NOx_alpha=alpha,
    scr_USD_per_m3=scr_USD_per_m3,
    air_temp_USD_per_kg_s=air_temp_USD,
    waste_heat_USD_per_MWth=wh_USD,
    include_waste_heat_capex=include_wh_capex,
)

# 두 케이스 계산
res_LT = compute_case(inp, "LT")
res_HT = compute_case(inp, "HT")

# ============================================================================
# 메인 헤더
# ============================================================================
st.title("🌬️ SCGT NOx Control Benchmark")
st.markdown(
    "**단순사이클 가스터빈** 배기가스 NOx 제어: "
    "`Air Tempering + 저온 SCR` vs `고온 SCR 직접` 비교 툴"
)
st.caption(
    "데이터 출처: EPA RBLC, EPA AP-42 §3.1, EPRI 3002022688/3002030747/3002030748, "
    "GE/Siemens/MHI 공개 사양"
)

# ============================================================================
# 탭 구성
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 종합비교", "⚡ 에너지분해", "💵 경제성", "📈 트렌드"])

# ----------------------------------------------------------------------------
# Tab 1: 종합비교
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("핵심 지표 비교")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        delta = res_HT["LCOC"] - res_LT["LCOC"]
        kind = "good" if delta < 0 else "bad"
        kpi_card(
            "LCOC (저온 SCR)",
            f"${res_LT['LCOC']:.2f}/kg-NOx",
            f"고온 SCR Δ {delta:+.2f}",
            "neutral",
        )
    with c2:
        kpi_card(
            "LCOC (고온 SCR)",
            f"${res_HT['LCOC']:.2f}/kg-NOx",
            f"vs 저온 {(-delta):+.2f}",
            "good" if delta > 0 else "bad",
        )
    with c3:
        kpi_card(
            "NOx 제거량 (연간)",
            f"{res_LT['NOx_removed_kg_yr']/1000:,.0f} t/yr",
            f"제거효율 {(1-NOx_out_ppm/NOx_in_ppm)*100:.1f}%",
            "neutral",
        )
    with c4:
        kpi_card(
            "고온 SCR 폐열 회수",
            f"{res_HT['wh_recoverable_MW']:.1f} MWth",
            f"≈ {res_HT['steam_credit']/1e6:.2f}M USD/yr 절감",
            "good",
        )

    st.markdown("---")

    # 비교 테이블
    df = pd.DataFrame({
        "지표": [
            "SCR 운전온도 (°C)",
            "혼합 후 유량 (kg/s)",
            "외기 혼합량 (kg/s)",
            "SCR 효율 - 능력 (%)",
            "SCR 효율 - 실제 (%)",
            "NH₃ 사용량 (kg/h)",
            "NH₃ slip (ppm)",
            "촉매 부피 (m³)",
            "SCR CAPEX (M USD)",
            "Air Tempering CAPEX (M USD)",
            "폐열회수 CAPEX (M USD)",
            "총 CAPEX (M USD)",
            "연간 OPEX (M USD/yr)",
            "LCOC (USD/kg-NOx)",
        ],
        "Air Temp + 저온 SCR": [
            f"{res_LT['scr_T_C']:.0f}",
            f"{res_LT['scr_kg_s']:.1f}",
            f"{res_LT['air_kg_s']:.1f}",
            f"{res_LT['eff_capability']:.1f}",
            f"{res_LT['eff_actual']:.1f}",
            f"{res_LT['NH3_kg_s']*3600:.2f}",
            f"{res_LT['NH3_slip_ppm']:.1f}",
            f"{res_LT['cat_vol_m3']:.1f}",
            f"{res_LT['scr_capex']/1e6:.2f}",
            f"{res_LT['air_capex']/1e6:.2f}",
            "0.00",
            f"{res_LT['capex_total']/1e6:.2f}",
            f"{res_LT['opex_annual']/1e6:.2f}",
            f"${res_LT['LCOC']:.2f}",
        ],
        "고온 SCR 직접": [
            f"{res_HT['scr_T_C']:.0f}",
            f"{res_HT['scr_kg_s']:.1f}",
            "0.0",
            f"{res_HT['eff_capability']:.1f}",
            f"{res_HT['eff_actual']:.1f}",
            f"{res_HT['NH3_kg_s']*3600:.2f}",
            f"{res_HT['NH3_slip_ppm']:.1f}",
            f"{res_HT['cat_vol_m3']:.1f}",
            f"{res_HT['scr_capex']/1e6:.2f}",
            "0.00",
            f"{res_HT['wh_capex']/1e6:.2f}",
            f"{res_HT['capex_total']/1e6:.2f}",
            f"{res_HT['opex_annual']/1e6:.2f}",
            f"${res_HT['LCOC']:.2f}",
        ],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    # 막대 차트
    fig = go.Figure()
    metrics = ["촉매 부피 (m³)", "총 CAPEX (M USD)", "연간 OPEX (M USD)", "LCOC (USD/kg)"]
    LT_vals = [res_LT["cat_vol_m3"], res_LT["capex_total"]/1e6,
               res_LT["opex_annual"]/1e6, res_LT["LCOC"]]
    HT_vals = [res_HT["cat_vol_m3"], res_HT["capex_total"]/1e6,
               res_HT["opex_annual"]/1e6, res_HT["LCOC"]]

    fig = make_subplots(rows=1, cols=4, subplot_titles=metrics)
    for i, (lt, ht) in enumerate(zip(LT_vals, HT_vals)):
        fig.add_trace(go.Bar(x=["저온+AT"], y=[lt], marker_color=COLOR_LT,
                             showlegend=(i == 0), name="저온 SCR + AT"),
                      row=1, col=i+1)
        fig.add_trace(go.Bar(x=["고온"], y=[ht], marker_color=COLOR_HT,
                             showlegend=(i == 0), name="고온 SCR"),
                      row=1, col=i+1)
    fig.update_layout(template=PLOTLY_TEMPLATE, height=320,
                      margin=dict(l=20, r=20, t=50, b=20),
                      legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 2: 에너지 분해
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("배기 에너지 흐름 비교")
    st.caption("저온 SCR은 외기 혼합으로 배기를 350°C까지 냉각 → 폐열 손실. "
               "고온 SCR은 배기 온도를 유지 → 폐열보일러로 회수 가능.")

    # Sankey-style energy decomposition
    cp = 1.10  # kJ/kg·K
    Q_in = inp.exh_kg_s * cp * (inp.exh_T_C - 25.0) / 1000.0  # MWth (25°C 기준)

    # 저온 SCR 경로
    Q_LT_to_350 = inp.exh_kg_s * cp * (inp.exh_T_C - 350.0) / 1000.0
    Q_LT_air_dilution = res_LT["air_kg_s"] * cp * (350.0 - 25.0) / 1000.0  # 외기로 혼합되는 에너지
    Q_LT_after_scr = res_LT["scr_kg_s"] * cp * (350.0 - 25.0) / 1000.0  # SCR 통과 후 배기 에너지
    Q_LT_recoverable = inp.exh_kg_s * cp * (350.0 - 200.0) / 1000.0  # SCR 후 200°C까지 회수 가능

    # 고온 SCR 경로
    Q_HT_recoverable = res_HT["wh_recoverable_MW"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**저온 SCR + Air Tempering**")
        fig = go.Figure(go.Sankey(
            node=dict(
                label=[
                    f"GT 배기<br>{Q_in:.1f} MWth",
                    "Air Tempering<br>혼합 손실",
                    "SCR 저온 운전",
                    f"폐열회수 가능<br>{Q_LT_recoverable:.1f} MWth",
                    "굴뚝 배출",
                ],
                color=[COLOR_LT, "#9aa0a6", "#94a3b8", "#22c55e", "#ef4444"],
                pad=15, thickness=20,
            ),
            link=dict(
                source=[0, 0, 2, 2],
                target=[1, 2, 3, 4],
                value=[Q_LT_to_350, Q_in - Q_LT_to_350, Q_LT_recoverable,
                       max(0, Q_in - Q_LT_to_350 - Q_LT_recoverable)],
                color=["rgba(154,160,166,0.4)", "rgba(96,165,250,0.4)",
                       "rgba(34,197,94,0.4)", "rgba(239,68,68,0.3)"],
            ),
        ))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=380,
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**고온 SCR 직접**")
        fig = go.Figure(go.Sankey(
            node=dict(
                label=[
                    f"GT 배기<br>{Q_in:.1f} MWth",
                    "고온 SCR 통과<br>온도 유지",
                    f"폐열회수<br>{Q_HT_recoverable:.1f} MWth",
                    "굴뚝 배출",
                ],
                color=[COLOR_HT, "#94a3b8", "#22c55e", "#ef4444"],
                pad=15, thickness=20,
            ),
            link=dict(
                source=[0, 1, 1],
                target=[1, 2, 3],
                value=[Q_in, Q_HT_recoverable, max(0, Q_in - Q_HT_recoverable)],
                color=["rgba(245,158,11,0.4)", "rgba(34,197,94,0.5)",
                       "rgba(239,68,68,0.3)"],
            ),
        ))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=380,
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("온도-효율 곡선")
    Ts = np.linspace(250, 650, 200)
    eff_LT = [scr_efficiency_curve(t, "LT") * 100 for t in Ts]
    eff_HT = [scr_efficiency_curve(t, "HT") * 100 for t in Ts]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Ts, y=eff_LT, name="저온 V₂O₅-WO₃/TiO₂",
                              line=dict(color=COLOR_LT, width=3)))
    fig.add_trace(go.Scatter(x=Ts, y=eff_HT, name="고온 제올라이트",
                              line=dict(color=COLOR_HT, width=3)))
    fig.add_vline(x=res_LT["scr_T_C"], line_dash="dot", line_color=COLOR_LT,
                  annotation_text=f"LT 운전점 {res_LT['scr_T_C']:.0f}°C")
    fig.add_vline(x=res_HT["scr_T_C"], line_dash="dot", line_color=COLOR_HT,
                  annotation_text=f"HT 운전점 {res_HT['scr_T_C']:.0f}°C")
    fig.add_vrect(x0=280, x1=420, fillcolor=COLOR_LT, opacity=0.05, line_width=0,
                  annotation_text="저온 윈도우", annotation_position="top left")
    fig.add_vrect(x0=380, x1=600, fillcolor=COLOR_HT, opacity=0.05, line_width=0,
                  annotation_text="고온 윈도우", annotation_position="top right")
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        xaxis_title="SCR 운전 온도 (°C)",
        yaxis_title="NOx 제거 효율 (%)",
        height=380,
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 3: 경제성
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("CAPEX / OPEX 분해")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**CAPEX 구성**")
        capex_data = pd.DataFrame({
            "항목": ["SCR 반응기", "Air Tempering", "폐열 회수 보일러"],
            "저온+AT": [res_LT["scr_capex"], res_LT["air_capex"], 0],
            "고온": [res_HT["scr_capex"], 0, res_HT["wh_capex"]],
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(name="저온+AT", x=capex_data["항목"],
                              y=capex_data["저온+AT"]/1e6, marker_color=COLOR_LT))
        fig.add_trace(go.Bar(name="고온", x=capex_data["항목"],
                              y=capex_data["고온"]/1e6, marker_color=COLOR_HT))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, barmode="group",
            yaxis_title="CAPEX (M USD)", height=340,
            legend=dict(orientation="h", y=1.15),
        )
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
                              y=[v/1e6 for v in LT_opex], marker_color=COLOR_LT))
        fig.add_trace(go.Bar(name="고온", x=opex_items,
                              y=[v/1e6 for v in HT_opex], marker_color=COLOR_HT))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, barmode="group",
            yaxis_title="연간 비용 (M USD/yr)", height=340,
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("연도별 누적 비용 (NPV 기반)")
    years = np.arange(0, plant_life_yr + 1)
    LT_cum = np.zeros(len(years))
    HT_cum = np.zeros(len(years))
    LT_cum[0] = res_LT["capex_total"]
    HT_cum[0] = res_HT["capex_total"]
    for i in range(1, len(years)):
        LT_cum[i] = LT_cum[i-1] + res_LT["opex_annual"] / (1 + discount_rate) ** i
        HT_cum[i] = HT_cum[i-1] + res_HT["opex_annual"] / (1 + discount_rate) ** i

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=LT_cum/1e6, mode="lines+markers",
                              name="저온 SCR + AT", line=dict(color=COLOR_LT, width=3)))
    fig.add_trace(go.Scatter(x=years, y=HT_cum/1e6, mode="lines+markers",
                              name="고온 SCR", line=dict(color=COLOR_HT, width=3)))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        xaxis_title="연도", yaxis_title="누적 비용 (M USD)",
        height=380, legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("LCOC 민감도 분석")

    sens_param = st.selectbox(
        "민감도 분석 변수",
        ["연간 가동시간", "NH₃ 단가", "스팀 단가", "고온 촉매 수명"],
    )

    if sens_param == "연간 가동시간":
        xs = np.linspace(1000, 8000, 30)
        LT_y, HT_y = [], []
        for x in xs:
            tmp = Inputs(**{**inp.__dict__, "annual_hours": x})
            LT_y.append(compute_case(tmp, "LT")["LCOC"])
            HT_y.append(compute_case(tmp, "HT")["LCOC"])
        x_label = "연간 가동시간 (h)"
    elif sens_param == "NH₃ 단가":
        xs = np.linspace(0.2, 1.5, 30)
        LT_y, HT_y = [], []
        for x in xs:
            tmp = Inputs(**{**inp.__dict__, "nh3_USD_per_kg": x})
            LT_y.append(compute_case(tmp, "LT")["LCOC"])
            HT_y.append(compute_case(tmp, "HT")["LCOC"])
        x_label = "NH₃ 단가 (USD/kg)"
    elif sens_param == "스팀 단가":
        xs = np.linspace(5, 70, 30)
        LT_y, HT_y = [], []
        for x in xs:
            tmp = Inputs(**{**inp.__dict__, "steam_USD_per_t": x})
            LT_y.append(compute_case(tmp, "LT")["LCOC"])
            HT_y.append(compute_case(tmp, "HT")["LCOC"])
        x_label = "스팀 단가 (USD/ton)"
    else:
        xs = np.linspace(1.5, 6.0, 20)
        LT_y, HT_y = [], []
        for x in xs:
            tmp = Inputs(**{**inp.__dict__, "catalyst_life_HT_yr": x})
            LT_y.append(compute_case(tmp, "LT")["LCOC"])
            HT_y.append(compute_case(tmp, "HT")["LCOC"])
        x_label = "고온 촉매 수명 (년)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=LT_y, mode="lines",
                              name="저온 SCR + AT", line=dict(color=COLOR_LT, width=3)))
    fig.add_trace(go.Scatter(x=xs, y=HT_y, mode="lines",
                              name="고온 SCR", line=dict(color=COLOR_HT, width=3)))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        xaxis_title=x_label, yaxis_title="LCOC (USD/kg-NOx)",
        height=380, legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 4: 트렌드 (LIT 데이터)
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("실측 성능 데이터 (EPA RBLC + EPRI)")
    st.caption("문헌·BACT/LAER 사례 데이터. 각 점을 클릭해서 상세 확인.")

    df_lit = pd.DataFrame(LIT_RBLC)
    df_lit["기술 분류"] = df_lit["tech"].apply(
        lambda t: "고온 SCR" if "고온" in t else "저온 SCR"
    )

    # 산점도: 운전온도 vs NOx 제거효율
    fig = px.scatter(
        df_lit,
        x="exh_T_C", y="NOx_red_pct",
        color="기술 분류",
        size="NH3_slip_ppm",
        hover_data=["facility", "turbine", "outlet_NOx_ppmvd_15O2", "year", "source"],
        color_discrete_map={"저온 SCR": COLOR_LT, "고온 SCR": COLOR_HT},
        labels={"exh_T_C": "SCR 운전온도 (°C)", "NOx_red_pct": "NOx 제거효율 (%)"},
    )
    # 현재 케이스 마커 추가
    fig.add_trace(go.Scatter(
        x=[res_LT["scr_T_C"]], y=[res_LT["eff_actual"]],
        mode="markers+text", marker=dict(size=20, symbol="star", color="#ffffff",
                                          line=dict(color=COLOR_LT, width=3)),
        text=["현재 LT"], textposition="top center", name="현재 입력 (LT)",
    ))
    fig.add_trace(go.Scatter(
        x=[res_HT["scr_T_C"]], y=[res_HT["eff_actual"]],
        mode="markers+text", marker=dict(size=20, symbol="star", color="#ffffff",
                                          line=dict(color=COLOR_HT, width=3)),
        text=["현재 HT"], textposition="top center", name="현재 입력 (HT)",
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=460,
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("문헌 데이터 테이블")
    st.dataframe(
        df_lit[["facility", "turbine", "mode", "tech", "exh_T_C",
                "outlet_NOx_ppmvd_15O2", "NOx_red_pct", "NH3_slip_ppm",
                "year", "source"]].rename(columns={
            "facility": "사업장", "turbine": "GT", "mode": "운전",
            "tech": "기술", "exh_T_C": "온도(°C)",
            "outlet_NOx_ppmvd_15O2": "출구NOx(ppm)",
            "NOx_red_pct": "효율(%)", "NH3_slip_ppm": "Slip(ppm)",
            "year": "연도", "source": "출처",
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown("---")
    st.subheader("AP-42 §3.1 미통제 NOx 배출계수 (참고)")
    fig = go.Figure(go.Bar(
        x=list(AP42_NOX_FACTORS.keys()),
        y=list(AP42_NOX_FACTORS.values()),
        marker_color=[COLOR_HT, "#a78bfa", "#ec4899", COLOR_LT],
        text=[f"{v:.3f}" for v in AP42_NOX_FACTORS.values()],
        textposition="outside",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        yaxis_title="NOx 배출계수 (lb/MMBtu)",
        height=360, margin=dict(t=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "📚 **참고문헌 핵심 결론**\n\n"
        "- EPA RBLC BACT 한도: SCGT는 통상 **2.5~3.0 ppmvd@15% O₂**가 BACT, **2.0 ppm**이 LAER 수준\n"
        "- EPRI: 제올라이트 기반 고온 SCR은 **1100°F(593°C)까지 운전 가능**, "
        "Air Tempering 제거로 **폐열 회수율 +3~8% 절대 효율 상승**\n"
        "- V₂O₅-WO₃/TiO₂는 **>420°C에서 V 휘발 + TiO₂ 상변태**로 사용 불가 → "
        "Aeroderivative GT(~480°C)도 air tempering 필요\n"
        "- 일반적 **NH₃/NOx α=1.05**, slip <5 ppm 설계가 BAAQMD/CARB 표준"
    )

# ============================================================================
# 푸터
# ============================================================================
st.markdown("---")
st.caption(
    "🌬️ SCGT NOx Control Benchmark v1.0 · "
    "Methodology: EPA AP-42 §3.1, EPA Cost Manual Ch.2 (SCR), EPRI 3002022688/3002030748 · "
    "본 결과는 개념설계 단계의 비교 검토용이며, 상세 설계는 OEM 견적과 사이트 데이터에 기반해야 함."
)
