# SCGT NOx Control Benchmark

단순사이클 가스터빈(SCGT) 배기가스 NOx 제어 기술 비교 툴.

## 비교 대상

1. **Air Tempering + 저온 SCR** (V₂O₅-WO₃/TiO₂, 280~420°C)
2. **고온 SCR 직접 운전** (제올라이트 계열, 400~600°C)

## 주요 기능

- GT 배기 조건 입력 → SCR 효율, NH₃ 사용량, 촉매 부피 자동 계산
- CAPEX/OPEX 분해 및 LCOC (USD/kg-NOx removed) 산출
- 폐열 회수 가능량 정량화
- EPA RBLC 실측 데이터 12개 포인트와 현재 입력값 비교
- 4종 변수 민감도 분석 (가동시간, NH₃ 단가, 스팀 단가, 촉매 수명)

## 데이터 출처

- EPA RBLC (BACT/LAER 데이터베이스)
- EPA AP-42 §3.1 (Stationary Gas Turbines)
- EPA Cost Manual Chapter 2 (SCR)
- EPRI 3002022688 / 3002030747 / 3002030748
- GE LM6000/LMS100, Siemens SGT-A65, MHI 공개 사양

## 실행 (로컬)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 배포 (Streamlit Community Cloud)

1. 이 저장소를 fork 또는 clone
2. [share.streamlit.io](https://share.streamlit.io) 접속 → GitHub 로그인
3. **New app** → 저장소/브랜치/`app.py` 선택 → **Deploy**

## 면책

본 툴의 결과는 개념설계 단계의 비교 검토용입니다. 상세 설계는 OEM 견적과 사이트 데이터에 기반해야 합니다.
