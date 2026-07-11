"""
AI 기반 보행약자 맞춤형 이동시간 예측 서비스
- SDG 10 (불평등 감소), SDG 11 (지속가능한 도시와 공동체) 연계
- 대회 제출용 프로토타입 (Streamlit)

핵심 차별점:
기존 지도앱은 '평균적인 성인'의 보행 속도를 기준으로 도착 예상 시간을 제공한다.
이 서비스는 사용자의 나이·보행 특성·개인 이동 이력을 반영하여
"나에게 맞는 실제 도착 예상 시간"을 예측하고, 적정 출발 시각까지 추천한다.
"""

import streamlit as st  # type: ignore[import]
import pandas as pd
import datetime
import os

# ----------------------------------------------------------------------------
# 기본 설정
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AI 맞춤 이동시간 예측", page_icon="🚶", layout="centered")

HISTORY_FILE = "travel_history.csv"

# 사용자 특성별 보정 계수 (기준: 일반 성인 = 1.0)
# 실제 서비스에서는 이 계수를 AI 모델(회귀/학습 기반)이 개인 이력 데이터로 계속 갱신한다.
BASE_COEFFICIENTS = {
    "청소년": 0.9,
    "성인": 1.0,
    "노인": 1.45,
}

MOBILITY_COEFFICIENTS = {
    "일반 보행자": 1.0,
    "보행 속도가 느린 사용자": 1.25,
    "목발 사용자": 1.55,
    "휠체어 사용자": 1.35,
}

# 안전 여유시간 (병목 구간, 신호 대기, 횡단보도 등 고려)
SAFETY_MARGIN_MIN = 3

# 개인화 전환 속도 조절 상수
# n(누적 실제 기록 건수)이 늘어날수록 '통계 기반 계수'의 비중은 줄고
# '개인 실측 데이터로 학습한 계수'의 비중이 커진다.
# weight_personal = n / (n + PERSONALIZATION_K)
# K가 작을수록 적은 기록만으로도 빠르게 개인화된다.
PERSONALIZATION_K = 3

# ----------------------------------------------------------------------------
# 이동 이력 저장/불러오기 (CSV 기반 - 대회 프로토타입용 간단 구현)
# ----------------------------------------------------------------------------
def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=[
        "기록시각", "출발지", "목적지", "연령대", "보행특성",
        "지도앱_기본시간(분)", "AI_예측시간(분)", "실제_소요시간(분)"
    ])

def save_history(df):
    df.to_csv(HISTORY_FILE, index=False)

if "history_df" not in st.session_state:
    st.session_state.history_df = load_history()

if "personal_ratio" not in st.session_state:
    # 개인이 실제로 기록한 (실제 소요시간 / 지도앱 기본시간)의 평균.
    # 아직 기록이 없으면 None → 이 경우 통계 기반 계수만 사용.
    st.session_state.personal_ratio = None

if "record_count" not in st.session_state:
    st.session_state.record_count = 0


# ----------------------------------------------------------------------------
# AI 예측 로직
# 1단계(통계 기반): 연령대·보행특성별 계수 → "일반적인 노인/장애인 통계"에 근거한 예측
# 2단계(개인화 전환): 실제 이동 기록이 쌓일수록, 그 사람만의 실측 데이터가
#                     통계 기반 계수보다 점점 더 큰 비중을 갖도록 블렌딩한다.
# 실제 구현 시에는 rule_based_coef 산출부를 회귀모델/공공데이터 기반 값으로 대체 가능
# ----------------------------------------------------------------------------
def get_rule_based_coef(age_group, mobility_type):
    """통계(가정) 기반 계수 — 노인/장애인 등 집단 평균치를 반영한 기본 예측."""
    age_coef = BASE_COEFFICIENTS[age_group]
    mobility_coef = MOBILITY_COEFFICIENTS[mobility_type]
    return (age_coef * 0.4) + (mobility_coef * 0.6)


def get_personalization_weight(n_records):
    """실측 기록 수(n)가 늘수록 0→1로 증가하는 개인화 비중."""
    return round(n_records / (n_records + PERSONALIZATION_K), 3)


def predict_travel_time(base_minutes, age_group, mobility_type, personal_ratio, n_records):
    rule_based_coef = get_rule_based_coef(age_group, mobility_type)
    weight_personal = get_personalization_weight(n_records)

    if personal_ratio is None:
        # 아직 실측 기록이 없다면 통계 기반 계수만 사용
        final_coef = rule_based_coef
        weight_personal = 0.0
    else:
        # 기록이 쌓일수록 개인 실측 비율의 영향력이 커짐 (weight_personal → 1)
        final_coef = (1 - weight_personal) * rule_based_coef + weight_personal * personal_ratio

    predicted = base_minutes * final_coef + SAFETY_MARGIN_MIN
    return round(predicted, 1), round(rule_based_coef, 3), weight_personal


def update_personal_model(history_df):
    """실제 이동시간을 지도앱 기본시간과 비교해, '그 사람만의' 실측 보정 비율을 학습한다.
    (AI 예측치가 아니라 지도앱 원본 시간 대비 비율로 학습해야, 통계 계수의 오차가
    개인 모델에 누적되어 왜곡되는 것을 막을 수 있다.)"""
    valid = history_df.dropna(subset=["실제_소요시간(분)"])
    n = len(valid)
    st.session_state.record_count = n
    if n >= 1:
        ratios = valid["실제_소요시간(분)"].astype(float) / valid["지도앱_기본시간(분)"].astype(float)
        st.session_state.personal_ratio = round(ratios.mean(), 3)
    else:
        st.session_state.personal_ratio = None


update_personal_model(st.session_state.history_df)


# ----------------------------------------------------------------------------
# 사이드바 - 사용자 프로필 등록 (기능 1)
# ----------------------------------------------------------------------------
st.sidebar.header("👤 내 프로필 설정")
age_group = st.sidebar.radio("연령대를 선택하세요", list(BASE_COEFFICIENTS.keys()), index=1)
mobility_type = st.sidebar.radio("보행 특성을 선택하세요", list(MOBILITY_COEFFICIENTS.keys()), index=0)

st.sidebar.markdown("---")
_w = get_personalization_weight(st.session_state.record_count)
st.sidebar.markdown("📊 **예측 방식 구성비**")
st.sidebar.progress(_w if st.session_state.personal_ratio is not None else 0.0)
if st.session_state.personal_ratio is None:
    st.sidebar.caption(
        f"통계 기반 계수 100% (누적 실측 기록: {st.session_state.record_count}건)\n\n"
        "아직 실제 이동시간 기록이 없어, 연령대·보행특성 통계 계수로만 예측합니다."
    )
else:
    st.sidebar.caption(
        f"통계 기반 계수 {round((1-_w)*100)}% + 개인 실측 데이터 {round(_w*100)}%\n\n"
        f"(누적 실측 기록: {st.session_state.record_count}건, "
        f"개인 실측 비율: {st.session_state.personal_ratio})\n\n"
        "기록이 쌓일수록 '나만의 실측 데이터' 비중이 점점 커집니다."
    )

# ----------------------------------------------------------------------------
# 메인 화면
# ----------------------------------------------------------------------------
st.title("🚶 AI 맞춤형 이동시간 예측 서비스")
st.markdown(
    "지도 앱은 **평균적인 성인 기준**으로 도착 시간을 계산합니다.\n\n"
    "이 서비스는 **당신의 나이와 보행 특성**을 반영해 "
    "**실제로 걸리는 시간**을 예측합니다."
)

st.markdown("### 📍 경로 입력")
col1, col2 = st.columns(2)
with col1:
    origin = st.text_input("출발지", placeholder="예: 우리집")
with col2:
    destination = st.text_input("목적지", placeholder="예: 서울역")

st.markdown("### 🗺️ 지도 앱 기본 정보")
st.caption("실제 서비스에서는 카카오맵/네이버지도/구글지도 API의 거리·기본 예상시간을 자동으로 불러옵니다. "
           "프로토타입에서는 아래에 직접 입력(또는 가상 API 결과 사용)하세요.")

use_dummy_api = st.checkbox("가상 API 데이터 사용 (거리 1.2km 기준 예시)", value=False)

if use_dummy_api:
    distance_km = 1.2
    base_minutes = 15.0
    st.info(f"가상 API 결과 → 거리: {distance_km}km / 지도앱 기본 예상시간: {base_minutes}분")
else:
    base_minutes = st.number_input("지도앱 기본 예상 이동시간(분)", min_value=1.0, max_value=180.0, value=15.0, step=1.0)

st.markdown("### 🕑 약속 시간 (선택)")
use_appointment = st.checkbox("도착해야 하는 약속 시간이 있어요")
appointment_time = None
if use_appointment:
    appointment_time = st.time_input("약속 시간", value=datetime.time(14, 0))

# ----------------------------------------------------------------------------
# 예측 실행 (기능 2, 3)
# ----------------------------------------------------------------------------
if st.button("🔮 AI 맞춤 이동시간 예측하기", type="primary"):
    if not origin or not destination:
        st.warning("출발지와 목적지를 입력해주세요.")
    else:
        predicted_minutes, rule_coef, weight_personal = predict_travel_time(
            base_minutes, age_group, mobility_type,
            st.session_state.personal_ratio, st.session_state.record_count
        )
        diff = round(predicted_minutes - base_minutes, 1)

        st.markdown("---")
        st.markdown("### 결과 비교")
        c1, c2 = st.columns(2)
        c1.metric("지도앱 기본 예상시간", f"{base_minutes:.0f}분")
        c2.metric("AI 맞춤 예측시간", f"{predicted_minutes:.0f}분", delta=f"+{diff:.0f}분" if diff > 0 else f"{diff:.0f}분")

        st.success(
            f"**{age_group} / {mobility_type}** 기준, "
            f"{origin} → {destination} 이동에는 실제로 약 **{predicted_minutes:.0f}분**이 걸릴 것으로 예측됩니다."
        )

        if weight_personal == 0.0:
            st.caption("ℹ️ 아직 실측 기록이 없어 통계 기반 계수(연령대·보행특성)만으로 예측했습니다.")
        else:
            st.caption(
                f"ℹ️ 이번 예측은 통계 기반 계수 {round((1-weight_personal)*100)}% + "
                f"회원님의 실측 이동 데이터 {round(weight_personal*100)}%를 반영했습니다."
            )

        # 출발 추천 시간 계산 (기능 3)
        if use_appointment and appointment_time:
            today = datetime.date.today()
            appt_dt = datetime.datetime.combine(today, appointment_time)
            depart_dt = appt_dt - datetime.timedelta(minutes=predicted_minutes)
            st.markdown("### 🕒 출발 추천 시간")
            st.info(
                f"⏰ 약속 시간 **{appointment_time.strftime('%H:%M')}**에 맞추려면 "
                f"**{depart_dt.strftime('%H:%M')}** 이전에 출발하는 것을 추천합니다."
            )

        # 세션에 마지막 예측 결과 저장 (이후 실제 시간 기록에 사용)
        st.session_state.last_prediction = {
            "출발지": origin,
            "목적지": destination,
            "연령대": age_group,
            "보행특성": mobility_type,
            "지도앱_기본시간(분)": base_minutes,
            "AI_예측시간(분)": predicted_minutes,
        }

# ----------------------------------------------------------------------------
# 이동 이력 기록 (기능 4)
# ----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📝 실제 이동시간 기록하기 (AI 학습용)")
st.caption("실제로 이동을 마친 뒤 걸린 시간을 기록하면, 다음 예측이 더 정확해집니다.")

if "last_prediction" in st.session_state:
    actual_minutes = st.number_input("실제로 걸린 시간(분)", min_value=1.0, max_value=300.0, value=float(st.session_state.last_prediction["AI_예측시간(분)"]), step=1.0)

    if st.button("✅ 이동 이력 저장하기"):
        record = st.session_state.last_prediction.copy()
        record["기록시각"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        record["실제_소요시간(분)"] = actual_minutes

        new_row = pd.DataFrame([record])
        st.session_state.history_df = pd.concat([st.session_state.history_df, new_row], ignore_index=True)
        save_history(st.session_state.history_df)
        update_personal_model(st.session_state.history_df)

        _w_after = get_personalization_weight(st.session_state.record_count)
        st.success(
            f"저장되었습니다! 누적 기록 {st.session_state.record_count}건 → "
            f"이제 예측에서 개인 실측 데이터 비중이 약 {round(_w_after*100)}%로 반영됩니다."
        )
        st.rerun()
else:
    st.info("먼저 위에서 'AI 맞춤 이동시간 예측하기'를 실행해주세요.")

# ----------------------------------------------------------------------------
# 이동 이력 조회
# ----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📚 나의 이동 이력")
if len(st.session_state.history_df) > 0:
    st.dataframe(st.session_state.history_df, use_container_width=True)

    if st.button("🗑️ 이력 전체 삭제"):
        st.session_state.history_df = pd.DataFrame(columns=st.session_state.history_df.columns)
        save_history(st.session_state.history_df)
        st.session_state.personal_ratio = None
        st.session_state.record_count = 0
        st.rerun()
else:
    st.caption("아직 기록된 이동 이력이 없습니다.")

# ----------------------------------------------------------------------------
# 서비스 설명 (하단)
# ----------------------------------------------------------------------------
with st.expander("ℹ️ 이 서비스에 대해 더 알아보기"):
    st.markdown(
        """
**문제의식**
기존 지도앱(네이버지도, 카카오맵, 구글지도)은 평균적인 성인 보행 속도를 기준으로
도착 예상 시간을 계산합니다. 그러나 노인, 목발 사용자, 보행 속도가 느린 사람 등
보행약자는 실제 이동 시간이 이보다 훨씬 오래 걸립니다.

**이 서비스의 차별점**
- 새로운 경로를 찾는 것이 아니라, **같은 경로라도 사람마다 실제 걸리는 시간이 다르다**는
  문제를 해결하는 데 초점을 둡니다.
- 사용자의 나이, 보행 특성, 그리고 실제 이동 이력을 학습하여
  점점 더 정확한 '나만의 이동시간'을 예측합니다.

**예측 방식: 통계 기반 → 개인화 전환**
- 처음에는 노인·장애인 등 집단 통계에 기반한 계수로 예측합니다.
- 실제 이동시간을 기록할 때마다, 그 사람만의 실측 데이터 비중이
  `개인화 비중 = 기록 수 / (기록 수 + 3)` 공식에 따라 점점 커집니다.
- 예: 기록 3건 → 개인화 50%, 기록 10건 → 개인화 약 77%
- 즉 초반에는 "노인이라면 대체로 이 정도 걸린다"는 통계를 참고하고,
  기록이 쌓일수록 "이 사람은 실제로 이렇게 걷는다"는 데이터가 우선합니다.

**SDGs 연계**
- SDG 10 (불평등 감소): 보행약자도 자신의 특성에 맞는 이동 정보를 제공받을 수 있도록 지원
- SDG 11 (지속가능한 도시와 공동체): 모든 시민이 이용 가능한 포용적 이동 서비스 제공
        """
    )
