import streamlit as st
import joblib
import pandas as pd
import numpy as np

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ChurnGuard - Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS  (dark theme to match design mockups)
# =========================================================

st.markdown("""
<style>

/* ================================
   MAIN APP BACKGROUND
================================ */

[data-testid="stAppViewContainer"] {
    background: #0b0e1a;
}

[data-testid="stHeader"] {
    background: transparent;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

* {
    color: #e5e7eb;
}


/* ================================
   SIDEBAR
================================ */

[data-testid="stSidebar"] {
    background: #05070d !important;
    border-right: 1px solid #1c2030;
}

[data-testid="stSidebar"] > div:first-child {
    background: #05070d !important;
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}


/* ================================
   SIDEBAR NAV (custom pill style)
================================ */

[data-testid="stSidebar"] .stRadio > div {
    gap: 4px;
}

/* hide the native radio circle */
[data-testid="stSidebar"] .stRadio input[type="radio"] {
    display: none;
}

/* each nav option becomes a clickable row */
[data-testid="stSidebar"] .stRadio label {
    display: flex;
    align-items: center;
    width: 100%;
    padding: 10px 14px;
    border-radius: 10px;
    margin-bottom: 2px;
    cursor: pointer;
    transition: 0.15s;
    color: #9ca3af !important;
    font-size: 14.5px;
    font-weight: 500;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: #14172a;
    color: #ffffff !important;
}

/* the label text inside */
[data-testid="stSidebar"] .stRadio label p {
    color: inherit !important;
    font-size: 14.5px !important;
    margin: 0;
}

/* selected option — Streamlit marks the checked one's parent div */
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: linear-gradient(90deg, rgba(124,58,237,0.25), rgba(124,58,237,0.05));
    border: 1px solid rgba(124,58,237,0.4);
    color: #ffffff !important;
}

[data-testid="stSidebar"] .stRadio label:has(input:checked) p {
    color: #ffffff !important;
    font-weight: 700;
}



/* ================================
   TITLE / TEXT
================================ */

.badge-pill {
    display: inline-block;
    background: rgba(124, 58, 237, 0.15);
    border: 1px solid rgba(124, 58, 237, 0.4);
    color: #c4b5fd !important;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 6px 14px;
    border-radius: 20px;
    margin-bottom: 18px;
}

.main-title {
    font-size: 46px;
    font-weight: 800;
    color: #ffffff !important;
    line-height: 1.15;
    margin-bottom: 0px;
}

.main-title-gradient {
    font-size: 46px;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.15;
    margin-bottom: 14px;
}

.subtitle {
    font-size: 16px;
    color: #9ca3af !important;
    margin-bottom: 26px;
    max-width: 560px;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #ffffff !important;
    margin-top: 10px;
    margin-bottom: 4px;
}

.section-sub {
    font-size: 14px;
    color: #9ca3af !important;
    margin-bottom: 20px;
}


/* ================================
   CARDS
================================ */

.card {
    background: #12141f;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #232838;
    margin-bottom: 18px;
    height: 100%;
}

.card h2 {
    color: #ffffff !important;
    margin: 4px 0 2px 0;
}

.card h3 {
    color: #ffffff !important;
    margin-top: 0;
}

.card p {
    color: #9ca3af !important;
}

.small-text {
    font-size: 11px;
    letter-spacing: 0.8px;
    color: #6b7280 !important;
    font-weight: 700;
}

.mini-card {
    background: #12141f;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #232838;
    height: 100%;
}

.mini-card .icon-box {
    font-size: 20px;
    margin-bottom: 8px;
}

.mini-card b {
    color: #ffffff !important;
    font-size: 15px;
}

.mini-card p {
    color: #9ca3af !important;
    font-size: 13px;
    margin-top: 4px;
}

.feature-card {
    background: #12141f;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #232838;
}

.feature-card .icon-circle {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    background: #1c2030;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    margin-bottom: 10px;
}

.feature-card b {
    color: #ffffff !important;
    font-size: 15px;
}

.feature-card p {
    color: #9ca3af !important;
    font-size: 13px;
    margin-top: 4px;
}


/* ================================
   HERO ORB
================================ */

.hero-orb-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 260px;
}

.hero-orb {
    width: 200px;
    height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 30%, #a78bfa, #7c3aed 60%, rgba(124,58,237,0) 75%);
    box-shadow: 0 0 90px 25px rgba(124, 58, 237, 0.25);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 60px;
}


/* ================================
   HOW IT WORKS STEP FLOW
================================ */

.step-flow {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin: 25px 0 30px 0;
}

.step-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    flex: 1;
    padding: 0 6px;
}

.step-circle {
    width: 46px;
    height: 46px;
    border-radius: 50%;
    background: #1c2030;
    border: 1px solid #2b3146;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    margin-bottom: 10px;
}

.step-item b {
    color: #ffffff !important;
    font-size: 13px;
}

.step-item p {
    color: #9ca3af !important;
    font-size: 11.5px;
    margin-top: 4px;
}

.step-arrow {
    color: #4b5563 !important;
    font-size: 20px;
    padding-top: 12px;
}


/* ================================
   CTA BANNER
================================ */

.cta-banner {
    background: linear-gradient(90deg, rgba(124,58,237,0.18), rgba(236,72,153,0.12));
    border: 1px solid rgba(124,58,237,0.35);
    border-radius: 14px;
    padding: 22px 26px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 30px;
}

.cta-banner h3 {
    color: #ffffff !important;
    margin: 0 0 4px 0;
    font-size: 18px;
}

.cta-banner p {
    color: #c4b5fd !important;
    margin: 0;
    font-size: 13px;
}


/* ================================
   INPUT CONTAINERS
================================ */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #12141f !important;
    border: 1px solid #232838 !important;
    border-radius: 14px !important;
    padding: 15px !important;
}

div[data-baseweb="input"] {
    background: #1a1d2b !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] > div {
    background: #1a1d2b !important;
    border-radius: 8px !important;
    border-color: #2b3146 !important;
}

input, textarea {
    color: #ffffff !important;
}

label, .stMarkdown p {
    color: #d1d5db !important;
}

label p {
    color: #d1d5db !important;
    font-size: 13px !important;
}


/* ================================
   BUTTONS
================================ */

.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 10px;
    border: none;
    background: linear-gradient(90deg, #7c3aed, #6d28d9);
    color: white !important;
    font-size: 15px;
    font-weight: 700;
    transition: 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #6d28d9, #5b21b6);
    color: white !important;
}

.stButton > button p {
    color: white !important;
}

.secondary-btn > button {
    background: #1c2030 !important;
    border: 1px solid #2b3146 !important;
}


/* ================================
   RESULT CARD
================================ */

.result-card {
    background: #12141f;
    padding: 24px;
    border-radius: 14px;
    border: 1px solid #232838;
    margin-bottom: 18px;
}

.result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.result-header h3 {
    color: #ffffff !important;
    margin: 0;
    font-size: 18px;
}

.completed-tag {
    color: #22c55e !important;
    font-size: 12.5px;
    font-weight: 600;
}

.avatar-circle {
    width: 46px;
    height: 46px;
    border-radius: 50%;
    background: linear-gradient(135deg, #a78bfa, #7c3aed);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white !important;
    font-weight: 700;
    font-size: 15px;
}

.risk-pill {
    display: inline-block;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 3px 10px;
    border-radius: 12px;
    margin-bottom: 4px;
}

.risk-high {
    color: #f87171 !important;
    font-size: 20px;
    font-weight: 800;
}

.risk-medium {
    color: #fbbf24 !important;
    font-size: 20px;
    font-weight: 800;
}

.risk-low {
    color: #4ade80 !important;
    font-size: 20px;
    font-weight: 800;
}

.risk-pill-high {
    background: rgba(248, 113, 113, 0.15);
    color: #f87171 !important;
}

.risk-pill-medium {
    background: rgba(251, 191, 36, 0.15);
    color: #fbbf24 !important;
}

.risk-pill-low {
    background: rgba(74, 222, 128, 0.15);
    color: #4ade80 !important;
}

.risk-desc {
    color: #9ca3af !important;
    font-size: 13px;
    margin-top: 4px;
}

.metric-box {
    background: #0e101a;
    border: 1px solid #232838;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
}

.metric-box .small-text {
    display: block;
    margin-bottom: 6px;
}

.metric-val-red {
    font-size: 22px;
    font-weight: 800;
    color: #f87171 !important;
}

.metric-val-green {
    font-size: 22px;
    font-weight: 800;
    color: #4ade80 !important;
}

.metric-val-blue {
    font-size: 22px;
    font-weight: 800;
    color: #60a5fa !important;
}

.metric-val-blue span {
    font-size: 13px;
    color: #9ca3af !important;
    font-weight: 500;
}


/* ================================
   REASON / RECOMMENDATION LISTS
================================ */

.reason-item {
    color: #d1d5db !important;
    font-size: 13.5px;
    padding: 7px 0;
    border-bottom: 1px solid #1c2030;
}

.reason-item:last-child {
    border-bottom: none;
}

.reason-item span.dot {
    color: #f87171 !important;
    margin-right: 8px;
}

.recommendation {
    color: #d1d5db !important;
    font-size: 13.5px;
    padding: 7px 0;
    border-bottom: 1px solid #1c2030;
}

.recommendation:last-child {
    border-bottom: none;
}

.recommendation span.check {
    color: #4ade80 !important;
    margin-right: 8px;
}


/* ================================
   SIDEBAR RADIO
================================ */

[data-testid="stSidebar"] .stRadio label {
    color: white !important;
}


/* ================================
   MISC
================================ */

hr {
    border-color: #1c2030 !important;
}

[data-testid="stSlider"] [role="slider"] {
    background-color: #ef4444 !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_models():

    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    gender_encoder = joblib.load("gender_encoder.pkl")
    contract_encoder = joblib.load("contract_encoder.pkl")
    payment_encoder = joblib.load("payment_encoder.pkl")

    return (
        model,
        scaler,
        gender_encoder,
        contract_encoder,
        payment_encoder
    )


model, scaler, gender_encoder, contract_encoder, payment_encoder = load_models()


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_churn(
    age,
    gender,
    tenure,
    monthly_charges,
    contract,
    payment_method,
    total_charges
):

    customer = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Contract": contract,
        "PaymentMethod": payment_method,
        "TotalCharges": total_charges
    }])

    # Scale numerical features
    numerical_scaled = scaler.transform(
        customer[
            [
                "Age",
                "Tenure",
                "MonthlyCharges",
                "TotalCharges"
            ]
        ]
    )

    # Encode categorical features
    gender_encoded = gender_encoder.transform(
        customer[["Gender"]]
    )

    contract_encoded = contract_encoder.transform(
        customer[["Contract"]]
    )

    payment_encoded = payment_encoder.transform(
        customer[["PaymentMethod"]]
    )

    # Combine features
    X_new = np.hstack([
        numerical_scaled,
        gender_encoded,
        contract_encoded,
        payment_encoded
    ])

    # Preserve training feature order
    X_new = pd.DataFrame(
        X_new,
        columns=model.feature_names_in_
    )

    prediction = model.predict(X_new)[0]

    probability = model.predict_proba(X_new)[0][1]

    return prediction, probability
    


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding:15px 0 30px 0;">
            <div style="font-size:45px;">📊</div>
            <h1 style="font-size:25px; margin:0;">
                ChurnGuard
            </h1>
            <p style="color:#9ca3af !important;">
                Customer Intelligence
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Navigation")

    page = st.radio(
        "",
        ["🏠 Overview", "🔮 Predict Churn", "ℹ️ About Model"],
        key="page",
        label_visibility="collapsed"
    )



    st.markdown("---")

    st.markdown(
        """
        <div style="
            background:#1f2937;
            padding:15px;
            border-radius:10px;
            text-align:center;
        ">
            <div style="font-size:13px; color:#9ca3af !important;">
                MODEL STATUS
            </div>
            <div style="
                font-size:16px;
                font-weight:700;
                color:#22c55e !important;
                margin-top:5px;
            ">
                ● Online
            </div>
            <div style="font-size:11.5px; color:#6b7280 !important; margin-top:4px;">
                Model is active and ready for predictions.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )




# =========================================================
# REUSABLE: AI CHURN INTELLIGENCE SECTION
# =========================================================

def render_ai_intelligence_section():

    st.markdown(
        '<div class="section-title">✨ AI Churn Intelligence</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-sub">Powerful capabilities to help you understand, predict and reduce customer churn.</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="mini-card">
                <div class="icon-box">🗨️</div>
                <b>Early Warning Detection</b>
                <p>Detect subtle behavioral changes and patterns that indicate customers may be at risk of churning.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="mini-card">
                <div class="icon-box">💡</div>
                <b>Explainable Predictions</b>
                <p>Understand the key factors driving each prediction with clear and interpretable insights.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="mini-card">
                <div class="icon-box">🎁</div>
                <b>Retention Recommendations</b>
                <p>Get AI-powered, actionable strategies to engage customers and reduce churn effectively.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)


# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "🏠 Overview":

    hero_left, hero_right = st.columns([1.4, 1])

    with hero_left:

        st.markdown(
            '<div class="badge-pill">✨ AI-POWERED RETENTION</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="main-title">Customer Churn</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="main-title-gradient">Prediction System</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="subtitle">'
            'An intelligent machine learning system that identifies customers '
            'at risk of churn before they leave.'
            '</div>',
            unsafe_allow_html=True
        )



    with hero_right:
        st.markdown(
            """
            <div class="hero-orb-wrap">
                <div class="hero-orb">🧠</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="icon-circle">📈</div>
                <b>Predict</b>
                <p>Smart predictions powered by ML</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="icon-circle">🔵</div>
                <b>Detect</b>
                <p>Identify at-risk customers early</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="icon-circle">🟡</div>
                <b>Protect</b>
                <p>Reduce churn and retain customers</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            """
            <div class="feature-card">
                <div class="icon-circle">📊</div>
                <b>Grow</b>
                <p>Drive loyalty and business growth</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)

    render_ai_intelligence_section()


# =========================================================
# PREDICTION PAGE
# =========================================================

elif page == "🔮 Predict Churn":

    left_col, right_col = st.columns([1, 1])

    # -------------------------------------------------
    # LEFT: INPUT FORM
    # -------------------------------------------------

    with left_col:

        with st.container(border=True):

            st.markdown("### Predict Churn")
            st.markdown(
                '<p style="color:#9ca3af; margin-top:-8px;">'
                'Provide customer details to get a churn prediction.</p>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input(
                    "Age",
                    min_value=18,
                    max_value=100,
                    value=35,
                    step=1
                )

            with col2:
                gender = st.selectbox(
                    "Gender",
                    ["Male", "Female"]
                )

            tenure = st.slider(
                "Tenure (months)",
                min_value=0,
                max_value=72,
                value=24
            )

            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            )

            col3, col4 = st.columns(2)

            with col3:
                monthly_charges = st.number_input(
                    "Monthly Charges ($)",
                    min_value=0.0,
                    value=89.85,
                    step=1.0
                )

            with col4:
                total_charges = st.number_input(
                    "Total Charges ($)",
                    min_value=0.0,
                    value=2145.60,
                    step=1.0
                )

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Credit card",
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer"
                ]
            )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            predict_button = st.button(
                "✨  Analyze Customer",
                use_container_width=True,
                type="primary"
            )

    # -------------------------------------------------
    # RIGHT: PREDICTION RESULT
    # -------------------------------------------------

    with right_col:

        if predict_button:

            prediction, probability = predict_churn(
                age,
                gender,
                tenure,
                monthly_charges,
                contract,
                payment_method,
                total_charges
            )

            confidence_score = max(probability, 1 - probability) * 100

            # Determine risk level
            if probability >= 0.70:
                risk_level = "HIGH RISK"
                risk_class = "risk-high"
                risk_pill_class = "risk-pill-high"
                risk_icon = "⚠️"
                risk_desc = "This customer is highly likely to churn."

            elif probability >= 0.40:
                risk_level = "MEDIUM RISK"
                risk_class = "risk-medium"
                risk_pill_class = "risk-pill-medium"
                risk_icon = "⚠️"
                risk_desc = "This customer shows moderate signs of churn risk."

            else:
                risk_level = "LOW RISK"
                risk_class = "risk-low"
                risk_pill_class = "risk-pill-low"
                risk_icon = "✅"
                risk_desc = "This customer is likely to stay."

            initials = (gender[0] + "C").upper()

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-header">
                        <h3>AI Prediction Result</h3>
                        <div class="completed-tag">✓ Prediction Completed</div>
                    </div>
                    <div style="display:flex; align-items:center; gap:16px; margin-top:18px;">
                        <div class="avatar-circle">{initials}</div>
                        <div>
                            <div class="risk-pill {risk_pill_class}">RISK LEVEL</div>
                            <div class="{risk_class}">{risk_level} {risk_icon}</div>
                        </div>
                    </div>
                    <div class="risk-desc">{risk_desc}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            m1, m2, m3 = st.columns(3)

            with m1:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <span class="small-text">CHURN PROBABILITY</span>
                        <div class="metric-val-red">{probability:.0%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with m2:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <span class="small-text">CONFIDENCE SCORE</span>
                        <div class="metric-val-green">{confidence_score:.0f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with m3:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <span class="small-text">RISK SCORE</span>
                        <div class="metric-val-blue">{probability:.2f}<span>/1.00</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

            
            # Recommendations
            if probability >= 0.70:
                recommendations = [
                    "Offer a special discount for a yearly contract upgrade",
                    "Provide loyalty rewards to increase engagement",
                    "Assign a dedicated customer success agent",
                    "Proactive follow-up and satisfaction survey"
                ]
            elif probability >= 0.40:
                recommendations = [
                    "Monitor the customer's engagement and billing behavior",
                    "Consider offering incentives for a longer-term contract",
                    "Send personalized offers based on customer usage"
                ]
            else:
                recommendations = [
                    "Continue regular customer engagement",
                    "Offer loyalty benefits to maintain the relationship",
                    "Monitor the customer for changes in future behavior"
                ]

            recs_html = "".join(
                f'<div class="recommendation"><span class="check">✓</span>{rec}</div>'
                for rec in recommendations
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <h3 style="color:#ffffff; font-size:16px; margin-bottom:10px;">
                        🎁 AI Retention Recommendations
                    </h3>
                    {recs_html}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="result-card" style="text-align:center; padding:60px 24px;">
                    <div style="font-size:40px; margin-bottom:10px;">🔮</div>
                    <h3 style="color:#ffffff; font-size:16px;">No Prediction Yet</h3>
                    <p style="color:#9ca3af; font-size:13.5px;">
                        Fill in the customer details on the left and click
                        <b>Analyze Customer</b> to see the AI prediction result.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# ABOUT MODEL PAGE
# =========================================================

elif page == "ℹ️ About Model":

    st.markdown(
        '<div class="main-title" style="font-size:34px;">About the Model</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">'
        'Information about the customer churn prediction pipeline.'
        '</div>',
        unsafe_allow_html=True
    )

    render_ai_intelligence_section()

    st.markdown(
        '<div class="section-title">How ChurnGuard Works</div>',
        unsafe_allow_html=True
    )

    steps = [
        ("👤", "1. Customer Data", "Collects customer demographic, subscription and behavioral data."),
        ("🛠️", "2. Feature Engineering", "Processes and transforms raw data into meaningful features."),
        ("🧠", "3. ML Analysis", "Machine learning model analyzes patterns and customer behavior."),
        ("🎯", "4. Risk Scoring", "Calculates churn probability score for each customer."),
        ("🎁", "5. Retention Strategy", "Provides actionable recommendations to retain at-risk customers."),
    ]

    step_html = '<div class="step-flow">'
    for i, (icon, title, desc) in enumerate(steps):
        step_html += f"""
            <div class="step-item">
                <div class="step-circle">{icon}</div>
                <b>{title}</b>
                <p>{desc}</p>
            </div>
        """
        if i < len(steps) - 1:
            step_html += '<div class="step-arrow">→</div>'
    step_html += "</div>"

    st.markdown(step_html, unsafe_allow_html=True)

    cta_col1, cta_col2 = st.columns([3, 1])

    with cta_col1:
        st.markdown(
            """
            <div class="cta-banner" style="width:100%;">
                <div>
                    <h3>Proactive Retention. Stronger Relationships. Sustainable Growth.</h3>
                    <p>ChurnGuard helps you act before it's too late.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with cta_col2:
       st.info("👈 Use the sidebar menu to navigate to **Predict Churn**.")

    st.markdown(
        '<div class="section-title">Customer Churn Prediction</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-sub">An intelligent machine learning system for identifying customers at risk of churn.</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="card">
                <div class="small-text">SYSTEM</div>
                <h2>ChurnGuard</h2>
                <div class="small-text" style="text-transform:none;">ML Prediction System</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="card">
                <div class="small-text">PREDICTION</div>
                <h2>Binary</h2>
                <div class="small-text" style="text-transform:none;">Churn / Stay</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="card">
                <div class="small-text">MODEL</div>
                <h2>ML Model</h2>
                <div class="small-text" style="text-transform:none;">Classification</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            """
            <div class="card">
                <div class="small-text">OUTPUT</div>
                <h2>Risk Score</h2>
                <div class="small-text" style="text-transform:none;">Probability Based</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">How it works</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="card">
                <h3>1️⃣ Customer Data</h3>
                <p>Enter demographic, subscription and payment information about the customer.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="card">
                <h3>2️⃣ Machine Learning</h3>
                <p>The trained machine learning model processes the customer's information.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="card">
                <h3>3️⃣ Risk Assessment</h3>
                <p>The system returns churn probability, risk level and possible reasons.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info(
        "💡 Go to **Predict Churn** from the sidebar to assess a customer."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="card">
                <h3>📥 Input Features</h3>
                <p><b>Age</b> — Customer age</p>
                <p><b>Gender</b> — Customer gender</p>
                <p><b>Tenure</b> — Number of months with the company</p>
                <p><b>Monthly Charges</b> — Monthly customer charge</p>
                <p><b>Contract</b> — Customer contract type</p>
                <p><b>Payment Method</b> — Customer payment method</p>
                <p><b>Total Charges</b> — Total amount charged</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="card">
                <h3>⚙️ Prediction Pipeline</h3>
                <p>1. Customer input is collected.</p>
                <p>2. Numerical features are scaled.</p>
                <p>3. Categorical features are encoded.</p>
                <p>4. Features are passed to the trained model.</p>
                <p>5. Churn probability is calculated.</p>
                <p>6. Customer risk level is displayed.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info(
        "The probability shown by this application is a machine learning "
        "estimate and should be used as a decision-support signal rather "
        "than a guaranteed prediction."
    )