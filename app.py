import streamlit as st
import pandas as pd
from preprocessing import load_data, preprocess_data
from model import train_model

# ======================
# 🎨 PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="💳",
    layout="wide"
)

# ======================
# 🎨 ADVANCED CSS (PRO UI)
# ======================
st.markdown("""
<style>
/* Background */
.main {
    background: linear-gradient(135deg, #0e1117, #111827);
}

/* Title */
.title {
    font-size: 48px;
    font-weight: bold;
    color: white;
}
.highlight {
    color: #22c55e;
}

/* Subtitle */
.subtitle {
    font-size: 20px;
    color: #9ca3af;
}

/* Card */
.card {
    background: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.4);
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #22c55e, #16a34a);
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0f172a;
}

/* Metric Style */
.metric {
    font-size: 18px;
    color: #9ca3af;
}
.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ======================
# 📌 HEADER
# ======================
st.markdown("""
<div class="title">
💳 CREDIT <span class="highlight">RISK</span> ANALYSIS DASHBOARD
</div>
<div class="subtitle">
Predict Credit Risk. Make Smarter Decisions.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ======================
# 📂 LOAD DATA
# ======================
@st.cache_data
def load_and_process():
    df = load_data()
    return preprocess_data(df)

df = load_and_process()

# ======================
# 📊 TRAIN MODEL
# ======================
@st.cache_resource
def get_model(data):
    return train_model(data)

model, report, features = get_model(df)

# ======================
# 📊 METRIC CARDS
# ======================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="card">
        <div class="metric">Total Records</div>
        <div class="metric-value">{len(df)}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <div class="metric">Features</div>
        <div class="metric-value">{len(features)}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card">
        <div class="metric">Model</div>
        <div class="metric-value">Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ======================
# 📊 MAIN LAYOUT
# ======================
left, right = st.columns([2, 1])

# ======================
# 📊 LEFT PANEL
# ======================
with left:

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("📈 Model Performance")
    st.code(report)

# ======================
# 🔮 RIGHT PANEL (FORM STYLE)
# ======================
with right:

    st.markdown("### 🔮 Enter Customer Details")

    age = st.slider("Age", 18, 100, 30)
    credit_amount = st.number_input("Credit Amount", 100, 20000, 3000)
    duration = st.slider("Duration (months)", 1, 72, 12)

    sex = st.selectbox("Sex", ["male", "female"])
    housing = st.selectbox("Housing", ["own", "rent", "free"])

    input_data = pd.DataFrame({
        'Age': [age],
        'Credit amount': [credit_amount],
        'Duration': [duration],
        'Sex': [sex],
        'Housing': [housing]
    })

    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=features, fill_value=0)

    st.markdown("### 🎯 Prediction")

    if st.button("Predict Risk"):
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.markdown("""
            <div style="background:#7f1d1d;padding:15px;border-radius:10px;">
            ⚠️ <b style="color:white;">High Risk Customer</b><br>
            <span style="color:#fecaca;">Loan approval not recommended</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#14532d;padding:15px;border-radius:10px;">
            ✅ <b style="color:white;">Low Risk Customer</b><br>
            <span style="color:#bbf7d0;">Safe to approve loan</span>
            </div>
            """, unsafe_allow_html=True)

# ======================
# 📌 FOOTER
# ======================
st.markdown("---")
st.markdown(
    "<center style='color:gray;'>Built with ❤️ using Streamlit | Machine Learning Project</center>",
    unsafe_allow_html=True
)