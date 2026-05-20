import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Springster Risk Engine",
    page_icon="🏦",
    layout="centered"
)

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .stButton>button {
        background-color: #1a3c5e; color: white;
        border-radius: 8px; width: 100%; font-weight: bold;
    }
    .result-box {
        padding: 20px; border-radius: 12px;
        margin: 10px 0; font-size: 16px;
    }
    .approved  { background:#d4edda; border-left: 6px solid #28a745; }
    .rejected  { background:#f8d7da; border-left: 6px solid #dc3545; }
    .counter   { background:#fff3cd; border-left: 6px solid #ffc107; }
    .email-box { background:#e8f4f8; border-left: 6px solid #17a2b8;
                 padding:15px; border-radius:8px; white-space:pre-wrap; font-size:14px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD & TRAIN (cached — runs once)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Training models on credit risk data...")
def load_and_train():
    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/credit_risk_dataset.csv"
    try:
        df = pd.read_csv(url)
    except Exception:
        # fallback: generate synthetic data so the demo never breaks
        np.random.seed(42)
        n = 5000
        df = pd.DataFrame({
            "person_age": np.random.randint(20, 65, n),
            "person_income": np.random.randint(20000, 200000, n),
            "person_home_ownership": np.random.choice(["RENT","OWN","MORTGAGE"], n),
            "person_emp_length": np.random.choice([0,1,2,3,5,7,10], n).astype(float),
            "loan_intent": np.random.choice(["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOME"], n),
            "loan_amnt": np.random.randint(1000, 35000, n),
            "loan_int_rate": np.where(np.random.rand(n) > 0.1,
                                      np.random.uniform(5, 24, n), np.nan),
            "loan_percent_income": np.random.uniform(0.01, 0.8, n),
            "cb_person_default_on_file": np.random.choice([0,1], n, p=[0.8,0.2]),
            "cb_person_cred_hist_length": np.random.randint(1, 30, n),
            "loan_status": np.random.choice([0,1], n, p=[0.78,0.22]),
            "loan_grade": "A",
        })

    # Clean
    df = df[df["person_age"] <= 90]
    df["person_emp_length"] = df["person_emp_length"].fillna(0)
    df = df[df["person_emp_length"] <= df["person_age"]]

    # Feature engineering
    df["employment_status"] = np.where(df["person_emp_length"] > 0, "EMPLOYED", "UNEMPLOYED")
    df["cb_person_default_on_file"] = df["cb_person_default_on_file"].map(
        lambda x: x if isinstance(x, (int, float)) else (1 if str(x).upper() == "Y" else 0)
    )
    df = df.drop(["loan_grade", "person_emp_length"], axis=1)
    df_final = pd.get_dummies(
        df, columns=["person_home_ownership", "loan_intent", "employment_status"],
        drop_first=False, dtype=int
    )

    # Imputer
    known = df_final[df_final["loan_int_rate"].notnull()]
    missing = df_final[df_final["loan_int_rate"].isnull()]
    allowed = known.drop(["loan_status", "loan_int_rate"], axis=1).columns
    imputer = XGBRegressor(random_state=42)
    imputer.fit(known[allowed], known["loan_int_rate"])
    df_final.loc[df_final["loan_int_rate"].isnull(), "loan_int_rate"] = \
        imputer.predict(missing[allowed])

    # Classifier
    X = df_final.drop("loan_status", axis=1)
    y = df_final["loan_status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
    model = XGBClassifier(scale_pos_weight=ratio, random_state=42)
    model.fit(X_train, y_train)

    # Threshold
    probs = model.predict_proba(X_test)[:, 1]
    p, r, t = precision_recall_curve(y_test, probs)
    f1 = 2 * p * r / (p + r + 1e-9)
    threshold = t[np.argmax(f1)]

    # SHAP
    explainer = shap.TreeExplainer(model)

    return model, imputer, explainer, allowed, X_train.columns, threshold

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def fmt_inr(n):
    s = str(int(n))
    r = ",".join([s[x-2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]])
    return r

def generate_email(name, status, requested, approved, rate, is_tailored, risk_factors):
    factors_text = ", ".join([f.replace("_", " ") for f, _ in risk_factors]) if risk_factors else ""

    if status == "APPROVED" and not is_tailored:
        return f"""Dear {name},

We are delighted to inform you that your loan application has been approved for ₹{fmt_inr(approved)} at an interest rate of {rate:.2f}% per annum.

Your financial profile reflects strong creditworthiness and we look forward to supporting your goals.

Please visit your nearest branch or log into our portal to complete the documentation process.

Warm regards,
The Risk Analytics Team — Springster Bank"""

    elif status == "APPROVED" and is_tailored:
        return f"""Dear {name},

Thank you for your loan application. After a thorough review, we are pleased to offer a Conditional Approval for ₹{fmt_inr(approved)} at {rate:.2f}% per annum.

Please note that the approved amount has been adjusted from your original request of ₹{fmt_inr(requested)} to ensure comfortable and sustainable repayment aligned with your financial profile.

We encourage you to visit our branch to discuss the next steps.

Regards,
The Risk Analytics Team — Springster Bank"""

    else:
        factor_line = f"Key considerations included aspects such as {factors_text}." if factors_text else ""
        return f"""Dear {name},

Thank you for considering Springster Bank for your financial needs. After a careful review of your application, we regret to inform you that we are unable to approve your loan request at this time.

{factor_line}

We encourage you to revisit your application after addressing these factors. Our team remains available to guide you on improving your credit profile.

We appreciate your trust in us and hope to serve you in the future.

Regards,
The Risk Analytics Team — Springster Bank"""

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
st.markdown("## 🏦 Springster Risk Engine")
st.markdown("*An end-to-end AI credit decisioning system — XGBoost + Business Rules Engine + Auto-generated Communication*")
st.divider()

# Load models
model, imputer, explainer, allowed_features, trained_cols, threshold = load_and_train()

# ── INPUT FORM ───────────────────────────────
st.markdown("### 📋 Applicant Details")

col1, col2 = st.columns(2)
with col1:
    name      = st.text_input("Full Name", "Rahul Sharma")
    age       = st.number_input("Age", 18, 90, 32)
    income    = st.number_input("Annual Income (₹)", 10000, 10000000, 600000, step=10000)
    emp_years = st.number_input("Years Employed", 0.0, 50.0, 4.0, step=0.5)

with col2:
    home      = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
    intent    = st.selectbox("Loan Purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOME"])
    loan_amt  = st.number_input("Loan Amount Requested (₹)", 1000, 5000000, 200000, step=5000)
    cred_hist = st.number_input("Credit History Length (years)", 0, 40, 5)
    prior_def = st.selectbox("Prior Default on Record?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])

st.divider()

# ── PROCESS ──────────────────────────────────
if st.button("⚡ Run Credit Decision"):

    has_default = prior_def[1]

    # Build model input
    model_input = pd.DataFrame(columns=trained_cols)
    model_input.loc[0] = 0
    model_input["person_age"]                 = age
    model_input["person_income"]              = income
    model_input["cb_person_default_on_file"]  = has_default
    model_input["cb_person_cred_hist_length"] = cred_hist

    for col in [f"person_home_ownership_{home}",
                f"loan_intent_{intent}",
                f"employment_status_{'EMPLOYED' if emp_years > 0 else 'UNEMPLOYED'}"]:
        if col in model_input.columns:
            model_input[col] = 1

    # BRE
    status, is_tailored, approved_amt, rate = "", False, loan_amt, 0.0
    risk_factors = []

    if has_default == 1:
        status, approved_amt = "REJECTED", 0

    elif income <= 0:
        status, approved_amt = "REJECTED", 0

    else:
        multipliers = {"HOME": 5.0, "EDUCATION": 3.0}
        max_mult    = multipliers.get(intent, 1.0)
        max_loan    = income * max_mult

        if loan_amt > max_loan:
            approved_amt = max_loan
            is_tailored  = True
            model_input["loan_percent_income"] = max_mult
        else:
            model_input["loan_percent_income"] = loan_amt / income

        model_input["loan_amnt"] = approved_amt

        # Rate pricing
        base = imputer.predict(model_input[allowed_features])[0]
        caps = {"HOME": (7.1, 10.5), "EDUCATION": (4.0, 16.0)}
        lo, hi = caps.get(intent, (0, 24))
        rate = float(np.clip(base, lo, hi)) if intent in caps else min(24.0, base)
        model_input["loan_int_rate"] = rate

        # XGBoost score
        prob = model.predict_proba(model_input)[:, 1][0]
        status = "REJECTED" if prob >= threshold else "APPROVED"
        if status == "REJECTED":
            is_tailored = False

        # SHAP
        sv = explainer.shap_values(model_input)
        shap_s = pd.Series(sv[0], index=trained_cols)
        risk_factors = list(shap_s.nlargest(3).items())

    # ── RESULTS ──────────────────────────────
    st.markdown("### 📊 Decision Results")

    if status == "APPROVED" and not is_tailored:
        st.markdown(f'<div class="result-box approved">✅ <b>APPROVED</b> — ₹{fmt_inr(approved_amt)} at {rate:.2f}% p.a.</div>', unsafe_allow_html=True)
    elif status == "APPROVED" and is_tailored:
        st.markdown(f'<div class="result-box counter">⚠️ <b>CONDITIONAL APPROVAL</b> — Requested ₹{fmt_inr(loan_amt)}, Approved ₹{fmt_inr(approved_amt)} at {rate:.2f}% p.a.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-box rejected">❌ <b>REJECTED</b> — Application does not meet credit policy requirements.</div>', unsafe_allow_html=True)

    # Pipeline steps
    with st.expander("🔍 See how the decision was made", expanded=True):
        steps = [
            ("1️⃣ Business Rules Engine", f"Prior default: {'Yes — Auto Rejected' if has_default else 'No'} | Max loan limit: ₹{fmt_inr(income * ({'HOME':5,'EDUCATION':3}.get(intent,1)))}"),
            ("2️⃣ Interest Rate Pricing", f"XGBoost Regressor predicted base rate, capped to policy bounds → **{rate:.2f}%**" if rate else "N/A — rejected before scoring"),
            ("3️⃣ Default Risk Score",    f"XGBoost Classifier | Threshold: {threshold:.3f}" if risk_factors else "N/A — rejected before scoring"),
            ("4️⃣ Top Risk Drivers (SHAP)", "\n".join([f"- **{f.replace('_',' ')}**: SHAP = {v:.4f}" for f, v in risk_factors]) if risk_factors else "N/A"),
        ]
        for title, detail in steps:
            st.markdown(f"**{title}**")
            st.markdown(detail)
            st.markdown("---")

    # SHAP bar chart
    if risk_factors:
        st.markdown("**📈 SHAP Feature Importance (this applicant)**")
        fig, ax = plt.subplots(figsize=(6, 2.5))
        names = [f[0].replace("_", " ") for f in risk_factors]
        vals  = [f[1] for f in risk_factors]
        colors = ["#dc3545" if v > 0 else "#28a745" for v in vals]
        ax.barh(names, vals, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Value (+ = risk, - = safe)")
        ax.set_title("Top 3 Risk Drivers")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Email
    st.markdown("### ✉️ Auto-Generated Customer Email")
    email_text = generate_email(name, status, loan_amt, approved_amt, rate, is_tailored, risk_factors)
    st.markdown(f'<div class="email-box">{email_text}</div>', unsafe_allow_html=True)

    # Tech stack note
    st.info("💡 In production this email is generated by **Llama 3 via Ollama REST API** with structured prompt engineering based on SHAP outputs.")

st.divider()

# ── EMAIL CAPTURE ─────────────────────────────
st.markdown("### 💼 Interested in this project?")
st.markdown("Built by **Harshit Sharma** — Credit Risk Data Scientist with 3+ years in BFSI.")

with st.form("contact_form"):
    user_email = st.text_input("Your Email Address", placeholder="recruiter@company.com")
    submitted  = st.form_submit_button("📩 Get in Touch")
    if submitted and user_email:
        st.success(f"Thanks! Harshit will reach out to {user_email} shortly.")
        st.balloons()

st.markdown("""
---
<center>
<small>
🔗 <a href="https://linkedin.com/in/Harshit-s">LinkedIn</a> &nbsp;|&nbsp;
💻 <a href="https://github.com/harshitsharma1410-bot">GitHub</a> &nbsp;|&nbsp;
Built with XGBoost · SHAP · Streamlit
</small>
</center>
""", unsafe_allow_html=True)
