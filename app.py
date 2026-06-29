import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import sqlite3
import smtplib
import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from groq import Groq

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

st.set_page_config(page_title="CreditLens", page_icon="🔍", layout="centered")

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    xgb_model   = joblib.load('models/xgb_model.pkl')
    xgb_imputer = joblib.load('models/xgb_imputer.pkl')
    with open('models/trained_columns.json') as f:
        trained_columns = json.load(f)
    shap_explainer = None
    if SHAP_AVAILABLE:
        try:
            shap_explainer = joblib.load('models/shap_explainer.pkl')
        except Exception:
            pass
    return xgb_model, xgb_imputer, shap_explainer, trained_columns

xgb_model, xgb_imputer, shap_explainer, trained_columns = load_models()
imputer_features = [c for c in trained_columns if c != 'loan_int_rate']
client           = Groq(api_key=st.secrets["GROQ_API_KEY"])
BEST_THRESHOLD   = 0.643

# ── Database ──────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect('database.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        submitted_at TEXT, customer_name TEXT, customer_email TEXT,
        age INTEGER, income INTEGER, home_ownership TEXT,
        emp_length REAL, cred_hist INTEGER, loan_intent TEXT,
        loan_amnt INTEGER, prior_default INTEGER, decision TEXT,
        approved_amnt REAL, interest_rate REAL, is_tailored INTEGER, base_rate REAL,
        shap_explanation TEXT, generated_email TEXT)''')
    conn.commit()
    conn.close()

def save_application(cd, result):
    conn = sqlite3.connect('database.db')
    conn.execute('''INSERT INTO applications 
        (submitted_at, customer_name, customer_email, age, income,
         home_ownership, emp_length, cred_hist, loan_intent, loan_amnt,
         prior_default, decision, approved_amnt, interest_rate,
         is_tailored, base_rate, shap_explanation, generated_email)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        cd['name'], cd['email'],
        cd['person_age'], cd['person_income'],
        cd['person_home_ownership'], cd['person_emp_length'],
        cd['cb_person_cred_hist_length'], cd['loan_intent'],
        cd['loan_amnt'], cd['cb_person_default_on_file'],
        result['status'],
        float(result['approved_loan_amnt']),
        float(result['calculated_rate']),
        int(result['is_tailored']),
        float(result['base_rate']),
        result['shap_explanation'],
        result['email']))
    conn.commit()
    conn.close()

init_db()

# ── Email ─────────────────────────────────────────────────────────────────────
def send_email(to_addr, name, body):
    sender = st.secrets["EMAIL_SENDER"]
    pw     = st.secrets["EMAIL_APP_PASSWORD"]
    msg    = MIMEMultipart("alternative")
    msg['Subject'] = f"Your Loan Application Update — {name}"
    msg['From']    = f"CreditLens <{sender}>"
    msg['To']      = to_addr
    html = f"""<html><body style="font-family:Arial,sans-serif;background:#f9f9f9;padding:20px;">
    <div style="max-width:600px;margin:auto;background:white;border:1px solid #ddd;
                border-radius:10px;padding:35px;">
        <h2 style="color:#1a4e8a;">🔍 CreditLens</h2>
        <p style="color:#888;font-size:13px;">AI-Powered Credit Decisioning</p>
        <hr style="border:none;border-top:1px solid #eee;margin:20px 0;">
        <p style="line-height:1.8;">{body.replace(chr(10), '<br>')}</p>
        <hr style="border:none;border-top:1px solid #eee;margin:20px 0;">
        <p style="font-size:11px;color:#bbb;">Automated message from CreditLens. Do not reply.</p>
    </div></body></html>"""
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(sender, pw)
        s.sendmail(sender, to_addr, msg.as_string())

# ── Helpers ───────────────────────────────────────────────────────────────────
def format_inr(n):
    s = str(int(n))
    r = ",".join([s[x-2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]])
    return r

def get_shap(row, explainer, cols, top_n=3):
    if not SHAP_AVAILABLE or explainer is None:
        return "SHAP explanation available in local deployment.", []
    try:
        vals   = explainer.shap_values(row)
        series = pd.Series(vals[0], index=cols)
        top    = series.nlargest(top_n)
        lines  = []
        for feat, val in top.items():
            av = row[feat].values[0]
            d  = "increased" if val > 0 else "decreased"
            lines.append(
                f"- **{feat.replace('_',' ').title()}** = {av:.2f} "
                f"-> {d} default risk by {abs(val):.4f} SHAP units"
            )
        return "\n".join(lines), list(top.items())
    except Exception as e:
        return f"SHAP unavailable: {e}", []

# ── Core Pipeline ─────────────────────────────────────────────────────────────
def process(cd):
    df  = pd.DataFrame([cd])
    mi  = pd.DataFrame(columns=trained_columns)
    mi.loc[0] = 0

    intent  = df['loan_intent'][0].upper()
    income  = df['person_income'][0]
    req_amt = df['loan_amnt'][0]
    deflt   = df['cb_person_default_on_file'][0]

    mi['person_age']                 = df['person_age']
    mi['person_income']              = income
    mi['cb_person_default_on_file']  = deflt
    mi['cb_person_cred_hist_length'] = df['cb_person_cred_hist_length']

    home_col = f"person_home_ownership_{df['person_home_ownership'][0].upper()}"
    if home_col in mi.columns:
        mi[home_col] = 1

    intent_col = f"loan_intent_{intent}"
    if intent_col in mi.columns:
        mi[intent_col] = 1

    emp_col = f"employment_status_{'EMPLOYED' if df['person_emp_length'][0] > 0 else 'UNEMPLOYED'}"
    if emp_col in mi.columns:
        mi[emp_col] = 1

    is_tailored = False
    app_amt     = req_amt
    status      = ""
    rate        = 0.0
    shap_txt    = "N/A"
    shap_fac    = []

    if income <= 0:
        status  = "REJECTED"
        app_amt = 0
    else:
        mult     = 5.0 if intent == "HOME" else 3.0 if intent == "EDUCATION" else 2.0
        max_loan = income * mult

        if req_amt > max_loan:
            app_amt               = max_loan
            is_tailored           = True
            mi['loan_percent_income'] = mult
        else:
            mi['loan_percent_income'] = req_amt / income

        mi['loan_amnt'] = app_amt
        base_rate       = xgb_imputer.predict(mi[imputer_features])[0]

        if intent == "HOME":
            rate = max(7.10,  min(10.50, base_rate))
        elif intent == "EDUCATION":
            rate = max(8.00,  min(16.00, base_rate))
        else:
            rate = max(11.00, min(24.00, base_rate))

        mi['loan_int_rate'] = rate
        prob                = xgb_model.predict_proba(mi)[:, 1][0]
        status              = "REJECTED" if prob >= BEST_THRESHOLD else "APPROVED"
        if status == "REJECTED":
            is_tailored = False

        shap_txt, shap_fac = get_shap(mi, shap_explainer, list(trained_columns))

    shap_ctx = ""
    if shap_fac:
        factors  = ", ".join([f[0].replace('_', ' ') for f in shap_fac])
        shap_ctx = f"Primary risk factors: {factors}."

    prompt = f"""You are a Senior Loan Officer at CreditLens, an AI credit decisioning platform.
Write a professional email to a customer about their loan application.
Customer: {cd['name']}
Rate: {f'{rate:.2f}%' if rate > 0 else 'N/A'}
Requested: Rs.{format_inr(req_amt)} | Approved: Rs.{format_inr(app_amt)}
Tailored Down: {is_tailored}
Risk Context: {shap_ctx}
Decision: {status}
Rules:
- Do NOT state numerical income.
- Do NOT mention rate if REJECTED.
- APPROVED and not tailored: warm and congratulatory.
- APPROVED and tailored: Conditional Approval with revised amount.
- REJECTED: empathetic but firm, no technical jargon.
- Sign off as The CreditLens Risk Team. Under 150 words."""

    try:
        r = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant"
        )
        email_txt = r.choices[0].message.content
    except Exception as e:
        email_txt = f"Error generating email: {e}"

    return {
        "status"            : status,
        "approved_loan_amnt": app_amt,
        "calculated_rate"   : rate,
        "base_rate"         : float(base_rate) if income > 0 else 0.0,
        "is_tailored"       : is_tailored,
        "shap_explanation"  : shap_txt,
        "email"             : email_txt,
}

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔍 CreditLens")
st.caption("AI-Powered Credit Decisioning — XGBoost · SHAP · LLaMA 3.1")
st.divider()

with st.form("loan_application"):
    st.subheader("👤 Applicant Details")
    c1, c2 = st.columns(2)
    with c1:
        name   = st.text_input("Full Name")
        email  = st.text_input("Email Address")
        age    = st.number_input("Age", min_value=18, max_value=90, value=30)
        income = st.number_input("Annual Income (₹)", min_value=1, value=500000, step=10000)
    with c2:
        home   = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        emp    = st.number_input("Years of Employment", min_value=0.0, max_value=60.0, value=3.0, step=0.5)
        cred   = st.number_input("Credit History Length (years)", min_value=0, max_value=30, value=4)

    st.subheader("💰 Loan Details")
    c3, c4 = st.columns(2)
    with c3:
        intent = st.selectbox("Loan Purpose",
                              ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOME", "DEBTCONSOLIDATION"])
    with c4:
        amt    = st.number_input("Loan Amount Requested (₹)", min_value=1000, value=200000, step=5000)

    st.subheader("📋 Credit History")
    deflt = st.radio("Any prior loan default on record?", options=[0, 1],
                     format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)

    sub = st.form_submit_button("🚀 Submit Application", use_container_width=True)

if sub:
    if not name.strip():
        st.error("Please enter your full name.")
    elif "@" not in email or "." not in email:
        st.error("Please enter a valid email address.")
    else:
        with st.spinner("Analysing your application..."):
            cd = {
                "name": name, "email": email,
                "person_age": age, "person_income": income,
                "person_home_ownership": home, "person_emp_length": emp,
                "loan_intent": intent, "loan_amnt": amt,
                "cb_person_cred_hist_length": cred,
                "cb_person_default_on_file": deflt,
            }
            result = process(cd)
            save_application(cd, result)
            try:
                send_email(email, name, result['email'])
                sent = True
            except Exception:
                sent = False

        st.divider()
        st.success("Application Received!")
        st.markdown(
            f"Thank you **{name}**, your application has been submitted.\n\n"
            f"A detailed response has been sent to **{email}**.\n\n"
            f"Please check your inbox and spam folder."
        )
        if not sent:
            st.warning("Note: Email delivery issue. Please contact support.")
