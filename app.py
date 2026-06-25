import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import sqlite3
import smtplib
import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from groq import Groq

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CreditLens",
    page_icon="🔍",
    layout="centered"
)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    xgb_model      = joblib.load('models/xgb_model.pkl')
    xgb_imputer    = joblib.load('models/xgb_imputer.pkl')
    shap_explainer = joblib.load('models/shap_explainer.pkl')
    with open('models/trained_columns.json', 'r') as f:
        trained_columns = json.load(f)
    return xgb_model, xgb_imputer, shap_explainer, trained_columns

xgb_model, xgb_imputer, shap_explainer, trained_columns = load_models()
imputer_features = [c for c in trained_columns if c != 'loan_int_rate']
client           = Groq(api_key=st.secrets["GROQ_API_KEY"])
BEST_THRESHOLD   = 0.643

# ─────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect('database.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS applications (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            submitted_at     TEXT,
            customer_name    TEXT,
            customer_email   TEXT,
            age              INTEGER,
            income           INTEGER,
            home_ownership   TEXT,
            emp_length       REAL,
            cred_hist        INTEGER,
            loan_intent      TEXT,
            loan_amnt        INTEGER,
            prior_default    INTEGER,
            decision         TEXT,
            approved_amnt    REAL,
            interest_rate    REAL,
            is_tailored      INTEGER,
            shap_explanation TEXT,
            generated_email  TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_application(customer_data, result):
    conn = sqlite3.connect('database.db')
    conn.execute('''
        INSERT INTO applications (
            submitted_at, customer_name, customer_email,
            age, income, home_ownership, emp_length, cred_hist,
            loan_intent, loan_amnt, prior_default,
            decision, approved_amnt, interest_rate, is_tailored,
            shap_explanation, generated_email
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', (
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        customer_data['name'],
        customer_data['email'],
        customer_data['person_age'],
        customer_data['person_income'],
        customer_data['person_home_ownership'],
        customer_data['person_emp_length'],
        customer_data['cb_person_cred_hist_length'],
        customer_data['loan_intent'],
        customer_data['loan_amnt'],
        customer_data['cb_person_default_on_file'],
        result['status'],
        float(result['approved_loan_amnt']),
        float(result['calculated_rate']),
        int(result['is_tailored']),
        result['shap_explanation'],
        result['email']
    ))
    conn.commit()
    conn.close()

init_db()

# ─────────────────────────────────────────────
# EMAIL SENDER
# ─────────────────────────────────────────────
def send_email(to_address, customer_name, email_body):
    sender       = st.secrets["EMAIL_SENDER"]
    app_password = st.secrets["EMAIL_APP_PASSWORD"]

    msg            = MIMEMultipart("alternative")
    msg['Subject'] = f"Your Loan Application Update — {customer_name}"
    msg['From']    = f"CreditLens <{sender}>"
    msg['To']      = to_address

    html_body = f"""
    <html><body style="font-family: Arial, sans-serif; padding: 20px; color: #333; background:#f9f9f9;">
        <div style="max-width:600px; margin:auto; background:white; border:1px solid #ddd;
                    border-radius:10px; padding:35px;">
            <h2 style="color:#1a4e8a;">🔍 CreditLens</h2>
            <p style="color:#888; font-size:13px;">AI-Powered Credit Decisioning</p>
            <hr style="border:none; border-top:1px solid #eee; margin:20px 0;">
            <p style="line-height:1.8;">{email_body.replace(chr(10), '<br>')}</p>
            <hr style="border:none; border-top:1px solid #eee; margin:20px 0;">
            <p style="font-size:11px; color:#bbb;">
                This is an automated message from CreditLens. Please do not reply to this email.
            </p>
        </div>
    </body></html>
    """

    msg.attach(MIMEText(html_body, "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, app_password)
        server.sendmail(sender, to_address, msg.as_string())

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def format_inr(number):
    s, *d = str(int(number)).partition(".")
    r = ",".join([s[x-2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]])
    return "".join([r] + d)

def get_local_shap_explanation(model_input_row, explainer, feature_names, top_n=3):
    local_shap       = explainer.shap_values(model_input_row)
    shap_series      = pd.Series(local_shap[0], index=feature_names)
    top_risk_drivers = shap_series.nlargest(top_n)
    lines = []
    for feat, val in top_risk_drivers.items():
        actual_value = model_input_row[feat].values[0]
        direction    = "increased" if val > 0 else "decreased"
        lines.append(
            f"- **{feat.replace('_', ' ').title()}** = {actual_value:.2f} "
            f"→ {direction} default risk by {abs(val):.4f} SHAP units"
        )
    return "\n".join(lines), list(top_risk_drivers.items())

def process_loan_application(customer_data):
    input_df    = pd.DataFrame([customer_data])
    model_input = pd.DataFrame(columns=trained_columns)
    model_input.loc[0] = 0

    intent         = input_df['loan_intent'][0].upper()
    income         = input_df['person_income'][0]
    requested_loan = input_df['loan_amnt'][0]
    has_default    = input_df['cb_person_default_on_file'][0]

    model_input['person_age']                 = input_df['person_age']
    model_input['person_income']              = income
    model_input['cb_person_default_on_file']  = has_default
    model_input['cb_person_cred_hist_length'] = input_df['cb_person_cred_hist_length']

    home_col = f"person_home_ownership_{input_df['person_home_ownership'][0].upper()}"
    if home_col in model_input.columns:
        model_input[home_col] = 1

    intent_col = f"loan_intent_{intent}"
    if intent_col in model_input.columns:
        model_input[intent_col] = 1

    emp_status = 'EMPLOYED' if input_df['person_emp_length'][0] > 0 else 'UNEMPLOYED'
    emp_col    = f"employment_status_{emp_status}"
    if emp_col in model_input.columns:
        model_input[emp_col] = 1

    is_tailored        = False
    approved_loan_amnt = requested_loan
    status             = ""
    calculated_rate    = 0.0
    shap_explanation   = "N/A"
    top_shap_factors   = []

    if income <= 0:
        status             = "REJECTED"
        approved_loan_amnt = 0
    else:
        if intent == "HOME":        max_multiplier = 5.0
        elif intent == "EDUCATION": max_multiplier = 3.0
        else:                       max_multiplier = 2.0

        max_allowed_loan = income * max_multiplier
        if requested_loan > max_allowed_loan:
            approved_loan_amnt                 = max_allowed_loan
            is_tailored                        = True
            model_input['loan_percent_income'] = max_multiplier
        else:
            model_input['loan_percent_income'] = requested_loan / income

        model_input['loan_amnt'] = approved_loan_amnt
        base_rate                = xgb_imputer.predict(model_input[imputer_features])[0]

        if intent == "HOME":        calculated_rate = max(7.10, min(10.50, base_rate))
        elif intent == "EDUCATION": calculated_rate = max(4.00, min(16.00, base_rate))
        else:                       calculated_rate = min(24.00, base_rate)

        model_input['loan_int_rate'] = calculated_rate
        prob                         = xgb_model.predict_proba(model_input)[:, 1][0]

        if prob >= BEST_THRESHOLD:
            status      = "REJECTED"
            is_tailored = False
        else:
            status = "APPROVED"

        shap_explanation, top_shap_factors = get_local_shap_explanation(
            model_input, shap_explainer, feature_names=list(trained_columns)
        )

    display_rate      = f"{calculated_rate:.2f}%" if calculated_rate > 0 else "N/A"
    display_requested = format_inr(requested_loan)
    display_approved  = format_inr(approved_loan_amnt)

    shap_context = ""
    if top_shap_factors:
        factors_str  = ", ".join([f[0].replace('_', ' ') for f in top_shap_factors])
        shap_context = f"The primary risk factors identified were: {factors_str}."

    prompt = f"""
    You are a Senior Loan Officer at CreditLens, a modern AI-powered credit decisioning platform.
    Write a professional email to a customer regarding their loan application.

    Customer Profile:
    - Name: {customer_data['name']}
    - Assigned Rate: {display_rate}
    - Original Requested Amount: ₹{display_requested}
    - Final Approved Amount: ₹{display_approved}
    - Was Amount Tailored Down?: {is_tailored}
    - Risk Analysis Context: {shap_context}

    Bank Decision: {status}

    Instructions:
    - CRITICAL: DO NOT explicitly state the customer's numerical income.
    - CRITICAL: DO NOT mention the Assigned Rate if the Bank Decision is REJECTED.
    - If APPROVED AND NOT TAILORED: Be warm and congratulatory.
    - If APPROVED AND TAILORED DOWN: Offer a Conditional Approval with revised amount.
    - If REJECTED: Be empathetic but firm. Reference risk factors without technical jargon.
    - Sign off as "The CreditLens Risk Team". Keep it under 150 words.
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
        )
        email_text = chat_completion.choices[0].message.content
    except Exception as e:
        email_text = f"Error generating email: {e}"

    return {
        "status"            : status,
        "approved_loan_amnt": approved_loan_amnt,
        "calculated_rate"   : calculated_rate,
        "is_tailored"       : is_tailored,
        "shap_explanation"  : shap_explanation,
        "email"             : email_text,
    }

# ─────────────────────────────────────────────
# UI — CUSTOMER FORM
# ─────────────────────────────────────────────
st.title("🔍 CreditLens")
st.caption("AI-Powered Credit Decisioning — XGBoost · SHAP · LLaMA 3.1")
st.divider()

with st.form("loan_application"):
    st.subheader("👤 Applicant Details")
    col1, col2 = st.columns(2)
    with col1:
        name   = st.text_input("Full Name")
        email  = st.text_input("Email Address")
        age    = st.number_input("Age", min_value=18, max_value=90, value=30)
        income = st.number_input("Annual Income (₹)", min_value=1, value=500000, step=10000)
    with col2:
        home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        emp_length     = st.number_input("Years of Employment", min_value=0.0, max_value=60.0, value=3.0, step=0.5)
        cred_hist      = st.number_input("Credit History Length (years)", min_value=0, max_value=30, value=4)

    st.subheader("💰 Loan Details")
    col3, col4 = st.columns(2)
    with col3:
        loan_intent = st.selectbox(
            "Loan Purpose",
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOME", "DEBTCONSOLIDATION"]
        )
    with col4:
        loan_amnt = st.number_input("Loan Amount Requested (₹)", min_value=1000, value=200000, step=5000)

    st.subheader("📋 Credit History")
    prior_default = st.radio(
        "Any prior loan default on record?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        horizontal=True
    )

    submitted = st.form_submit_button("🚀 Submit Application", use_container_width=True)

if submitted:
    if not name.strip():
        st.error("Please enter your full name.")
    elif "@" not in email or "." not in email:
        st.error("Please enter a valid email address.")
    else:
        with st.spinner("Analysing your application..."):
            customer_data = {
                "name"                       : name,
                "email"                      : email,
                "person_age"                 : age,
                "person_income"              : income,
                "person_home_ownership"      : home_ownership,
                "person_emp_length"          : emp_length,
                "loan_intent"                : loan_intent,
                "loan_amnt"                  : loan_amnt,
                "cb_person_cred_hist_length" : cred_hist,
                "cb_person_default_on_file"  : prior_default,
            }
            result = process_loan_application(customer_data)
            save_application(customer_data, result)

            try:
                send_email(email, name, result['email'])
                email_sent = True
            except Exception as e:
                email_sent = False

        st.divider()
        st.success("✅ Application Received!")
        st.markdown(f"""
        Thank you **{name}**, your loan application has been successfully submitted.

        Our team has reviewed your application and a detailed response
        has been sent to **{email}**.

        Please check your inbox (and spam folder) for our decision letter.
        """)
        if not email_sent:
            st.warning("Note: There was an issue sending the confirmation email. Please contact support.")
