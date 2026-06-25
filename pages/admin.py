import streamlit as st
import sqlite3
import pandas as pd

st.set_page_config(page_title="CreditLens — Admin", page_icon="🔍", layout="wide")

# ─────────────────────────────────────────────
# PASSWORD GATE
# ─────────────────────────────────────────────
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

if not st.session_state.admin_authenticated:
    st.title("🔐 CreditLens Admin Login")
    st.divider()
    password = st.text_input("Enter Admin Password", type="password")
    if st.button("Login", use_container_width=True):
        if password == st.secrets["ADMIN_PASSWORD"]:
            st.session_state.admin_authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# ─────────────────────────────────────────────
# ADMIN DASHBOARD
# ─────────────────────────────────────────────
st.title("🔍 CreditLens — Admin Panel")
st.caption("Internal view — all application decisions, SHAP explanations, and generated emails.")

if st.button("🔓 Logout"):
    st.session_state.admin_authenticated = False
    st.rerun()

st.divider()

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
try:
    conn = sqlite3.connect('database.db')
    df   = pd.read_sql_query("SELECT * FROM applications ORDER BY submitted_at DESC", conn)
    conn.close()
except Exception as e:
    st.error(f"Database error: {e}")
    st.stop()

if df.empty:
    st.info("No applications submitted yet.")
    st.stop()

# ─────────────────────────────────────────────
# SUMMARY METRICS
# ─────────────────────────────────────────────
total    = len(df)
approved = len(df[df['decision'] == 'APPROVED'])
rejected = len(df[df['decision'] == 'REJECTED'])
tailored = len(df[df['is_tailored'] == 1])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Applications", total)
col2.metric("Approved",           approved, delta=f"{approved/total*100:.0f}%")
col3.metric("Rejected",           rejected, delta=f"-{rejected/total*100:.0f}%")
col4.metric("Tailored Loans",     tailored)

st.divider()

# ─────────────────────────────────────────────
# APPLICATION TABLE
# ─────────────────────────────────────────────
st.subheader("📋 All Applications")

display_df = df[[
    'submitted_at', 'customer_name', 'customer_email',
    'loan_intent', 'loan_amnt', 'decision',
    'approved_amnt', 'interest_rate', 'is_tailored'
]].copy()

display_df['approved_amnt']  = display_df['approved_amnt'].apply(lambda x: f"₹{float(x):,.0f}")
display_df['interest_rate']  = display_df['interest_rate'].apply(lambda x: f"{float(x):.2f}%")
display_df['loan_amnt']      = display_df['loan_amnt'].apply(lambda x: f"₹{int(x):,}")
display_df['is_tailored']    = display_df['is_tailored'].apply(lambda x: "Yes" if x else "No")

display_df.columns = [
    'Submitted At', 'Name', 'Email',
    'Purpose', 'Requested', 'Decision',
    'Approved Amount', 'Rate', 'Tailored'
]

def highlight_decision(val):
    if val == 'APPROVED': return 'background-color: #1a4e1a; color: white'
    if val == 'REJECTED': return 'background-color: #4e1a1a; color: white'
    return ''

st.dataframe(
    display_df.style.applymap(highlight_decision, subset=['Decision']),
    use_container_width=True,
    hide_index=True
)

st.divider()

# ─────────────────────────────────────────────
# INDIVIDUAL APPLICATION DETAIL
# ─────────────────────────────────────────────
st.subheader("🔍 Application Detail View")

app_ids    = df['id'].tolist()
app_labels = [
    f"#{row['id']} — {row['customer_name']} — {row['decision']} ({row['submitted_at']})"
    for _, row in df.iterrows()
]
selected = st.selectbox(
    "Select an application to inspect",
    options=app_ids,
    format_func=lambda x: app_labels[app_ids.index(x)]
)

row = df[df['id'] == selected].iloc[0]

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Applicant Info**")
    st.write(f"- Name: {row['customer_name']}")
    st.write(f"- Email: {row['customer_email']}")
    st.write(f"- Age: {row['age']}")
    st.write(f"- Income: ₹{int(row['income']):,}")
    st.write(f"- Home Ownership: {row['home_ownership']}")
    st.write(f"- Employment Length: {float(row['emp_length'])} yrs")
    st.write(f"- Credit History: {row['cred_hist']} yrs")
    st.write(f"- Prior Default: {'Yes' if row['prior_default'] else 'No'}")

with col_b:
    st.markdown("**Loan & Decision**")
    st.write(f"- Purpose: {row['loan_intent']}")
    st.write(f"- Requested: ₹{int(row['loan_amnt']):,}")
    st.write(f"- Approved: ₹{float(row['approved_amnt']):,.0f}")
    st.write(f"- Interest Rate: {float(row['interest_rate']):.2f}%")
    st.write(f"- Tailored Down: {'Yes' if row['is_tailored'] else 'No'}")
    if row['decision'] == 'APPROVED':
        st.success("✅ Decision: APPROVED")
    else:
        st.error("❌ Decision: REJECTED")

st.divider()

with st.expander("🔍 SHAP Risk Factor Breakdown", expanded=True):
    st.markdown(row['shap_explanation'])

with st.expander("📧 Generated Customer Email", expanded=True):
    st.markdown(row['generated_email'])
