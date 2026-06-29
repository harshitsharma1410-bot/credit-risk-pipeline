import streamlit as st
import sqlite3
import pandas as pd

st.set_page_config(page_title="CreditLens — Admin", page_icon="🔍", layout="wide")

# ── Password Gate ─────────────────────────────────────────────────────────────
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.title("🔐 CreditLens Admin Login")
    st.divider()
    pw = st.text_input("Enter Admin Password", type="password")
    if st.button("Login", use_container_width=True):
        if pw == st.secrets["ADMIN_PASSWORD"]:
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# ── Dashboard ─────────────────────────────────────────────────────────────────
st.title("🔍 CreditLens — Admin Panel")
st.caption("Internal view — all application decisions, SHAP explanations, and generated emails.")

if st.button("🔓 Logout"):
    st.session_state.auth = False
    st.rerun()

st.divider()

# ── Load Data ─────────────────────────────────────────────────────────────────
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

# ── Metrics ───────────────────────────────────────────────────────────────────
total    = len(df)
approved = len(df[df['decision'] == 'APPROVED'])
rejected = len(df[df['decision'] == 'REJECTED'])
tailored = len(df[df['is_tailored'] == 1])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Applications", total)
c2.metric("Approved", approved, delta=f"{approved/total*100:.0f}%")
c3.metric("Rejected", rejected, delta=f"-{rejected/total*100:.0f}%")
c4.metric("Tailored Loans", tailored)

st.divider()

# ── Table ─────────────────────────────────────────────────────────────────────
st.subheader("📋 All Applications")

disp = df[['submitted_at','customer_name','customer_email','loan_intent',
           'loan_amnt','decision','approved_amnt','base_rate','interest_rate','is_tailored']].copy()

disp['approved_amnt'] = disp['approved_amnt'].apply(lambda x: f"Rs.{float(x):,.0f}")
disp['base_rate'] = disp['base_rate'].apply(lambda x: f"{float(x):.2f}%")
disp['interest_rate'] = disp['interest_rate'].apply(lambda x: f"{float(x):.2f}%")
disp['loan_amnt']     = disp['loan_amnt'].apply(lambda x: f"Rs.{int(x):,}")
disp['is_tailored']   = disp['is_tailored'].apply(lambda x: "Yes" if x else "No")

disp.columns = ['Submitted At','Name','Email','Purpose',
                'Requested','Decision','Approved Amount','Raw Imputer Rate','Final Rate','Tailored']

def highlight(val):
    if val == 'APPROVED': return 'background-color:#1a4e1a;color:white'
    if val == 'REJECTED': return 'background-color:#4e1a1a;color:white'
    return ''

st.dataframe(disp.style.applymap(highlight, subset=['Decision']),
             use_container_width=True, hide_index=True)

st.divider()

# ── Detail View ───────────────────────────────────────────────────────────────
st.subheader("🔍 Application Detail View")

ids    = df['id'].tolist()
labels = [f"#{r['id']} — {r['customer_name']} — {r['decision']} ({r['submitted_at']})"
          for _, r in df.iterrows()]
sel    = st.selectbox("Select an application to inspect", options=ids,
                      format_func=lambda x: labels[ids.index(x)])

row = df[df['id'] == sel].iloc[0]

ca, cb = st.columns(2)
with ca:
    st.markdown("**Applicant Info**")
    st.write(f"- Name: {row['customer_name']}")
    st.write(f"- Email: {row['customer_email']}")
    st.write(f"- Age: {row['age']}")
    st.write(f"- Income: Rs.{int(row['income']):,}")
    st.write(f"- Home Ownership: {row['home_ownership']}")
    st.write(f"- Employment Length: {float(row['emp_length'])} yrs")
    st.write(f"- Credit History: {row['cred_hist']} yrs")
    st.write(f"- Prior Default: {'Yes' if row['prior_default'] else 'No'}")

with cb:
    st.markdown("**Loan & Decision**")
    st.write(f"- Purpose: {row['loan_intent']}")
    st.write(f"- Requested: Rs.{int(row['loan_amnt']):,}")
    st.write(f"- Approved: Rs.{float(row['approved_amnt']):,.0f}")
    st.write(f"- Interest Rate: {float(row['interest_rate']):.2f}%")
    st.write(f"- Tailored Down: {'Yes' if row['is_tailored'] else 'No'}")
    if row['decision'] == 'APPROVED':
        st.success("Decision: APPROVED")
    else:
        st.error("Decision: REJECTED")

st.divider()

with st.expander("🔍 SHAP Risk Factor Breakdown", expanded=True):
    st.markdown(row['shap_explanation'])

with st.expander("📧 Generated Customer Email", expanded=True):
    st.markdown(row['generated_email'])
