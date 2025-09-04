import math
from datetime import date, timedelta
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
st.set_page_config(page_title="Loan Calculator", page_icon="💸", layout="wide")
st.title("💸 Loan Calculator")
st.caption("Enter borrower + loan details, then explore the schedule, charts, and downloads.")

def inr(x: float) -> str:
    try:
        return f"₹{x:,.0f}"
    except Exception:
        return f"₹{x}"

@st.cache_data(show_spinner=False)
def compute_schedule(
    principal: float,
    annual_rate_pct: float,
    years: int,
    start_date: date,
    compounding: str = "Monthly",
    flat_interest: bool = False,
    extra_per_period: float = 0.0,
    fees_flat: float = 0.0,
    fees_pct: float = 0.0,
    insurance_per_period: float = 0.0,
) -> Tuple[pd.DataFrame, dict]:
    """Return amortization schedule dataframe and summary dict."""
    periods_per_year = {"Monthly": 12, "Quarterly": 4, "Yearly": 1}[compounding]
    n = max(1, int(years * periods_per_year))

  
    principal_with_fees = principal + fees_flat + principal * (fees_pct / 100.0)

    r = (annual_rate_pct / 100.0) / periods_per_year

    if flat_interest:
        total_interest_flat = principal_with_fees * (annual_rate_pct / 100.0) * years
        base_payment = (principal_with_fees + total_interest_flat) / n
    else:
        if r == 0:
            base_payment = principal_with_fees / n
        else:
            base_payment = principal_with_fees * r * (1 + r) ** n / ((1 + r) ** n - 1)

    sched = []
    bal = principal_with_fees
    cumulative = 0.0
    d = start_date

    for k in range(1, n + 1):
        if flat_interest:
            interest = principal_with_fees * r
            principal_comp = base_payment - interest
        else:
            interest = bal * r
            principal_comp = base_payment - interest

        principal_comp = min(principal_comp, bal)
        extra = min(extra_per_period, max(0.0, bal - principal_comp))

        payment_this_period = principal_comp + interest + extra + insurance_per_period
        bal = max(0.0, bal - principal_comp - extra)
        cumulative += payment_this_period

        sched.append({
            "Period": k,
            "Date": d,
            "Payment": round(base_payment, 2),
            "Principal": round(principal_comp, 2),
            "Interest": round(interest, 2),
            "Extra": round(extra, 2),
            "Insurance": round(insurance_per_period, 2),
            "Balance": round(bal, 2),
            "Cumulative_Paid": round(cumulative, 2),
        })

        if periods_per_year == 12:
            d += timedelta(days=30)
        elif periods_per_year == 4:
            d += timedelta(days=91)
        else:
            d += timedelta(days=365)

        if bal <= 0:
            break

    df = pd.DataFrame(sched)

    if df.empty:
        summary = {k: 0 for k in [
            "EMI/Payment", "Total Interest", "Total Paid", "Tenure (periods)", "Last Payment Date",
            "Principal (incl. fees)", "Total Extra", "Total Insurance"
        ]}
        return df, summary

    summary = {
        "EMI/Payment": round(df.loc[0, "Payment"], 2),
        "Total Interest": round(float(df["Interest"].sum()), 2),
        "Total Extra": round(float(df["Extra"].sum()), 2),
        "Total Insurance": round(float(df["Insurance"].sum()), 2),
        "Total Paid": round(float(df["Payment"].sum() + df["Extra"].sum() + df["Insurance"].sum()), 2),
        "Tenure (periods)": int(df.shape[0]),
        "Last Payment Date": df["Date"].iloc[-1],
        "Principal (incl. fees)": round(principal_with_fees, 2),
    }
    return df, summary


with st.sidebar:
    st.header("Borrower Details")
    name = st.text_input("Full Name", placeholder="e.g., Shrishti Gauhan")
    age = st.number_input("Age", min_value=18, max_value=100, step=1, value=27)
    start_date = st.date_input("Start Date", value=date.today())

    st.markdown("---")
    st.header("Loan Inputs")
    loan_type = st.selectbox("Loan Type", ["Home", "Car", "Personal", "Education", "Other"], index=0)

    purchase_price = st.number_input("Purchase Price (₹)", min_value=0.0, step=50_000.0, value=2_500_000.0)
    deposit = st.slider("Deposit / Down Payment (₹)", min_value=0, max_value=int(purchase_price), value=500_000, step=25_000)

    requested_loan = st.number_input("Requested Loan Amount (₹)", min_value=0.0, step=50_000.0, value=max(0.0, purchase_price - deposit))
    principal = max(0.0, requested_loan)

    annual_rate = st.slider("Annual Interest Rate (%)", 0.0, 36.0, 9.0, 0.1)
    years = st.slider("Duration (Years)", 1, 40, 20)
    compounding = st.selectbox("Compounding", ["Monthly", "Quarterly", "Yearly"], index=0)

    st.markdown("---")
    st.header("Options")
    use_flat = st.toggle("Use flat interest method (vs reducing EMI)", value=False)

    add_extra = st.toggle("Add extra payment each period", value=False)
    extra = st.number_input("Extra per period (₹)", min_value=0.0, step=1_000.0, value=0.0, disabled=not add_extra)

    add_ins = st.checkbox("Include insurance (per period)")
    insurance = st.number_input("Insurance per period (₹)", min_value=0.0, step=500.0, value=0.0, disabled=not add_ins)

    add_fees = st.checkbox("Add processing fees")
    fees_flat = st.number_input("Flat fee (₹, added to principal)", min_value=0.0, step=500.0, value=0.0, disabled=not add_fees)
    fees_pct = st.number_input("Percent fee (% of principal)", min_value=0.0, step=0.1, value=0.0, disabled=not add_fees)

    show_schedule = st.checkbox("Show amortization table", value=True)


schedule_df, summary = compute_schedule(
    principal=principal,
    annual_rate_pct=annual_rate,
    years=years,
    start_date=start_date,
    compounding=compounding,
    flat_interest=use_flat,
    extra_per_period=extra if add_extra else 0.0,
    fees_flat=fees_flat if add_fees else 0.0,
    fees_pct=fees_pct if add_fees else 0.0,
    insurance_per_period=insurance if add_ins else 0.0,
)


st.subheader("Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("EMI / Periodic Payment", inr(summary.get("EMI/Payment", 0)))
col2.metric("Total Interest", inr(summary.get("Total Interest", 0)))
col3.metric("Total Paid", inr(summary.get("Total Paid", 0)))
col4.metric("Tenure (periods)", summary.get("Tenure (periods)", 0))

st.info(
    f"Borrower: **{name or '—'}**  •  Age: **{age}**  •  Type: **{loan_type}**  •  Method: **{'Flat' if use_flat else 'Reducing (EMI)'}**  •  Compounding: **{compounding}**"
)
if not schedule_df.empty:
    tab1, tab2, tab3 = st.tabs(["Balance", "Breakdown", "Cumulative"])

    with tab1:
        st.markdown("###  Remaining Balance Over Time")
        fig1 = px.line(schedule_df, x="Period", y="Balance", hover_data=["Date"], title="Balance vs Period")
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        st.markdown("###  Payment Components per Period")
        melt = schedule_df.melt(id_vars=["Period", "Date"], value_vars=["Principal", "Interest", "Extra", "Insurance"],
                                var_name="Component", value_name="Amount")
        fig2 = px.bar(melt, x="Period", y="Amount", color="Component", hover_data=["Date"], title="Payment Breakdown")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("###  Cumulative Principal vs Interest")
        cum = schedule_df[["Period", "Principal", "Interest"]].copy()
        cum["Cum Principal"] = cum["Principal"].cumsum()
        cum["Cum Interest"] = cum["Interest"].cumsum()
        fig3 = px.area(cum, x="Period", y=["Cum Principal", "Cum Interest"], title="Cumulative Totals")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("###  Total Cost Breakdown")
    totals_df = pd.DataFrame({
        "Label": ["Principal (incl. fees)", "Interest", "Insurance", "Extra"],
        "Amount": [summary.get("Principal (incl. fees)", 0), summary.get("Total Interest", 0), summary.get("Total Insurance", 0), summary.get("Total Extra", 0)],
    })
    fig4 = px.pie(totals_df, names="Label", values="Amount")
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.warning("No schedule yet — adjust inputs to generate results.")

left, right = st.columns([2, 1])
with left:
    st.markdown("###  Amortization Schedule")
    if show_schedule and not schedule_df.empty:
        st.dataframe(schedule_df, use_container_width=True, height=420)
        st.download_button(
            label="Download schedule CSV",
            data=schedule_df.to_csv(index=False).encode("utf-8"),
            file_name="amortization_schedule.csv",
            mime="text/csv",
        )
    elif show_schedule:
        st.info("Enable inputs to view the schedule table.")

with right:
    st.markdown("### Summary Table")
    summary_table = pd.DataFrame({"Metric": list(summary.keys()), "Value": list(summary.values())})
    st.dataframe(summary_table, use_container_width=True, height=420)
    st.download_button(
        label="Download summary CSV",
        data=summary_table.to_csv(index=False).encode("utf-8"),
        file_name="loan_summary.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("Tip: toggle flat interest, add extra payments, insurance and fees to see how costs change.")
