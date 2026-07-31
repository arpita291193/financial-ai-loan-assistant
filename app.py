"""
Loan Portfolio Explorer
------------------------
A Streamlit dashboard for exploring a loan book dataset:
loan_id, customer_id, loan_type, loan_amount, interest_rate,
tenure_months, loan_status (Active / Closed / Default).

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Loan Portfolio Explorer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_PATH = "loans.csv"


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data
def load_data(file) -> pd.DataFrame:
    if hasattr(file, "name") and file.name.lower().endswith((".xlsx", ".xlsb", ".xls")):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]

    # Rough monthly EMI-style estimate (equal principal + simple interest), just for
    # a rough "monthly payment" view -- not a precise amortization formula.
    df["is_default"] = (df["loan_status"] == "Default").astype(int)
    df["annual_interest_amount"] = df["loan_amount"] * df["interest_rate"] / 100
    df["tenure_years"] = df["tenure_months"] / 12
    return df


st.sidebar.title("🏦 Data Source")
uploaded = st.sidebar.file_uploader("Upload a loans file (CSV / XLSX)", type=["csv", "xlsx", "xls"])

if uploaded is not None:
    df = load_data(uploaded)
    st.sidebar.success(f"Loaded {uploaded.name}")
else:
    df = load_data(DEFAULT_PATH)
    st.sidebar.info("Using default dataset: loans.xlsb")

# --------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------
st.sidebar.title("🔎 Filters")

loan_types = sorted(df["loan_type"].unique())
selected_types = st.sidebar.multiselect("Loan type", loan_types, default=loan_types)

statuses = sorted(df["loan_status"].unique())
selected_statuses = st.sidebar.multiselect("Loan status", statuses, default=statuses)

amt_min, amt_max = float(df["loan_amount"].min()), float(df["loan_amount"].max())
amt_range = st.sidebar.slider("Loan amount range", amt_min, amt_max, (amt_min, amt_max))

rate_min, rate_max = float(df["interest_rate"].min()), float(df["interest_rate"].max())
rate_range = st.sidebar.slider("Interest rate range (%)", rate_min, rate_max, (rate_min, rate_max))

tenure_min, tenure_max = int(df["tenure_months"].min()), int(df["tenure_months"].max())
tenure_range = st.sidebar.slider("Tenure range (months)", tenure_min, tenure_max, (tenure_min, tenure_max))

filtered = df[
    df["loan_type"].isin(selected_types)
    & df["loan_status"].isin(selected_statuses)
    & df["loan_amount"].between(*amt_range)
    & df["interest_rate"].between(*rate_range)
    & df["tenure_months"].between(*tenure_range)
]

st.sidebar.markdown(f"**{len(filtered):,} / {len(df):,} loans match filters**")

# --------------------------------------------------------------------------
# Header + KPIs
# --------------------------------------------------------------------------
st.title("🏦 Loan Portfolio Explorer")
st.caption("Explore loan amounts, interest rates, tenure, and default risk across the portfolio.")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Loans", f"{len(filtered):,}")
k2.metric("Total Disbursed", f"${filtered['loan_amount'].sum():,.0f}" if len(filtered) else "-")
k3.metric("Avg Interest Rate", f"{filtered['interest_rate'].mean():.2f}%" if len(filtered) else "-")
k4.metric("Avg Tenure", f"{filtered['tenure_months'].mean():.0f} mo" if len(filtered) else "-")
default_rate = filtered["is_default"].mean() * 100 if len(filtered) else 0
k5.metric("Default Rate", f"{default_rate:.1f}%")

st.divider()

tab_overview, tab_explore, tab_type, tab_model, tab_data = st.tabs(
    ["📊 Overview", "🔍 Explore", "🏷️ Loan Types", "🤖 Default Risk Model", "📋 Raw Data"]
)

# ---- Overview --------------------------------------------------------------
with tab_overview:
    if len(filtered) == 0:
        st.warning("No rows match the current filters.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Loan Status Breakdown")
            status_counts = filtered["loan_status"].value_counts().reset_index()
            status_counts.columns = ["loan_status", "count"]
            fig = px.pie(
                status_counts, names="loan_status", values="count", hole=0.4,
                color="loan_status",
                color_discrete_map={"Active": "#3498db", "Closed": "#2ecc71", "Default": "#e74c3c"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Default Rate by Loan Type")
            type_default = (
                filtered.groupby("loan_type")["is_default"].mean().mul(100).sort_values(ascending=False).reset_index()
            )
            fig = px.bar(
                type_default, x="loan_type", y="is_default",
                labels={"is_default": "Default Rate (%)", "loan_type": "Loan Type"},
                color="is_default", color_continuous_scale="Reds",
            )
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Loan Amount Distribution")
            fig = px.histogram(
                filtered, x="loan_amount", color="loan_status", nbins=40, barmode="overlay", opacity=0.7,
                color_discrete_map={"Active": "#3498db", "Closed": "#2ecc71", "Default": "#e74c3c"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.subheader("Interest Rate Distribution")
            fig = px.histogram(
                filtered, x="interest_rate", color="loan_status", nbins=30, barmode="overlay", opacity=0.7,
                color_discrete_map={"Active": "#3498db", "Closed": "#2ecc71", "Default": "#e74c3c"},
            )
            st.plotly_chart(fig, use_container_width=True)

# ---- Explore -----------------------------------------------------------------
with tab_explore:
    numeric_cols = ["loan_amount", "interest_rate", "tenure_months", "annual_interest_amount"]

    if len(filtered) == 0:
        st.warning("No rows match the current filters.")
    else:
        st.subheader("Custom Scatter Plot")
        c1, c2, c3 = st.columns(3)
        x_axis = c1.selectbox("X-axis", numeric_cols, index=0)
        y_axis = c2.selectbox("Y-axis", numeric_cols, index=1)
        color_by = c3.selectbox("Color by", ["loan_status", "loan_type"], index=0)

        fig = px.scatter(
            filtered, x=x_axis, y=y_axis, color=color_by,
            hover_data=["loan_id", "customer_id"],
            color_discrete_map={"Active": "#3498db", "Closed": "#2ecc71", "Default": "#e74c3c"} if color_by == "loan_status" else None,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Correlation Heatmap")
        corr = filtered[numeric_cols + ["is_default"]].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Tenure vs Interest Rate (by status)")
        fig = px.box(filtered, x="tenure_months", y="interest_rate", color="loan_status", points=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Largest Loans")
        n = st.slider("Number of rows", 5, 30, 10)
        top = filtered.nlargest(n, "loan_amount")
        st.dataframe(
            top[["loan_id", "customer_id", "loan_type", "loan_amount", "interest_rate", "tenure_months", "loan_status"]],
            use_container_width=True, hide_index=True,
        )

# ---- Loan Types --------------------------------------------------------------
with tab_type:
    if len(filtered) == 0:
        st.warning("No rows match the current filters.")
    else:
        st.subheader("Loan Type Breakdown")
        col1, col2 = st.columns(2)
        with col1:
            type_counts = filtered["loan_type"].value_counts().reset_index()
            type_counts.columns = ["loan_type", "count"]
            fig = px.pie(type_counts, names="loan_type", values="count", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            type_stats = (
                filtered.groupby("loan_type")
                .agg(
                    avg_amount=("loan_amount", "mean"),
                    avg_rate=("interest_rate", "mean"),
                    avg_tenure=("tenure_months", "mean"),
                    default_rate=("is_default", "mean"),
                    count=("loan_id", "count"),
                )
                .sort_values("avg_amount", ascending=False)
                .reset_index()
            )
            type_stats["default_rate"] = (type_stats["default_rate"] * 100).round(1)
            st.dataframe(type_stats, use_container_width=True, hide_index=True)

        st.subheader("Loan Amount by Type")
        fig = px.box(filtered, x="loan_type", y="loan_amount", color="loan_type")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Interest Rate by Type")
        fig = px.box(filtered, x="loan_type", y="interest_rate", color="loan_type")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ---- Default Risk Model ---------------------------------------------------------
with tab_model:
    st.subheader("Predicting Loan Default (Random Forest)")
    st.write(
        "Trains a classifier on the **filtered** data to predict whether a loan "
        "ends in `Default`, using loan amount, interest rate, tenure, and loan type."
    )

    if filtered["is_default"].nunique() < 2 or len(filtered) < 30:
        st.warning("Not enough data / class variety in the current filter selection to train a model. Broaden your filters.")
    else:
        model_df = filtered.copy()
        le = LabelEncoder()
        model_df["type_enc"] = le.fit_transform(model_df["loan_type"])

        features = ["loan_amount", "interest_rate", "tenure_months", "type_enc"]
        X = model_df[features]
        y = model_df["is_default"]

        test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
        n_estimators = st.slider("Number of trees", 50, 300, 150, 25)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight="balanced")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        m1, m2 = st.columns(2)
        m1.metric("Accuracy", f"{acc:.2%}")
        m2.metric("ROC AUC", f"{auc:.3f}")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Feature Importance**")
            imp = pd.DataFrame({"feature": features, "importance": clf.feature_importances_}).sort_values(
                "importance", ascending=False
            )
            fig = px.bar(imp, x="importance", y="feature", orientation="h")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.write("**Confusion Matrix**")
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(
                cm, text_auto=True,
                x=["Pred: Not Default", "Pred: Default"],
                y=["Actual: Not Default", "Actual: Default"],
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.write("**ROC Curve**")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig = px.line(x=fpr, y=tpr, labels={"x": "False Positive Rate", "y": "True Positive Rate"})
        fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.write("**Try a prediction**")
        p1, p2, p3, p4 = st.columns(4)
        in_amount = p1.number_input("Loan amount", value=float(model_df["loan_amount"].median()))
        in_rate = p2.number_input("Interest rate (%)", value=float(model_df["interest_rate"].median()))
        in_tenure = p3.number_input("Tenure (months)", value=int(model_df["tenure_months"].median()))
        in_type = p4.selectbox("Loan type", options=le.classes_)

        if st.button("Predict default risk"):
            type_code = le.transform([in_type])[0]
            row = [[in_amount, in_rate, in_tenure, type_code]]
            pred = clf.predict(row)[0]
            proba = clf.predict_proba(row)[0][1]
            if pred == 1:
                st.error(f"Predicted: **Default risk** (probability: {proba:.1%})")
            else:
                st.success(f"Predicted: **Low default risk** (probability of default: {proba:.1%})")

# ---- Raw Data --------------------------------------------------------------------
with tab_data:
    st.subheader("Filtered Dataset")
    display_cols = ["loan_id", "customer_id", "loan_type", "loan_amount", "interest_rate", "tenure_months", "loan_status"]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download filtered data as CSV",
        data=filtered[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="loans_export.csv",
        mime="text/csv",
    )

st.divider()
st.caption("Built with Streamlit • Data: loans dataset")
