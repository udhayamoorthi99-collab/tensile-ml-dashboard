

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tensile Strength ML Dashboard",
    layout="wide"
)

# ============================================================
# COLUMN NAMES (MATCH YOUR TRAINING DATA)
# ============================================================

ANGLE_COL = "Angle"
AREA_COL = "Cross-Section Area (mm²)"
TYPE_COL = "Type"
TARGET_COL = "Maximum Force ( N )"

FEATURE_COLS = [ANGLE_COL, AREA_COL, TYPE_COL]

TYPE_MAPPING = {
    "Planar": 0,
    "Non-Planar": 1
}

# ============================================================
# LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("rf.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("tensile_ml_dataset.csv")


try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Validate dataset
required_columns = FEATURE_COLS + [TARGET_COL]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"Missing columns: {missing_columns}")
    st.stop()

# ============================================================
# PREPARE MODEL DATA
# ============================================================

X = df[FEATURE_COLS].copy()
X[TYPE_COL] = X[TYPE_COL].map(TYPE_MAPPING)

X[ANGLE_COL] = pd.to_numeric(
    X[ANGLE_COL], errors="coerce"
)

X[AREA_COL] = pd.to_numeric(
    X[AREA_COL], errors="coerce"
)

y = pd.to_numeric(
    df[TARGET_COL], errors="coerce"
)

# Ensure input data is valid
if X.isna().any().any() or y.isna().any():
    st.error(
        "The dataset contains missing, invalid, or "
        "unrecognized values in the model columns."
    )
    st.stop()

# Recreate the original train/test split
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Predictions using the saved model
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# ============================================================
# TITLE
# ============================================================

st.title("Tensile Test — Machine Learning Dashboard")

st.markdown("""
Random Forest regression dashboard for predicting
**Maximum Tensile Force (N)** based on:

- Orientation Angle
- Cross-Sectional Area
- Specimen Type (Planar / Non-Planar)
""")

st.divider()

# ============================================================
# 1. MODEL PERFORMANCE
# ============================================================

st.header("1. Model Performance")

train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(
    mean_squared_error(y_train, y_train_pred)
)
train_r2 = r2_score(y_train, y_train_pred)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(
    mean_squared_error(y_test, y_test_pred)
)
test_r2 = r2_score(y_test, y_test_pred)

st.subheader("Training Performance")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Train MAE", f"{train_mae:.2f} N")

with c2:
    st.metric("Train RMSE", f"{train_rmse:.2f} N")

with c3:
    st.metric("Train R²", f"{train_r2:.4f}")

st.subheader("Testing Performance")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Test MAE", f"{test_mae:.2f} N")

with c2:
    st.metric("Test RMSE", f"{test_rmse:.2f} N")

with c3:
    st.metric("Test R²", f"{test_r2:.4f}")

st.caption(
    "Evaluation uses the original 80/20 random "
    "train-test split with random_state=42. "
    "The saved model must correspond to this dataset."
)

# Actual vs Predicted
st.subheader("Actual vs Predicted Force")

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    y_test,
    y_test_pred,
    alpha=0.7
)

minimum = min(y_test.min(), y_test_pred.min())
maximum = max(y_test.max(), y_test_pred.max())

ax.plot(
    [minimum, maximum],
    [minimum, maximum],
    "r--",
    label="Perfect Prediction"
)

ax.set_xlabel("Actual Maximum Force (N)")
ax.set_ylabel("Predicted Maximum Force (N)")
ax.set_title("Random Forest — Test Predictions")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)
plt.close(fig)

st.divider()

# ============================================================
# 2. EXPERIMENTAL MECHANICAL BEHAVIOUR
# ============================================================

st.header("2. Experimental Mechanical Behaviour")

col1, col2 = st.columns(2)

# ANGLE VS MAXIMUM FORCE
with col1:

    st.subheader("Orientation vs Maximum Force")

    fig, ax = plt.subplots(figsize=(7, 5))

    for specimen_type in ["Planar", "Non-Planar"]:

        subset = df[df[TYPE_COL] == specimen_type]

        if subset.empty:
            continue

        stats = (
            subset.groupby(ANGLE_COL)[TARGET_COL]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values(ANGLE_COL)
        )

        ax.errorbar(
            stats[ANGLE_COL],
            stats["mean"],
            yerr=stats["std"].fillna(0),
            marker="o",
            capsize=3,
            label=specimen_type
        )

    ax.set_xlabel("Orientation Angle (°)")
    ax.set_ylabel("Maximum Force (N)")
    ax.grid(alpha=0.3)
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

# AREA VS MAXIMUM FORCE
with col2:

    st.subheader("Area vs Maximum Force")

    fig, ax = plt.subplots(figsize=(7, 5))

    for specimen_type in ["Planar", "Non-Planar"]:

        subset = df[df[TYPE_COL] == specimen_type]

        if subset.empty:
            continue

        ax.scatter(
            subset[AREA_COL],
            subset[TARGET_COL],
            alpha=0.7,
            label=specimen_type
        )

    ax.set_xlabel("Cross-Sectional Area (mm²)")
    ax.set_ylabel("Maximum Force (N)")
    ax.grid(alpha=0.3)
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

st.divider()

# ============================================================
# 3. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

st.header("3. Random Forest Feature Importance")

importance_df = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "Importance": model.feature_importances_
}).sort_values(
    "Importance",
    ascending=False
)

fig, ax = plt.subplots(figsize=(8, 4))

ax.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)

ax.set_ylabel("Importance")
ax.set_title("Feature Importance")
ax.tick_params(axis="x", rotation=15)

st.pyplot(fig)
plt.close(fig)

st.dataframe(
    importance_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Feature importance indicates the relative contribution "
    "of the input variables to Random Forest splits. "
    "It does not establish causality."
)

st.divider()

# ============================================================
# 4. INTERACTIVE FORCE PREDICTION
# ============================================================

st.header("4. Predict Maximum Tensile Force")

st.markdown("""
Enter specimen parameters to estimate the
maximum tensile force using your trained Random Forest.
""")

col1, col2, col3 = st.columns(3)

with col1:
    angle = st.slider(
        "Orientation Angle (°)",
        min_value=0,
        max_value=90,
        value=45,
        step=5
    )

with col2:
    area = st.number_input(
        "Cross-Section Area (mm²)",
        min_value=0.01,
        value=50.0,
        step=0.1
    )

with col3:
    specimen_type = st.selectbox(
        "Specimen Type",
        ["Planar", "Non-Planar"]
    )

type_encoded = TYPE_MAPPING[specimen_type]

# Exact names and feature order used in training
input_data = pd.DataFrame({
    ANGLE_COL: [angle],
    AREA_COL: [area],
    TYPE_COL: [type_encoded]
})[FEATURE_COLS]

st.subheader("Input Parameters")
st.dataframe(input_data, hide_index=True)

if st.button(
    "Predict Maximum Force",
    type="primary"
):

    prediction = float(model.predict(input_data)[0])

    st.success("Prediction completed!")

    st.metric(
        "Predicted Maximum Force",
        f"{prediction:.2f} N"
    )

    st.caption(
        "This is a model estimate, not an experimental "
        "measurement. Predictions outside the training "
        "data range may be unreliable."
    )

st.divider()

# ============================================================
# 5. DATASET EXPLORER
# ============================================================

st.header("5. Experimental Dataset")

selected_type = st.selectbox(
    "Filter Specimen Type",
    ["All", "Planar", "Non-Planar"]
)

if selected_type == "All":
    filtered_df = df.copy()
else:
    filtered_df = df[
        df[TYPE_COL] == selected_type
    ].copy()

st.subheader("Dataset Summary")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Total Specimens",
        len(filtered_df)
    )

with c2:
    st.metric(
        "Maximum Force",
        f"{filtered_df[TARGET_COL].max():.2f} N"
        if not filtered_df.empty else "N/A"
    )

with c3:
    st.metric(
        "Average Force",
        f"{filtered_df[TARGET_COL].mean():.2f} N"
        if not filtered_df.empty else "N/A"
    )

with c4:
    st.metric(
        "Average Area",
        f"{filtered_df[AREA_COL].mean():.2f} mm²"
        if not filtered_df.empty else "N/A"
    )

st.subheader("Filtered Dataset")

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    f"Displayed specimens: {len(filtered_df)}"
)

# Download filtered data
csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered CSV",
    data=csv_data,
    file_name="filtered_tensile_data.csv",
    mime="text/csv"
)
