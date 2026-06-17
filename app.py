import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(
    page_title="Fetal Health Classification",
    page_icon="🩺",
    layout="wide",
)

# ------------------ CUSTOM STYLING ------------------ #
page_bg = """
<style>
/* Backgrounds */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #fce4ec, #e3f2fd);
}
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #bbdefb, #f8bbd0);
}

/* Text */
h1 {
    color: #2c3e50;
    text-align: center;
    font-size: 40px !important;
    font-weight: 800;
}
h2, h3, h4 {
    color: #2c3e50;
    font-weight: 600;
}
p, label, .stMarkdown {
    font-size: 18px !important;
    color: #34495e;
}

/* Number inputs */
input[type="number"] {
    border-radius: 10px;
    border: 1px solid #b0bec5;
    padding: 10px;
    font-size: 16px;
}

/* Buttons */
div.stButton > button {
    background-color: #64b5f6;
    color: white;
    border-radius: 12px;
    font-weight: bold;
    font-size: 18px;
    padding: 8px 20px;
    transition: all 0.3s;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
}
div.stButton > button:hover {
    background-color: #42a5f5;
    transform: scale(1.05);
}

/* Tables */
table {
    font-size: 17px !important;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------ #
xgb_model = joblib.load("models/xgb_model.pkl")

# ------------------ FEATURES ------------------ #
boruta_features = [
 'accelerations',
 'mean_value_of_long_term_variability',
 'uterine_contractions',
 'abnormal_short_term_variability',
 'histogram_mean',
 'histogram_variance',
 'percentage_of_time_with_abnormal_long_term_variability',
 'histogram_mode',
 'prolongued_decelerations',
 'mean_value_of_short_term_variability',
 'histogram_min'
]

label_map = {0: "Normal", 1: "Suspect", 2: "Pathological"}
color_map = {"Normal": "#2ecc71", "Suspect": "#f1c40f", "Pathological": "#e74c3c"}

# ------------------ SIDEBAR ------------------ #
st.sidebar.title("📘 About the Project")
st.sidebar.markdown("""
**Fetal Health Classification**  
Predicts the condition of a fetus as:
- 🟢 Normal  
- 🟡 Suspect  
- 🔴 Pathological  

**Algorithm:** XGBoost  
**Feature Selection:** BorutaShap (11 important features)  
**Accuracy:** ~94–95%
""")
st.sidebar.info("Developed by Nandhitha | Final Year Project")

# ------------------ MAIN HEADER ------------------ #
st.title("🩺 Fetal Health Classification App")
st.image("https://cdn-icons-png.flaticon.com/512/2940/2940756.png", width=130)
st.markdown("<h3 style='text-align:center;'>Predict Fetal Health using Machine Learning (Boruta + XGBoost)</h3>", unsafe_allow_html=True)

# ------------------ TABS ------------------ #
tab1, tab2 = st.tabs(["🧍‍♀️ Single Entry Prediction", "📂 Batch Prediction"])

# ============================================================
# 🧍‍♀️ SINGLE ENTRY MODE
# ============================================================
with tab1:
    st.markdown("<h3 style='color:#1565c0;'>🔹 Enter Fetal Monitoring Values:</h3>", unsafe_allow_html=True)

    with st.form("single_form"):
        # Two-column layout for inputs
        col1, col2 = st.columns(2)
        user_data = []

        for i, feature in enumerate(boruta_features):
            if i < len(boruta_features) / 2:
                val = col1.number_input(f"{feature}", value=0.0, step=0.1)
            else:
                val = col2.number_input(f"{feature}", value=0.0, step=0.1)
            user_data.append(val)

        single_submit = st.form_submit_button("🔍 Predict Fetal Health")

    if single_submit:
        user_df = pd.DataFrame([user_data], columns=boruta_features)
        prediction = xgb_model.predict(user_df)[0]
        result = label_map[prediction]
        color = color_map[result]

        st.markdown(
            f"<h2 style='text-align:center; color:{color}; font-weight:bold;'>"
            f"✅ Predicted Fetal Health: {result}</h2>",
            unsafe_allow_html=True
        )

        if hasattr(xgb_model, "predict_proba"):
            probs = xgb_model.predict_proba(user_df)[0]
            prob_df = pd.DataFrame({
                "Condition": list(label_map.values()),
                "Probability": np.round(probs, 3)
            })
            st.markdown("<h4>🔍 Prediction Confidence:</h4>", unsafe_allow_html=True)
            st.dataframe(prob_df)

        st.balloons()

# ============================================================
# 📂 BATCH MODE
# ============================================================
with tab2:
    st.markdown("<h3 style='color:#1565c0;'>📁 Upload CSV for Batch Prediction</h3>", unsafe_allow_html=True)
    st.markdown("""
    Upload a CSV file containing **exactly these 11 columns**:
    `accelerations`, `mean_value_of_long_term_variability`,  
    `uterine_contractions`, `abnormal_short_term_variability`,  
    `histogram_mean`, `histogram_variance`,  
    `percentage_of_time_with_abnormal_long_term_variability`,  
    `histogram_mode`, `prolongued_decelerations`,  
    `mean_value_of_short_term_variability`, `histogram_min`
    """)

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.write("### 🔍 Preview of Uploaded Data:")
            st.dataframe(batch_data.head())

            if all(col in batch_data.columns for col in boruta_features):
                preds = xgb_model.predict(batch_data[boruta_features])
                results = [label_map[p] for p in preds]
                batch_data["Predicted_Fetal_Health"] = results

                # Color the output column
                def color_result(val):
                    color = color_map.get(val, "black")
                    return f'color: {color}; font-weight:bold;'

                st.write("### ✅ Prediction Results:")
                st.dataframe(batch_data.style.applymap(color_result, subset=["Predicted_Fetal_Health"]))

                # Add summary counts
                st.markdown("### 📊 Summary of Predictions:")
                summary = batch_data["Predicted_Fetal_Health"].value_counts().rename_axis("Condition").reset_index(name="Count")
                summary["Color"] = summary["Condition"].map(color_map)
                for _, row in summary.iterrows():
                    st.markdown(f"- <span style='color:{row.Color}; font-weight:bold;'>{row.Condition}</span>: {row.Count}", unsafe_allow_html=True)

                # Download predictions
                csv = batch_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Predictions as CSV",
                    data=csv,
                    file_name='fetal_health_predictions.csv',
                    mime='text/csv'
                )
            else:
                st.error("❌ Uploaded CSV does not contain all required columns.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# ------------------ FOOTER ------------------ #
st.markdown("---")
st.caption("Developed by Nandhitha 💖 Fetal Health Prediction using XGBoost + Boruta")
