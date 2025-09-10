import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# Load model and dataset
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("wine_model.pkl")  # Make sure this is in same folder

@st.cache_data
def load_dataset():
    return pd.read_csv("winequality.csv")  # Only for metrics

model = load_model()
df = load_dataset()

# -----------------------------
# Streamlit Layout
# -----------------------------
st.title("🍷 Wine Quality Prediction App")

# -----------------------------
# Option Selection: Manual or Upload
# -----------------------------
option = st.radio("Choose Prediction Mode", ["Manual Input", "Upload CSV"])

if option == "Upload CSV":
    st.subheader("Upload CSV for Multiple Predictions")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file:
        try:
            input_df = pd.read_csv(uploaded_file)
            predictions = model.predict(input_df)
            input_df["Predicted Quality"] = ["Good (>=6)" if p==1 else "Bad (<6)" for p in predictions]
            
            st.success("✅ Predictions Successful!")
            st.dataframe(input_df)

            # Results Visualization
            st.subheader("Results Visualization")
            fig, ax = plt.subplots()
            sns.countplot(x=input_df["Predicted Quality"], ax=ax, palette="Set2")
            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("Count")
            ax.set_title("Predicted Wine Quality Distribution")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Error in file prediction: {e}")

elif option == "Manual Input":
    st.subheader("Enter Wine Features for Single Prediction")
    with st.form(key="input_form"):
        fixed_acidity = st.number_input("Fixed Acidity", value=7.0)
        volatile_acidity = st.number_input("Volatile Acidity", value=0.7)
        citric_acid = st.number_input("Citric Acid", value=0.0)
        residual_sugar = st.number_input("Residual Sugar", value=1.9)
        chlorides = st.number_input("Chlorides", value=0.076)
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0)
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0)
        density = st.number_input("Density", value=0.9978)
        pH = st.number_input("pH", value=3.51)
        sulphates = st.number_input("Sulphates", value=0.56)
        alcohol = st.number_input("Alcohol", value=9.4)
        submit_button = st.form_submit_button(label="Predict")

    if submit_button:
        user_features = pd.DataFrame([[ 
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
            density, pH, sulphates, alcohol
        ]], columns=df.columns.drop("quality"))

        prediction = model.predict(user_features)[0]
        st.success("✅ Prediction Successful!")
        st.info(f"Predicted Wine Quality: **{'Good (>=6)' if prediction==1 else 'Bad (<6)'}**")

# -----------------------------
# Model Performance Metrics (always visible)
# -----------------------------
st.subheader("📊 Model Performance on Full Dataset")
y = (df["quality"] >= 6).astype(int)
y_pred = model.predict(df.drop("quality", axis=1))

acc = accuracy_score(y, y_pred)
st.write(f"✅ Accuracy: **{acc:.2f}**")

cm = confusion_matrix(y, y_pred)
cm_labels = ["Bad (<6)", "Good (>=6)"]
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cm_labels, yticklabels=cm_labels, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
st.pyplot(fig)
