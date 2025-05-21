# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix

# Load datasets
@st.cache_data
def load_data():
    df_t = pd.read_csv("../Toddler Autism dataset July 2018.csv")
    df_a = pd.read_csv("../autism_screening.csv")

    df_t = df_t.dropna()
    df_t.columns = df_t.columns.str.strip()
    df_t.rename(columns={'Class/ASD Traits': 'ASD'}, inplace=True)

    df_a = df_a.dropna()
    df_a.columns = df_a.columns.str.strip()
    df_a.rename(columns={'Class/ASD': 'ASD'}, inplace=True)

    return df_t, df_a

df_t, df_a = load_data()

# Sidebar for dataset selection
st.sidebar.title("Dataset Selector")
dataset_choice = st.sidebar.radio("Select Dataset", ("Toddler", "Adult"))
df = df_t if dataset_choice == "Toddler" else df_a

tab1, tab2, tab3, tab4 = st.tabs(["📄 Dataset Overview", "📊 EDA", "🧠 Model Training", "🔍 Prediction"])


with tab1:
    st.header(f"{dataset_choice} Dataset Overview")
    st.markdown(f"- **Total Rows:** `{df.shape[0]}`")
    st.markdown(f"- **Total Columns:** `{df.shape[1]}`")

    # Column names listed in a scrollable code block
    st.markdown("**🧾 Column Names:**")
    st.code("\n".join(df.columns), language='text')
    if st.checkbox("Show first 5 rows"):
        st.dataframe(df.head())
    if st.checkbox("Show descriptive stats"):
        st.dataframe(df.describe(include='all'))
    if df.isnull().sum().sum() == 0:
        st.success("No missing values found.")
    else:
        st.warning("Missing values detected:")
        st.dataframe(df.isnull().sum()[df.isnull().sum() > 0])



def eda_toddler(df_t):
    st.subheader("Toddler Dataset: Ethnicity vs ASD")
    
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df_t, x="Ethnicity", hue="ASD", palette="Set2")
    plt.xticks(rotation=45)
    plt.xlabel("Ethnicity")
    plt.ylabel("Count")
    plt.title("Ethnicity Distribution of Toddlers by ASD Status")
    plt.legend(title="ASD")
    st.pyplot(plt.gcf())

    # Gender distribution
    st.subheader("Gender Distribution")
    st.bar_chart(df_t["Sex"].value_counts())


def eda_adult(df_a):
    st.subheader("Adult Dataset: Ethnicity vs ASD")

    plt.figure(figsize=(12, 6))
    sns.countplot(data=df_a, x="ethnicity", hue="ASD", palette="Set2")
    plt.xticks(rotation=45)
    plt.xlabel("Ethnicity")
    plt.ylabel("Count")
    plt.title("Ethnicity Distribution of Adults by ASD Status")
    plt.legend(title="ASD")
    st.pyplot(plt.gcf())

    # Gender distribution
    st.subheader("Gender Distribution")
    st.bar_chart(df_a["gender"].value_counts())
    
with tab2:
    st.header(f"📊 EDA {dataset_choice} Dataset")
    
    if dataset_choice == "Toddler":
        eda_toddler(df_t)
    else:
        eda_adult(df_a)
# === Tab 3: Model Training ===

with tab3:
    st.header("Models")
    dataset_type = dataset_choice.lower()
    model_dir = f"model/{dataset_type}"
    if dataset_choice == "Toddler":
        st.image (r"C:\Users\ROG\Desktop\Autism_Detection\knn_t.png")
        st.image(r"C:\Users\ROG\Desktop\Autism_Detection\svm_t.png")
        st.image(r"C:\Users\ROG\Desktop\Autism_Detection\DT_t.png")
        st.image(r"C:\Users\ROG\Desktop\Autism_Detection\tabnet_t.png")
        shap_img_path = "C:/Users/ROG/Desktop/Autism_Detection/shap.png"
        st.image(shap_img_path)
    else:
        st.image(r"C:\Users\ROG\Desktop\Autism_Detection\DT_a.png")
        st.image(r"C:\Users\ROG\Desktop\Autism_Detection\knn_a.png")
        st.image(r"C:\Users\ROG\Desktop\Autism_Detection\tabnet_a.png")


# === Tab 4: Prediction ===
with tab4:
    st.header("Make a Prediction")

    prefix = f"{dataset_choice.lower()}_"

    # === Step 1: Input Form (Adult vs Toddler) ===
    if dataset_choice == "Adult":
        input_dict = {
            'A9_Score': st.radio("A9 Score", [0, 1], horizontal=True, key=f"{prefix}A9"),
            'A6_Score': st.radio("A6 Score", [0, 1], horizontal=True, key=f"{prefix}A6"),
            'A5_Score': st.radio("A5 Score", [0, 1], horizontal=True, key=f"{prefix}A5"),
            'A4_Score': st.radio("A4 Score", [0, 1], horizontal=True, key=f"{prefix}A4"),
            'A3_Score': st.radio("A3 Score", [0, 1], horizontal=True, key=f"{prefix}A3"),
            'ethnicity': st.selectbox("Ethnicity", [
                'White-European', 'Latino', 'Black', 'Asian', 'Middle Eastern ',
                'Pasifika', 'South Asian', 'Hispanic', 'Turkish', 'others'
            ], key=f"{prefix}ethnicity"),
            'gender': st.selectbox("Gender", ["m", "f"], key=f"{prefix}gender"),
            'age': st.number_input("Age", min_value=1, max_value=100, value=25, key=f"{prefix}age")
        }

    else:  # Toddler
        input_dict = {
            'A7': st.radio("A7", [0, 1], horizontal=True, key=f"{prefix}A7"),
            'A9': st.radio("A9", [0, 1], horizontal=True, key=f"{prefix}A9"),
            'A6': st.radio("A6", [0, 1], horizontal=True, key=f"{prefix}A6"),
            'A5': st.radio("A5", [0, 1], horizontal=True, key=f"{prefix}A5"),
            'A1': st.radio("A1", [0, 1], horizontal=True, key=f"{prefix}A1"),
            'A2': st.radio("A2", [0, 1], horizontal=True, key=f"{prefix}A2"),
            'Sex': st.selectbox("Sex", ["m", "f", "others"], key=f"{prefix}sex"),
            'Ethnicity': st.selectbox("Ethnicity", [
                'middle eastern', 'White European', 'Hispanic', 'black', 'asian', 'south asian',
                'Native Indian', 'Others', 'Latino', 'mixed', 'Pacifica'
            ], key=f"{prefix}ethnicity"),
            'Age_Mons': st.number_input("Age (in months)", 6, 144, 36, key=f"{prefix}agemons")
        }

    input_df = pd.DataFrame([input_dict])

    # === Step 2: Prediction Logic ===
    try:
        # File paths
        model_path = f"model/{dataset_choice.lower()}/svm.pkl"
        scaler_path = f"model/{dataset_choice.lower()}/scaler.pkl"
        columns_path = f"model/{dataset_choice.lower()}/Xtrain_columns.pkl"

        # Load trained model and preprocessing assets
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        X_train_columns = joblib.load(columns_path)

        # Align input features
        encoded_input = pd.get_dummies(input_df)
        encoded_input = encoded_input.reindex(columns=X_train_columns, fill_value=0)

        # Scale input
        input_scaled = scaler.transform(encoded_input)

        # Predict
        prediction = model.predict(input_scaled)[0]
        st.success(f"Prediction: {'ASD Positive' if prediction == 1 else 'ASD Negative'}")

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
