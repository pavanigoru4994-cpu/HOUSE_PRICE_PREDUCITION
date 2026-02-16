import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="House Rent Prediction", layout="wide")

st.title("🏠 Hyderabad House Rent Prediction App")
st.markdown("Upload your dataset and predict house rent using Linear Regression")

# Upload file
uploaded_file = st.file_uploader("Upload Hyderabad_House_Data.csv", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(data.head())

    # -------------------------
    # DATA PREPROCESSING
    # -------------------------

    # Clean Washrooms
    data['Washrooms'] = pd.to_numeric(data['Washrooms'], errors='coerce')
    data['Washrooms'] = data['Washrooms'].fillna(data['Washrooms'].median())

    # Clean Tennants
    data['Tennants'] = pd.to_numeric(data['Tennants'], errors='coerce')
    data['Tennants'] = data['Tennants'].fillna(data['Tennants'].median())

    # Clean Area
    data['Area'] = data['Area'].astype(str).str.extract('(\d+)')
    data['Area'] = pd.to_numeric(data['Area'], errors='coerce')
    data['Area'] = data['Area'].fillna(data['Area'].median())

    # Clean Price
    data['Price'] = (
        data['Price']
        .astype(str)
        .str.replace(r'[^\d]', '', regex=True)
    )
    data['Price'] = pd.to_numeric(data['Price'], errors='coerce')

    # Drop remaining null values
    data = data.dropna()

    # -------------------------
    # FEATURE & TARGET
    # -------------------------

    X = data.drop("Price", axis=1)
    y = data["Price"]

    # Encode categorical
    X = pd.get_dummies(X, drop_first=True)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.subheader("📊 Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("MSE", f"{mse:.2f}")
    col3.metric("RMSE", f"{rmse:.2f}")
    col4.metric("R2 Score", f"{r2:.2f}")

    # -------------------------
    # NEW HOUSE PREDICTION
    # -------------------------

    st.subheader("🔮 Predict New House Rent")

    area = st.number_input("Area (sq ft)", min_value=100, value=1400)
    bedrooms = st.number_input("Bedrooms", min_value=1, value=2)
    washrooms = st.number_input("Washrooms", min_value=1, value=2)

    if st.button("Predict Price"):

        new_house = pd.DataFrame({
            "Area": [area],
            "Bedrooms": [bedrooms],
            "Washrooms": [washrooms]
        })

        new_house = pd.get_dummies(new_house)
        new_house = new_house.reindex(columns=X.columns, fill_value=0)

        new_scaled = scaler.transform(new_house)
        predicted_price = model.predict(new_scaled)

        st.success(f"💰 Predicted House Rent: ₹ {predicted_price[0]:,.2f}")

    # -------------------------
    # VISUALIZATION
    # -------------------------

    st.subheader("📈 Actual vs Predicted Prices")

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    st.pyplot(fig)

else:
    st.info("Please upload the dataset to continue.")
