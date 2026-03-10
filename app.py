import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Care Load Forecast", layout="wide")

st.title("Predictive Forecasting of Care Load & Placement Demand")

# ---------------------------------------------------
# 1. CREATE DATASET IF NOT EXISTS
# ---------------------------------------------------
if not os.path.exists("uac_dataset.csv"):

    data = {
        "Date": [
            "2025-01-01","2025-01-02","2025-01-03","2025-01-04",
            "2025-01-05","2025-01-06","2025-01-07","2025-01-08"
        ],
        "Children in HHS Care":[5000,4980,5020,5030,5040,5050,5070,5080],
        "Children discharged from HHS Care":[350,360,330,345,350,340,355,360]
    }

    pd.DataFrame(data).to_csv("uac_dataset.csv", index=False)

# ---------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("uac_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df)

# ---------------------------------------------------
# 3. VISUALIZATION
# ---------------------------------------------------
st.subheader("Children in HHS Care Trend")

fig, ax = plt.subplots()
ax.plot(df["Date"], df["Children in HHS Care"])
ax.set_xlabel("Date")
ax.set_ylabel("Children in HHS Care")
plt.xticks(rotation=45)

st.pyplot(fig)

# ---------------------------------------------------
# 4. MACHINE LEARNING FORECAST
# ---------------------------------------------------
st.subheader("AI Forecast Model")

df = df.sort_values("Date")
df["Day"] = np.arange(len(df))

X = df[["Day"]]
y = df["Children in HHS Care"]

model = LinearRegression()
model.fit(X,y)

future_days = st.slider("Forecast Days",7,60,30)

future_X = np.arange(len(df),len(df)+future_days).reshape(-1,1)

predictions = model.predict(future_X)

future_dates = pd.date_range(df["Date"].iloc[-1], periods=future_days)

forecast = pd.DataFrame({
    "Date":future_dates,
    "Predicted Children in HHS Care":predictions.astype(int)
})

st.subheader("Forecast Results")
st.dataframe(forecast)

# ---------------------------------------------------
# 5. FORECAST GRAPH
# ---------------------------------------------------
st.subheader("Forecast Visualization")

fig2, ax2 = plt.subplots()

ax2.plot(df["Date"],y,label="Actual")
ax2.plot(future_dates,predictions,label="Predicted")

ax2.legend()

st.pyplot(fig2)

# ---------------------------------------------------
# 6. KPIs
# ---------------------------------------------------
st.subheader("Key Indicators")

col1,col2,col3 = st.columns(3)

col1.metric("Current Children in Care", int(df["Children in HHS Care"].iloc[-1]))
col2.metric("Average Discharge", int(df["Children discharged from HHS Care"].mean()))
col3.metric("Forecast Peak", int(max(predictions)))

st.success("System running successfully")