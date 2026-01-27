import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.models.load_model(r'C:\Users\bey77\OneDrive\Desktop\Projects\FinanceProj\Stock Predictions Model Keras\Stock Predictions Model.keras')
# model = tf.keras.models.load_model(r'C:\Users\bey77\PycharmProjects\ML\Stock Predictions Model Keras\Stock Predictions Model.keras')

st.header('Stock Market Predictor')
stock = st.text_input('Enter Stock Ticker', 'MSFT')
start = '2012-01-01'
end = '2020-01-01'

# data = yf.download(stock, start, end)
# st.subheader('Stock Data')
# st.write(data)

# # slicing with the data
# data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
# data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

# 1. Download data with auto_adjust to ensure 'Close' is present
data = yf.download(stock, start=start, end=end, auto_adjust=True)

# 2. Check if data is empty
if data.empty:
    st.error(f"No data found for {stock}. Please check the ticker symbol or date range.")
    st.stop()  # Stops the Streamlit app execution here

st.subheader('Stock Data')
st.write(data.tail())

# 3. Handle potential Multi-Index columns (yfinance 0.2.x+ behavior)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# 4. Slicing with .iloc for safety
train_size = int(len(data) * 0.80)
data_train = pd.DataFrame(data['Close'].iloc[0:train_size])
data_test = pd.DataFrame(data['Close'].iloc[train_size:])

# 5. Now fit the scaler
scaler = MinMaxScaler(feature_range=(0, 1))
if not data_train.empty:
    scaler.fit(data_train)
else:
    st.error("Training set is empty. Not enough data for the selected range.")
    st.stop()

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_train)

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs Moving Average for 50 Days')
st.subheader('Price = Green; MA 50 = Red')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA 50 days vs MA 100 days')
st.subheader('Price = Green; MA 50 = Red; MA 100 = Blue')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)


st.subheader('Price vs MA 50 days vs MA 100 days vs MA 200 days')
st.subheader('Price = Green; MA 50 = Red; MA 100 = Blue, MA 200 = Yellow')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(ma_200_days, 'y')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

x = []
t = []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i - 100: i])
    t.append(data_test_scale[i, 0])
x, t = np.array(x), np.array(t)

pred = model.predict(x)
scale = 1/scaler.scale_
pred = pred * scale
t = t * scale

pred = pred.flatten()
t = t.flatten()

pred_dir = np.sign(np.diff(pred))
true_dir = np.sign(np.diff(t))

mask = true_dir != 0
if mask.sum() > 0:
    directional_accuracy = (pred_dir[mask] == true_dir[mask]).mean()
else:
    directional_accuracy = (pred_dir == true_dir).mean()

st.subheader('Directional accuracy')
st.write(f"Directional accuracy (excluding flat true moves): {directional_accuracy * 100:.2f}%")

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(pred, 'r', label='Original Price')
plt.plot(t, 'g', label='Predicted Price')
plt.xlabel('Time (in days)')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)


