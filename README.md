Stock Market Predictor 📈
A machine learning-powered web application that predicts stock prices using LSTM (Long Short-Term Memory) networks. Built with Python, TensorFlow, and Streamlit, this tool fetches real-time historical data from Yahoo Finance to visualize trends and predict future price movements.

🚀 Features
Real-time Data Fetching: Uses yfinance to pull historical stock data (Default: MSFT).

Moving Average Visualizations: Automatically calculates and plots 50-day, 100-day, and 200-day Moving Averages (MA) to identify market trends.

Deep Learning Predictions: Utilizes a pre-trained Keras LSTM model to predict price movements based on the last 100 days of data.

Interactive Dashboard: A user-friendly interface built with Streamlit, allowing users to enter any stock ticker.



🛠️ Tech Stack
Frontend: Streamlit

Data Analysis: Pandas, NumPy

Visualization: Matplotlib

Machine Learning: TensorFlow/Keras (LSTM Model)

Data Source: Yahoo Finance API (yfinance)



📊 Methodology
Data Slicing: The data is split into training (80%) and testing (20%) sets.

Preprocessing: Data is scaled between 0 and 1 using MinMaxScaler for optimal neural network performance.

Prediction Logic: The model accepts the previous 100 days of closing prices as input to predict the next day's price.

Post-processing: Predicted values are scaled back to their original price format for visualization against actual data.

⚠️ Disclaimer
This tool is for educational purposes only. Stock market investments carry risks, and machine learning models are not 100% accurate. Do not make financial decisions based solely on these predictions.
