# StockProGPT2.0: A Data-driven Approach to Predicting Stock Price
About The Project
I've always had a interest in stock and options trading. It can be a risky and volatile endeavor. I wanted to find a way to make the process easier and more profitable. This project aims to automate the analysis and forecasting of stock prices using machine learning techniques and financial indicators. By leveraging historical stock data and options chain information, the project seeks to provide insights into stock behavior and forecast price movements.

Key Componenets

Data Collection: Historical stock price data is retrieved using the Yahoo Finance API (yfinance). Options chain data is obtained for a specified ticker symbol to analyze call and put options.

Data Preprocessing: Missing values are handled by dropping rows with missing data. Various financial indicators are calculated, including Simple Moving Average (SMA), Relative Strength Index (RSI), Exponential Moving Average (EMA), Moving Average Convergence Divergence (MACD), Volume Weighted Average Price (VWAP), Bollinger Bands, Stochastic Oscillator, Average True Range (ATR), On-Balance Volume (OBV), Money Flow Index (MFI), and Chaikin Money Flow (CMF).

Time Series Analysis: Augmented Dickey-Fuller (ADF) test is conducted to check for stationarity in the stock price time series data. LSTM (Long Short-Term Memory) neural network model is trained on the preprocessed data for time series forecasting.

Model Training and Evaluation: The LSTM model is trained using historical stock data with features engineered from financial indicators. The model is evaluated using mean squared error (MSE), mean absolute error (MAE), and R-squared metrics on a test set. Predictions are made for future stock prices using the trained model.

Visualization: Historical and predicted stock prices are visualized using matplotlib to provide a clear understanding of model performance and forecasted trends.

Dataset

The dataset contains historical stock data for the ticker 'SPY' from the dates chosen to look between. Each row represents daily stock metrics including:

Date

Open

High

Low

Close

Adjusted Close

Volume

Simple Moving Average (SMA)

Relative Strength Index (RSI)

Exponential Moving Average (EMA)

MACD Histogram

Volume Weighted Average Price (VWAP)

Upper Bollinger Band (Upper BB)

Lower Bollinger Band (Lower BB)

Stochastic Oscillator

Average True Range (ATR)

On-Balance Volume (OBV)

Money Flow Index (MFI)

Chaikin Money Flow (CMF)

Data Split for Training and Testing:

The data is split into features (X) and the target variable (y), where 'Close' is used as the target variable. The dataset is then split into training, validation, and testing sets using a train-test split ratio of 80-10-10. The training set comprises 80% of the data, the validation set comprises 10%, and the testing set comprises the remaining 10%. Additionally, the LSTM model uses a lookback window of 60 days (n_steps = 60) for training, meaning it considers the past 60 days of stock data to make predictions.

Therefore, the dataset is divided into:

Training data: Contains 80% of the total data, used for training the LSTM model. Validation data: Contains 10% of the total data, used for validating the model during training. Testing data: Contains 10% of the total data, used for evaluating the model's performance on unseen data after training.

Conclusion
This project demonstrates an automated pipeline for stock analysis and forecasting, incorporating both technical indicators and machine learning techniques. By utilizing historical data and options chain information, investors can gain valuable insights into stock behavior and make informed decisions regarding trading strategies.

Future Work

Adding real time data so the model is continuously learning as market data comes in.

Incorporating sentiment analysis from news articles or social media data for additional insights.

Implementing ensemble methods or alternative machine learning algorithms for comparison.

Building a web interface for easy access and visualization of analysis results.

Overall, the project offers a robust framework for automated stock analysis and forecasting, empowering investors with tools to navigate the dynamic financial markets.
