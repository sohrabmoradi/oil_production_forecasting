import numpy as np
import matplotlib.pyplot as plt
from pandas import read_excel
import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import sys

# Set default encoding to utf-8
sys.stdout.reconfigure(encoding='utf-8')


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# Fix random seed for reproducibility
tf.random.set_seed(7)

# Load the dataset
dataframe = read_excel('dataset_1_well_1.xlsx',
                       sheet_name='Calculated Data', usecols=[2])
dataframe = dataframe.iloc[1:]
dataset = dataframe.values
dataset = dataset.astype('float32')

# Handle NaN values
print(np.isnan(dataset).sum())
dataset = dataset[~np.isnan(dataset).any(axis=1)]
dataset = np.nan_to_num(dataset, nan=np.nanmean(dataset))
print(f"NaNs in dataset after handling: {np.isnan(dataset).sum()}")

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# Reshape into X=t and Y=t+1
look_back = 20
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Create and fit the LSTM network
model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back)))
model.add(Dense(1, activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print(f'Train Score: {trainScore:.2f} RMSE')
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print(f'Test Score: {testScore:.2f} RMSE')

# Shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# Shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2) +
                1:len(dataset)-1, :] = testPredict

# Plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset),
         label='Actual Oil Volume', color='blue')
plt.plot(trainPredictPlot, label='Train Predictions', color='green')
plt.plot(testPredictPlot, label='Test Predictions', color='orange')

# Forecasting the next 90 days
forecast = []
last_data = dataset[-look_back:]  # Start with the last available data point
for i in range(90):
    last_data_reshaped = np.reshape(last_data, (1, 1, look_back))
    next_pred = model.predict(last_data_reshaped)
    next_pred = np.maximum(next_pred, 0)  # Ensure forecast is non-negative
    next_pred = scaler.inverse_transform(next_pred)
    forecast.append(next_pred[0, 0])
    # Update last_data with the new prediction for the next loop iteration
    last_data = np.append(last_data[1:], scaler.transform(next_pred))

# Prepare the forecast plot
forecastPlot = np.empty_like(dataset)
forecastPlot[:, :] = np.nan
forecastPlot = np.append(dataset, scaler.transform(
    np.array(forecast).reshape(-1, 1)), axis=0)

# Plot the forecast
plt.plot(range(len(dataset), len(dataset) + 90), forecast,
         label='Forecasted Oil Volume', color='red')

# Add legends and labels
plt.legend()
plt.xlabel('Time (Days)')
plt.ylabel('Oil Volume (stb)')
plt.title('Oil Volume Prediction and Forecasting well7')
plt.show()
