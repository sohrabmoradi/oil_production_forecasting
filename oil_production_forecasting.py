import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Define the specific file paths for the three wells
file_paths = [
    'C:/Users/Sohrab/Desktop/data projects/dataset_1_well_1.xlsx',
    'C:/Users/Sohrab/Desktop/data projects/dataset_1_well_2.xlsx',
    'C:/Users/Sohrab/Desktop/data projects/dataset_1_well_3.xlsx'
]

# Initialize lists to store results for each well
all_results = []

for file_path in file_paths:
    # Load and preprocess data
    excel_file = pd.ExcelFile(file_path)
    sheet_name = 'Calculated Data'
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    df = df[['Time (Days)', 'Gas Volume', 'Oil Volume', 'Gas Production']]

    df['Time (Days)'] = pd.to_numeric(df['Time (Days)'], errors='coerce')
    df['Gas Volume'] = pd.to_numeric(df['Gas Volume'], errors='coerce')
    df['Oil Volume'] = pd.to_numeric(df['Oil Volume'], errors='coerce')
    df['Gas Production'] = pd.to_numeric(df['Gas Production'], errors='coerce')

    df = df.dropna()

    if len(df) == 0:
        raise ValueError(
            "No data available for modeling after preprocessing. Please check your dataset.")

    features = ['Time (Days)', 'Gas Volume', 'Gas Production']
    X = df[features]
    y = df['Oil Volume']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    print(f'Well {file_path.split("_")[-1][:-5]}')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')
    print('----------------------------------')

    results_df = X_test.copy()
    results_df['Actual Oil Volume'] = y_test
    results_df['Predicted Oil Volume'] = y_pred
    results_df = results_df.sort_values(by='Time (Days)')

    last_time = df['Time (Days)'].max()
    future_days = np.arange(last_time + 1, last_time + 301).reshape(-1, 1)

    gas_volume_trend = np.polyfit(df['Time (Days)'], df['Gas Volume'], 1)
    gas_production_trend = np.polyfit(
        df['Time (Days)'], df['Gas Production'], 1)

    future_gas_volume = gas_volume_trend[0] * future_days + gas_volume_trend[1]
    future_gas_production = gas_production_trend[0] * \
        future_days + gas_production_trend[1]

    future_features = np.hstack(
        [future_days, future_gas_volume, future_gas_production])

    future_predictions = model.predict(future_features)

    future_df = pd.DataFrame(future_features, columns=features)
    future_df['Oil Volume'] = future_predictions

    df['Type'] = 'Actual'
    results_df['Type'] = 'Predicted'
    future_df['Type'] = 'Forecasted'

    final_df = pd.concat([df[['Time (Days)', 'Oil Volume', 'Type']],
                          results_df[['Time (Days)', 'Actual Oil Volume', 'Type']].rename(
                              columns={'Actual Oil Volume': 'Oil Volume'}),
                          future_df[['Time (Days)', 'Oil Volume', 'Type']]], ignore_index=True)

    all_results.append((file_path.split("_")[-1][:-5], final_df))

# Plot the results for each well separately
plt.figure(figsize=(14, 8))

for well_name, final_df in all_results:
    plt.figure()
    for label, df in final_df.groupby('Type'):
        if label == 'Forecasted':
            plt.plot(df['Time (Days)'], df['Oil Volume'],
                     label=label, marker='x', linewidth=0.1)
        else:
            plt.plot(df['Time (Days)'], df['Oil Volume'],
                     label=label, marker='o' if label == 'Actual' else 'x')

    plt.xlabel('Time (days)')
    plt.ylabel('Oil Production (stb)')
    plt.title(f'Actual, Predicted, and Forecasted Oil Production Over Time for Well {
              well_name}')
    plt.legend()
    plt.grid(True)
    plt.show()
