
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# Load the dataset
# The file 'DXY.csv' is expected to be in the same directory as this script.
try:
    df = pd.read_csv('DXY.csv')
except FileNotFoundError:
    print("Error: DXY.csv not found. Please ensure the file is in the correct directory.")
    exit()

# Ensure the necessary columns exist
required_columns = ['Open', 'High', 'Low', 'Close']
if not all(col in df.columns for col in required_columns):
    print(f"Error: Missing one or more required columns ({required_columns}) in DXY.csv.")
    exit()

# Sort by date if a 'Date' or 'Time' column exists to ensure correct time series order
# Assuming the data is already sorted by time if no specific date/time column is provided
# If there's a datetime column, you might want to sort:
# df['Date'] = pd.to_datetime(df['Date']) # Uncomment and adjust column name if date column exists
# df = df.sort_values(by='Date') # Uncomment if you sort by a date column

# Create lagged features for the previous candle's OHLC
# We shift the OHLC columns by 1 to use the previous candle's data to predict the current one.
df['Prev_Open'] = df['Open'].shift(1)
df['Prev_High'] = df['High'].shift(1)
df['Prev_Low'] = df['Low'].shift(1)
df['Prev_Close'] = df['Close'].shift(1)

# Drop rows with NaN values, which will be the first row after shifting
df.dropna(inplace=True)

# Define features (X) and target (y)
# X will be the OHLC of the previous candle
X = df[['Prev_Open', 'Prev_High', 'Prev_Low', 'Prev_Close']]

# y will be the OHLC of the current candle that we want to predict
y = df[['Open', 'High', 'Low', 'Close']]

# Split the data into training and testing sets
# We use a 80/20 split for training and testing (test_size=0.2 means 20% for testing).
# For time series data, it's often better to split chronologically to avoid data leakage.
# However, for simplicity and a general approach, train_test_split is used here.
# A more robust approach for time series would be:
# train_size = int(len(df) * 0.8)
# X_train, X_test = X[:train_size], X[train_size:]
# y_train, y_test = y[:train_size], y[train_size:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestRegressor model
# RandomForestRegressor is a robust choice for regression tasks and can handle multi-output prediction.
# n_estimators: The number of trees in the forest. More trees generally lead to better performance but take longer.
# random_state: For reproducibility.
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Train the model
print("Training the model...")
model.fit(X_train, y_train)
print("Model training complete.")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("\n--- Model Evaluation ---")

# Mean Absolute Error (MAE) for each predicted OHLC component
mae_open = mean_absolute_error(y_test['Open'], y_pred[:, 0])
mae_high = mean_absolute_error(y_test['High'], y_pred[:, 1])
mae_low = mean_absolute_error(y_test['Low'], y_pred[:, 2])
mae_close = mean_absolute_error(y_test['Close'], y_pred[:, 3])

print(f"Mean Absolute Error (Open): {mae_open:.4f}")
print(f"Mean Absolute Error (High): {mae_high:.4f}")
print(f"Mean Absolute Error (Low): {mae_low:.4f}")
print(f"Mean Absolute Error (Close): {mae_close:.4f}")

# R-squared (coefficient of determination) for overall model performance
# A higher R-squared indicates a better fit to the data (closer to 1.0).
r2_overall = r2_score(y_test, y_pred)
print(f"Overall R-squared: {r2_overall:.4f}")

# You can also print the R-squared for individual components if desired
r2_open = r2_score(y_test['Open'], y_pred[:, 0])
r2_high = r2_score(y_test['High'], y_pred[:, 1])
r2_low = r2_score(y_test['Low'], y_pred[:, 2])
r2_close = r2_score(y_test['Close'], y_pred[:, 3])

print(f"R-squared (Open): {r2_open:.4f}")
print(f"R-squared (High): {r2_high:.4f}")
print(f"R-squared (Low): {r2_low:.4f}")
print(f"R-squared (Close): {r2_close:.4f}")

# Calculate win rate for Close price direction
# Get the actual previous close from the test set features (X_test)
prev_close_for_direction_check = X_test['Prev_Close']

# Determine actual direction for the close price (1 for up, -1 for down, 0 for no change)
actual_direction = np.sign(y_test['Close'] - prev_close_for_direction_check)

# Determine predicted direction for the close price
predicted_direction = np.sign(y_pred[:, 3] - prev_close_for_direction_check)

# Calculate correct predictions where the direction is not flat (i.e., actual_direction is not 0)
# A "win" is counted if both the actual and predicted directions are the same and not zero.
correct_direction_predictions = (actual_direction == predicted_direction) & (actual_direction != 0)

# Total number of instances where the price actually moved (either up or down)
total_directional_moves = (actual_direction != 0).sum()

# Calculate the win rate
win_rate = (correct_direction_predictions.sum() / total_directional_moves) * 100 \
           if total_directional_moves > 0 else 0

print(f"Win Rate (Close Price Direction): {win_rate:.2f}%")


# Example of how to use the trained model for a new prediction
# Let's take the last candle's OHLC from the original data as an example for prediction
if not df.empty:
    last_candle_data = df[['Open', 'High', 'Low', 'Close']].iloc[-1].values
    # Create a DataFrame for the new prediction input with correct column names
    # This ensures consistency with how the model was trained (with feature names).
    new_prediction_input = pd.DataFrame(
        [last_candle_data],
        columns=['Prev_Open', 'Prev_High', 'Prev_Low', 'Prev_Close']
    )

    predicted_next_candle = model.predict(new_prediction_input)
    print("\n--- Example Prediction ---")
    print(f"Using previous candle OHLC: {last_candle_data}")
    print(f"Predicted next candle OHLC (Open, High, Low, Close): {predicted_next_candle[0].round(4)}")
else:
    print("\nNo data available for example prediction after processing.")
