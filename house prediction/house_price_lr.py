import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Load training data
train = pd.read_csv(r"C:\Users\house prediction\HousePrices\train.csv")

# Create TotalBath feature
train['TotalBath'] = train['FullBath'] + 0.5 * train['HalfBath']

# Features and target
X = train[['GrLivArea', 'BedroomAbvGr', 'TotalBath']]
y = train['SalePrice']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_val)

# Evaluation
print("MSE:", mean_squared_error(y_val, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_val, y_pred)))
print("R² Score:", r2_score(y_val, y_pred))

# Plot Actual vs Predicted
plt.scatter(y_val, y_pred, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Actual vs Predicted Sale Prices")
plt.show()
