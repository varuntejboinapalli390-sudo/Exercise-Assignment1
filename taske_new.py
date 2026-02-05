import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Step 1: Read portfolio dataset
portfolio_df = pd.read_csv("Task3_Data_Process.csv")

risk_values = portfolio_df["x"].to_numpy().reshape(-1, 1)
return_values = portfolio_df["y"].to_numpy()

# Step 2: Visualize original data
plt.figure(figsize=(8, 6))
plt.scatter(risk_values, return_values)
plt.xlabel("Risk Factor (x)")
plt.ylabel("Portfolio Return (y)")
plt.title("Portfolio Risk vs Return")
plt.show()

# Step 3: Split data into training and testing sets
risk_train, risk_test, return_train, return_test = train_test_split(
    risk_values, return_values, test_size=0.2, random_state=1
)

# Step 4: Train Support Vector Regression model
svr_model = SVR(kernel="rbf", C=8, epsilon=0.1)
svr_model.fit(risk_train, return_train)

# Step 5: Predict on test data
predicted_returns = svr_model.predict(risk_test)

# Step 6: Evaluate model performance
mse_value = mean_squared_error(return_test, predicted_returns)
r2_value = r2_score(return_test, predicted_returns)

print("Mean Squared Error:", round(mse_value, 4))
print("R-squared Score:", round(r2_value, 4))

# Step 7: Plot regression curve
risk_range = np.linspace(risk_values.min(), risk_values.max(), 200).reshape(-1, 1)
regression_output = svr_model.predict(risk_range)

plt.figure(figsize=(8, 6))
plt.scatter(risk_values, return_values)
plt.plot(risk_range, regression_output)
plt.xlabel("Risk Factor (x)")
plt.ylabel("Portfolio Return (y)")
plt.title("SVR Based Non-Linear Regression")
plt.show()
