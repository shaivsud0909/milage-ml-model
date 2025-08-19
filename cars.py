import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("cars.csv")

x = data[["Engine_Litre","Power_BHP","Torque_Nm","Weight_kg"]]
y = data["Mileage_kmpl"]

# Split train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:",np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))


plt.figure(figsize=(10, 6))

# Scatter: Actual vs Predicted
plt.scatter(y_test, y_pred, color='blue', label="Predicted vs Actual")
# Labels and title
plt.xlabel("Actual Mileage (kmpl)")
plt.ylabel("Predicted Mileage (kmpl)")
plt.legend()

plt.show()




# User input
name = input("Enter the name of the car: ")
engine_litre = float(input("Enter the engine litre: "))
power_bhp = float(input("Enter the power in BHP: "))
torque_nm = float(input("Enter the torque in Nm: "))
weight_kg = float(input("Enter the weight in kg: "))

input_data = pd.DataFrame([[engine_litre, power_bhp, torque_nm, weight_kg]],
                          columns=["Engine_Litre", "Power_BHP", "Torque_Nm","Weight_kg"])

predicted_mileage = model.predict(input_data)

print(f"The predicted mileage for {name} is {predicted_mileage[0]} kmpl")
