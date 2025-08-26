import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


data = pd.read_csv("cars.csv")

categorical_cols = ["Make", "Fuel_Type", "Transmission", "Drivetrain"]
numerical_cols = ["Weight_kg", "Engine_cc", "Wheel_Diameter_in", "Power_hp", "Cylinders"]

#one hot
data_encoded = pd.get_dummies(data[categorical_cols], drop_first=True)  # drop first to avoid dummy variable trap


x = pd.concat([data[numerical_cols], data_encoded], axis=1) #axis 1 because horizontal concatenation
y = data["Mileage_kmpl"]

#split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)


y_pred = model.predict(x_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

#input
name = input("Enter the name of the car: ")
make = input("Enter the make of the car: ")
fuel = input("Enter fuel type (Petrol/Diesel): ")
trans = input("Enter transmission (Manual/Automatic/DCT/CVT): ")
drive = input("Enter drivetrain (FWD/RWD/4WD): ")
weight_kg = float(input("Enter the weight in kg: "))
engine_cc = float(input("Enter engine size in cc: "))
wheel_dia = float(input("Enter wheel diameter (inches): "))
power_hp = float(input("Enter power in HP: "))
cylinders = int(input("Enter number of cylinders: "))

# Prepare input
input_data = pd.DataFrame([{
    "Weight_kg": weight_kg,
    "Engine_cc": engine_cc,
    "Wheel_Diameter_in": wheel_dia,
    "Power_hp": power_hp,
    "Cylinders": cylinders,
    "Make": make,
    "Fuel_Type": fuel,
    "Transmission": trans,
    "Drivetrain": drive
}])

# Encode 
input_encoded = pd.get_dummies(input_data, columns=categorical_cols)

# Align 
input_encoded = input_encoded.reindex(columns=x.columns, fill_value=0)

# Predict
predicted_mileage = model.predict(input_encoded)
print(f"\nThe predicted mileage for {name} is {predicted_mileage[0]:.2f} kmpl")
