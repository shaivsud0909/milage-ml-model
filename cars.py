# Repo url is "https://github.com/SharadJ19/milage-ml-model"
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# Load dataset
data = pd.read_csv("cars.csv")


x = data[["Weight_kg","Engine_cc","Wheel_Diameter_in","Power_hp","Cylinders"]]
y = data["Mileage_kmpl"]

# Split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)


y_pred = model.predict(x_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))


name = input("Enter the name of the car: ")
weight_kg = float(input("Enter the weight in kg: "))
engine_cc = float(input("Enter engine size in cc: "))
wheel_dia = float(input("Enter wheel diameter (inches): "))
power_hp = float(input("Enter power in HP: "))
cylinders = int(input("Enter number of cylinders: "))


input_data = pd.DataFrame([[weight_kg, engine_cc, wheel_dia, power_hp, cylinders]],
                          columns=["Weight_kg","Engine_cc","Wheel_Diameter_in","Power_hp","Cylinders"])


predicted_mileage = model.predict(input_data)
print(f"The predicted mileage for {name} is {predicted_mileage[0]:.2f} kmpl")
