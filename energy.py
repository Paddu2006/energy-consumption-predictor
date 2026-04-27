# Energy Consumption Predictor
# By Padma Shree
# Project 23 of 25

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1 - Generate realistic energy data
print("=== ENERGY CONSUMPTION PREDICTOR ===")
np.random.seed(42)

dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="h")
n = len(dates)

# Features
hour = dates.hour
month = dates.month
day_of_week = dates.dayofweek
is_weekend = (day_of_week >= 5).astype(int)

# Temperature pattern
temperature = (
    25 + 10 * np.sin(2 * np.pi * month / 12) +
    5 * np.sin(2 * np.pi * hour / 24) +
    np.random.normal(0, 2, n)
)

# Energy consumption pattern
base_load = 2000
time_factor = (
    500 * np.sin(2 * np.pi * (hour - 6) / 24) +
    300 * np.sin(2 * np.pi * month / 12)
)
temp_factor = 50 * (temperature - 25)
weekend_factor = -200 * is_weekend
noise = np.random.normal(0, 100, n)

energy = np.clip(base_load + time_factor + temp_factor +
                 weekend_factor + noise, 500, 5000)

df = pd.DataFrame({
    "DateTime": dates,
    "Hour": hour,
    "Month": month,
    "DayOfWeek": day_of_week,
    "IsWeekend": is_weekend,
    "Temperature": temperature.round(1),
    "Energy_MW": energy.round(1)
})

print("Total records:", len(df))
print("Date range:", df["DateTime"].min(), "to", df["DateTime"].max())
print(f"Average energy: {df['Energy_MW'].mean():.1f} MW")
print(f"Peak energy: {df['Energy_MW'].max():.1f} MW")
print(f"Min energy: {df['Energy_MW'].min():.1f} MW")

# Step 2 - Train model
print("\n=== TRAINING ML MODEL ===")
X = df[["Hour", "Month", "DayOfWeek", "IsWeekend", "Temperature"]]
y = df["Energy_MW"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Accuracy (R2): {r2:.4f}")
print(f"Mean Absolute Error: {mae:.2f} MW")

# Step 3 - Analysis
print("\n=== ENERGY INSIGHTS ===")
hourly_avg = df.groupby("Hour")["Energy_MW"].mean()
monthly_avg = df.groupby("Month")["Energy_MW"].mean()
print("Peak hour:", hourly_avg.idxmax(), ":00")
print("Off-peak hour:", hourly_avg.idxmin(), ":00")
print("Peak month:", monthly_avg.idxmax())
print("Lowest month:", monthly_avg.idxmin())

# Step 4 - Charts
# Chart 1 - Hourly pattern
plt.figure(figsize=(12,6))
hourly_avg.plot(kind="line", color="blue", marker="o")
plt.title("Average Energy Consumption by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Energy (MW)")
plt.xticks(range(0,24))
plt.tight_layout()
plt.savefig(r"C:\Users\Padma shree jena\Desktop\PadduDS_Journey\04_capstone\energy_predictor\hourly_pattern.png")
plt.show()
print("Chart 1 saved!!")

# Chart 2 - Monthly pattern
plt.figure(figsize=(12,6))
monthly_avg.plot(kind="bar", color="orange")
plt.title("Average Energy Consumption by Month")
plt.xlabel("Month")
plt.ylabel("Energy (MW)")
plt.xticks(range(12), ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"], rotation=45)
plt.tight_layout()
plt.savefig(r"C:\Users\Padma shree jena\Desktop\PadduDS_Journey\04_capstone\energy_predictor\monthly_pattern.png")
plt.show()
print("Chart 2 saved!!")

# Chart 3 - Feature importance
feature_importance = pd.Series(model.feature_importances_,
                               index=X.columns)
feature_importance.sort_values().plot(kind="barh", color="green", figsize=(10,6))
plt.title("Factors That Predict Energy Consumption")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(r"C:\Users\Padma shree jena\Desktop\PadduDS_Journey\04_capstone\energy_predictor\feature_importance.png")
plt.show()
print("Chart 3 saved!!")

# Chart 4 - Weekend vs Weekday
plt.figure(figsize=(10,6))
df.groupby(["Hour","IsWeekend"])["Energy_MW"].mean().unstack().plot(
    figsize=(12,6))
plt.title("Energy Consumption — Weekday vs Weekend by Hour")
plt.xlabel("Hour")
plt.ylabel("Energy (MW)")
plt.legend(["Weekday", "Weekend"])
plt.tight_layout()
plt.savefig(r"C:\Users\Padma shree jena\Desktop\PadduDS_Journey\04_capstone\energy_predictor\weekday_vs_weekend.png")
plt.show()
print("Chart 4 saved!!")

print("\nEnergy prediction complete!!")