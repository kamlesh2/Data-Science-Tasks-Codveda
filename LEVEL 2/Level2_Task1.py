import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("forbes_gold_india.csv")

print("Dataset shape:", df.shape)
print(df.head())
print(df.describe())

X = df[['Year']]
y = df['Gold_Price_INR']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"\n{name} Evaluation:")
    print("RMSE:", round(rmse, 2))
    print("R² Score:", round(r2, 4))

future_years = pd.DataFrame({'Year': list(range(2026, 2031))})
best_model = models["Random Forest"]
future_predictions = best_model.predict(future_years)

print("\nPredicted Gold Prices in India (2026–2030):")
for year, price in zip(future_years['Year'], future_predictions):
    print(f"{year}: Rs.{int(price):,}")

plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, label="Actual", color='black')
for name, y_pred in predictions.items():
    plt.plot(X_test, y_pred, label=name)
plt.xlabel("Year")
plt.ylabel("Gold Price (INR)")
plt.title("Gold Price Prediction in India (Model Comparison)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(future_years, future_predictions, label="Future Prediction (Random Forest)", color='red', linestyle='--')
plt.xlabel("Year")
plt.ylabel("Gold Price (INR)")
plt.title("Predicted Gold Prices in India (2026–2030)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
