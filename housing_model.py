# housing_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv("Housing.csv")

# Features and labels
X = df[["area", "bedrooms", "bathrooms"]]
y = df["price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")
