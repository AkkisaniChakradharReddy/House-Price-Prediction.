import joblib
import pandas as pd

# Load the model
model = joblib.load("model.pkl")

# Sample input
sample = pd.DataFrame({
    "area": [1500],
    "bedrooms": [3],
    "age": [5]
})

prediction = float(model.predict(sample)[0])

print("Test prediction: Rs.", format(prediction, ",.2f"))

# Validate prediction
assert prediction > 0, "Predicted price must be positive"

print("Model test passed.")