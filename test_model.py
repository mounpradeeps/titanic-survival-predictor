import joblib
import pandas as pd

# Load the final ML pipeline
model = joblib.load("titanic_final_pipeline.pkl")

# Create a test passenger
passenger = pd.DataFrame([{
    "Pclass": 3,
    "Sex": 0,
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S",
    "Title": "Mr"
}])

# Make prediction
prediction = model.predict(passenger)

# Check prediction
assert len(prediction) == 1
assert prediction[0] in [0, 1]

print("SUCCESS: Final pipeline loaded and prediction test passed!")
