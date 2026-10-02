import joblib
import pandas as pd

model = joblib.load("titanic_model.pkl")

passenger = pd.DataFrame([{
    "Pclass": 3,
    "Sex": 0,
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked_C": 0,
    "Embarked_Q": 0,
    "Embarked_S": 1
}])

prediction = model.predict(passenger)

assert len(prediction) == 1
assert prediction[0] in [0, 1]

print("SUCCESS: Model loaded and prediction test passed!")
