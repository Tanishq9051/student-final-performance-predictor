import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ----------------------------------------
# LOAD DATA
# ----------------------------------------

data = pd.read_csv("student-mat.csv", sep=";")

# Features
X = data[["G1", "G2"]]

# Target
y = data["G3"]


# ----------------------------------------
# TRAIN / TEST SPLIT
# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ----------------------------------------
# TRAIN MODEL
# ----------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)


# ----------------------------------------
# EVALUATE MODEL
# ----------------------------------------

test_predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)

print("MODEL PERFORMANCE")
print("-----------------")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")


# ----------------------------------------
# PREDICTION FUNCTION
# ----------------------------------------

def predict_final_grade(g1, g2):
    new_student = pd.DataFrame({
        "G1": [g1],
        "G2": [g2]
    })

    prediction = model.predict(new_student)
    return prediction[0]


# ----------------------------------------
# TEST WITH A NEW STUDENT
# ----------------------------------------

g1 = float(input("\nEnter first-period grade (G1, 0-20): "))
g2 = float(input("Enter second-period grade (G2, 0-20): "))

predicted_g3 = predict_final_grade(g1, g2)

print(f"\nPredicted final grade: {predicted_g3:.2f} / 20")