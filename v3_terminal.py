import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

data = pd.read_csv("student-mat.csv", sep=";")


# --------------------------------------------------
# FEATURES AND TARGET
# --------------------------------------------------

X = data[["G1", "G2"]]
y = data["G3"]


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)


# --------------------------------------------------
# EVALUATE MODEL
# --------------------------------------------------

test_predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_final_grade(g1, g2):
    new_student = pd.DataFrame({
        "G1": [g1],
        "G2": [g2]
    })

    prediction = model.predict(new_student)[0]

    # G3 is on a 0–20 scale.
    prediction = max(0, min(20, prediction))

    return prediction




# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

print("=" * 50)
print("       STUDENT FINAL PERFORMANCE PREDICTOR")
print("=" * 50)

print("\nMODEL INFORMATION")
print(f"MAE: {mae:.2f} grade points")
print(f"R²:  {r2:.2f}")

print("\nEnter the student's previous exam grades.")

while True:
    try:
        g1 = float(input("First-period grade (G1, 0-20): "))

        if 0 <= g1 <= 20:
            break

        print("Please enter a value between 0 and 20.")

    except ValueError:
        print("Please enter a valid number.")


while True:
    try:
        g2 = float(input("Second-period grade (G2, 0-20): "))

        if 0 <= g2 <= 20:
            break

        print("Please enter a value between 0 and 20.")

    except ValueError:
        print("Please enter a valid number.")


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

predicted_g3 = predict_final_grade(g1, g2)

percentage = (predicted_g3 / 20) * 100

if predicted_g3 >= 10:
    result = "PASS"
else:
    result = "FAIL"


# --------------------------------------------------
# DISPLAY RESULT
# --------------------------------------------------

print("\n" + "=" * 50)
print("               PREDICTION")
print("=" * 50)

print(f"Predicted final grade : {predicted_g3:.2f} / 20")
print(f"Predicted percentage  : {percentage:.2f}%")
print(f"Predicted result      : {result}")
print(f"Typical prediction error: ±{mae:.2f} grade points")

print("=" * 50)