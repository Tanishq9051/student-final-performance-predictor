import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

data = pd.read_csv("student-mat.csv", sep=";")

# Target
y = data["G3"]


# --------------------------------------------------
# MODEL 1: G1 + G2
# --------------------------------------------------

X1 = data[["G1", "G2"]]

X1_train, X1_test, y_train, y_test = train_test_split(
    X1,
    y,
    test_size=0.2,
    random_state=42
)

model1 = LinearRegression()
model1.fit(X1_train, y_train)

prediction1 = model1.predict(X1_test)

mae1 = mean_absolute_error(y_test, prediction1)
r2_1 = r2_score(y_test, prediction1)


# --------------------------------------------------
# MODEL 2: G1 + G2 + STUDYTIME
# --------------------------------------------------

X2 = data[["G1", "G2", "studytime"]]

X2_train, X2_test, y_train, y_test = train_test_split(
    X2,
    y,
    test_size=0.2,
    random_state=42
)

model2 = LinearRegression()
model2.fit(X2_train, y_train)

prediction2 = model2.predict(X2_test)

mae2 = mean_absolute_error(y_test, prediction2)
r2_2 = r2_score(y_test, prediction2)


# --------------------------------------------------
# MODEL 3: G1 + G2 + STUDYTIME + FAILURES + ABSENCES
# --------------------------------------------------

X3 = data[
    [
        "G1",
        "G2",
        "studytime",
        "failures",
        "absences"
    ]
]

X3_train, X3_test, y_train, y_test = train_test_split(
    X3,
    y,
    test_size=0.2,
    random_state=42
)

model3 = LinearRegression()
model3.fit(X3_train, y_train)

prediction3 = model3.predict(X3_test)

mae3 = mean_absolute_error(y_test, prediction3)
r2_3 = r2_score(y_test, prediction3)


# --------------------------------------------------
# MODEL ALGORITHM COMPARISON
# Using G1 + G2 because this was our best feature set
# --------------------------------------------------

X = data[["G1", "G2"]]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

linear_model = LinearRegression()

tree_model = DecisionTreeRegressor(
    max_depth=4,
    random_state=42
)

forest_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

linear_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
forest_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)
tree_predictions = tree_model.predict(X_test)
forest_predictions = forest_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_r2 = r2_score(y_test, linear_predictions)

tree_mae = mean_absolute_error(y_test, tree_predictions)
tree_r2 = r2_score(y_test, tree_predictions)

forest_mae = mean_absolute_error(y_test, forest_predictions)
forest_r2 = r2_score(y_test, forest_predictions)


# --------------------------------------------------
# DISPLAY FEATURE EXPERIMENT RESULTS
# --------------------------------------------------

print("FEATURE EXPERIMENT")
print("=" * 50)

print("\nModel 1: G1 + G2")
print(f"MAE: {mae1:.2f}")
print(f"R²:  {r2_1:.2f}")

print("\nModel 2: G1 + G2 + Studytime")
print(f"MAE: {mae2:.2f}")
print(f"R²:  {r2_2:.2f}")

print("\nModel 3: G1 + G2 + Studytime + Failures + Absences")
print(f"MAE: {mae3:.2f}")
print(f"R²:  {r2_3:.2f}")

print("\nModel 3 coefficients:")
for feature, coefficient in zip(X3.columns, model3.coef_):
    print(f"{feature}: {coefficient:.3f}")


# --------------------------------------------------
# DISPLAY ALGORITHM COMPARISON
# --------------------------------------------------

print("\n\nMODEL ALGORITHM COMPARISON")
print("=" * 50)

print("\nLinear Regression")
print(f"MAE: {linear_mae:.2f}")
print(f"R²:  {linear_r2:.2f}")

print("\nDecision Tree")
print(f"MAE: {tree_mae:.2f}")
print(f"R²:  {tree_r2:.2f}")

print("\nRandom Forest")
print(f"MAE: {forest_mae:.2f}")
print(f"R²:  {forest_r2:.2f}")


# --------------------------------------------------
# FIND BEST MODEL
# --------------------------------------------------

models_results = {
    "Linear Regression": linear_mae,
    "Decision Tree": tree_mae,
    "Random Forest": forest_mae
}

best_model_name = min(
    models_results,
    key=models_results.get
)

print("\nBest model based on MAE:")
print(best_model_name)


# --------------------------------------------------
# ACTUAL VS PREDICTED GRAPH
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(y_test, linear_predictions)

minimum = min(
    y_test.min(),
    linear_predictions.min()
)

maximum = max(
    y_test.max(),
    linear_predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Final Grade (G3)")
plt.ylabel("Predicted Final Grade")
plt.title("Actual vs Predicted Final Grades")

plt.show()


# --------------------------------------------------
# ERROR ANALYSIS
# --------------------------------------------------

errors = y_test - linear_predictions

print("\nError Analysis")
print("----------------")

print(f"Mean error: {errors.mean():.2f}")
print(f"Largest overprediction: {errors.min():.2f}")
print(f"Largest underprediction: {errors.max():.2f}")


# --------------------------------------------------
# WORST PREDICTIONS
# --------------------------------------------------

results = X_test.copy()

results["Actual_G3"] = y_test
results["Predicted_G3"] = linear_predictions
results["Error"] = y_test - linear_predictions
results["Absolute_Error"] = abs(results["Error"])

worst_predictions = results.sort_values(
    "Absolute_Error",
    ascending=False
).head(5)

print("\nWorst 5 Predictions")
print("-------------------")

print(worst_predictions)