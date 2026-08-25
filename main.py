import tkinter as tk
from tkinter import messagebox
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

    # Keep prediction within the valid G3 range
    prediction = max(0, min(20, prediction))

    return prediction


# --------------------------------------------------
# GUI FUNCTIONS
# --------------------------------------------------

def predict():
    try:
        g1 = float(g1_entry.get())
        g2 = float(g2_entry.get())

        # Validate inputs
        if not 0 <= g1 <= 20:
            raise ValueError("G1 must be between 0 and 20.")

        if not 0 <= g2 <= 20:
            raise ValueError("G2 must be between 0 and 20.")

        predicted_g3 = predict_final_grade(g1, g2)
        percentage = (predicted_g3 / 20) * 100

        if predicted_g3 >= 10:
            result = "PASS"
        else:
            result = "FAIL"

        prediction_label.config(
            text=f"Predicted final grade: {predicted_g3:.2f} / 20"
        )

        percentage_label.config(
            text=f"Predicted percentage: {percentage:.2f}%"
        )

        result_label.config(
            text=f"Predicted result: {result}"
        )

    except ValueError as error:
        messagebox.showerror("Invalid Input", str(error))


def clear_inputs():
    g1_entry.delete(0, tk.END)
    g2_entry.delete(0, tk.END)

    prediction_label.config(
        text="Predicted final grade: -- / 20"
    )

    percentage_label.config(
        text="Predicted percentage: --"
    )

    result_label.config(
        text="Predicted result: --"
    )


# --------------------------------------------------
# CREATE WINDOW
# --------------------------------------------------

root = tk.Tk()

root.title("Student Final Performance Predictor")
root.geometry("600x600")
root.resizable(False, False)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

title_label = tk.Label(
    root,
    text="Student Final Performance Predictor",
    font=("Arial", 20, "bold")
)

title_label.pack(pady=20)


# --------------------------------------------------
# DESCRIPTION
# --------------------------------------------------

description_label = tk.Label(
    root,
    text="Predict final performance using the student's\n"
         "first-period and second-period grades.",
    font=("Arial", 11)
)

description_label.pack(pady=5)


# --------------------------------------------------
# INPUT FRAME
# --------------------------------------------------

input_frame = tk.Frame(root)

input_frame.pack(pady=25)


# G1

g1_label = tk.Label(
    input_frame,
    text="First-period grade (G1):",
    font=("Arial", 12)
)

g1_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")


g1_entry = tk.Entry(
    input_frame,
    font=("Arial", 12),
    width=15
)

g1_entry.grid(row=0, column=1, padx=10, pady=10)


# G2

g2_label = tk.Label(
    input_frame,
    text="Second-period grade (G2):",
    font=("Arial", 12)
)

g2_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")


g2_entry = tk.Entry(
    input_frame,
    font=("Arial", 12),
    width=15
)

g2_entry.grid(row=1, column=1, padx=10, pady=10)


# --------------------------------------------------
# BUTTONS
# --------------------------------------------------

button_frame = tk.Frame(root)

button_frame.pack(pady=10)


predict_button = tk.Button(
    button_frame,
    text="Predict Final Grade",
    command=predict,
    font=("Arial", 12, "bold"),
    width=18
)

predict_button.grid(row=0, column=0, padx=10)


clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear_inputs,
    font=("Arial", 12),
    width=10
)

clear_button.grid(row=0, column=1, padx=10)


# --------------------------------------------------
# RESULT FRAME
# --------------------------------------------------

result_frame = tk.Frame(root)

result_frame.pack(pady=25)

prediction_label = tk.Label(
    result_frame,
    text="Predicted final grade: -- / 20",
    font=("Arial", 14, "bold")
)

prediction_label.pack(pady=5)

percentage_label = tk.Label(
    result_frame,
    text="Predicted percentage: --",
    font=("Arial", 13)
)

percentage_label.pack(pady=5)

result_label = tk.Label(
    result_frame,
    text="Predicted result: --",
    font=("Arial", 13)
)

result_label.pack(pady=5)


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

model_frame = tk.Frame(root)

model_frame.pack(pady=10)

model_info_label = tk.Label(
    model_frame,
    text=(
        f"Model: Linear Regression\n"
        f"MAE: {mae:.2f} grade points\n"
        f"R²: {r2:.2f}\n"
        f"Predictions are estimates, not guaranteed results."
    ),
    font=("Arial", 10),
    justify="center"
)

model_info_label.pack()


# --------------------------------------------------
# START APPLICATION
# --------------------------------------------------

root.mainloop()