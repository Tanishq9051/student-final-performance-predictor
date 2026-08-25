# 🎓 Student Final Performance Predictor

A Python-based **Machine Learning application** that predicts a student's final academic performance from their previous examination grades.

The project was developed as a complete ML workflow, starting with a basic regression model and progressing through feature experimentation, algorithm comparison, model evaluation, error analysis, and a graphical user interface.

---

## 📌 Overview

The **Student Final Performance Predictor** uses historical student performance data to estimate a student's final grade (`G3`) from their first-period (`G1`) and second-period (`G2`) grades.

Unlike a simple marks calculator, the prediction is produced by a **machine learning model trained on historical student data**.

The project follows this workflow:

```text
Student Dataset
      ↓
Data Preparation
      ↓
Feature Selection
      ↓
Train / Test Split
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Algorithm Comparison
      ↓
Error Analysis
      ↓
Final Prediction Application
```

---

## 🎯 Motivation

The goal of this project was to explore how **machine learning can be used to predict a numerical academic outcome** from previous student performance.

Instead of manually creating a formula for predicting final marks, the model learns the relationship between previous grades and final grades from historical data.

The project also focuses on understanding an important part of ML development:

> A more complicated model or more features do not automatically produce better predictions.

---

## 📊 Results at a Glance

| Attribute         | Result                          |
| ----------------- | ------------------------------- |
| Dataset           | UCI Student Performance         |
| Dataset used      | Mathematics (`student-mat.csv`) |
| Students          | **395**                         |
| Final model       | **Linear Regression**           |
| Input features    | **G1, G2**                      |
| Target            | **G3**                          |
| Test split        | **20%**                         |
| MAE               | **1.26 grade points**           |
| R²                | **0.79**                        |
| Algorithms tested | **3**                           |
| GUI               | **Tkinter**                     |

---

## ✨ Features

* Final grade prediction using Machine Learning
* Linear Regression based prediction
* Train/test data splitting
* MAE and R² evaluation
* Feature experimentation
* Comparison of multiple regression algorithms
* Actual vs predicted visualization
* Prediction error analysis
* Identification of worst predictions
* Input validation
* Tkinter graphical user interface
* Displays model performance alongside predictions
* Preserved development versions showing project progression

---

# 🤖 Machine Learning Approach

## Input Features

The final model uses:

| Feature | Description         |
| ------- | ------------------- |
| `G1`    | First-period grade  |
| `G2`    | Second-period grade |

## Target

| Target | Description        |
| ------ | ------------------ |
| `G3`   | Final-period grade |

The grades in the dataset are represented on a **0–20 scale**.

The final prediction pipeline is:

```text
G1 + G2
   ↓
Linear Regression
   ↓
Predicted G3
   ↓
Percentage Conversion
   ↓
PASS / FAIL
```

---

## 🧮 Linear Regression

The final model learns a relationship of the form:

```text
G3 = b₀ + b₁(G1) + b₂(G2)
```

Where:

* `G1` = first-period grade
* `G2` = second-period grade
* `G3` = predicted final grade
* `b₀` = intercept
* `b₁` and `b₂` = coefficients learned from the training data

These coefficients are **learned from the dataset** rather than manually chosen.

---

# 📚 Dataset

This project uses the **Student Performance** dataset from the UCI Machine Learning Repository.

The mathematics dataset contains **395 student records** and includes academic, demographic, social, and school-related variables.

Although the dataset contains many variables, the final predictor uses only `G1` and `G2`. Additional features were tested during development to determine whether they improved the model.

### Dataset Source

**UCI Machine Learning Repository — Student Performance**

https://archive.ics.uci.edu/dataset/320/student+performance

The dataset is subject to its own licensing and attribution requirements and is separate from the license applied to this project's source code.

---

# 🧪 Feature Experiments

An important part of the project was testing whether additional information improved prediction.

## Model 1 — Baseline

```text
G1 + G2 → G3
```

| Metric |   Result |
| ------ | -------: |
| MAE    | **1.26** |
| R²     | **0.79** |

## Model 2 — Add Study Time

```text
G1 + G2 + studytime → G3
```

| Metric |   Result |
| ------ | -------: |
| MAE    |     1.27 |
| R²     | **0.80** |

The change in R² was very small and MAE became slightly worse.

## Model 3 — Add Additional Academic Information

```text
G1 + G2 + studytime + failures + absences → G3
```

| Metric | Result |
| ------ | -----: |
| MAE    |   1.34 |
| R²     |   0.78 |

### Feature Experiment Summary

| Feature Set                               |    MAE ↓ |     R² ↑ |
| ----------------------------------------- | -------: | -------: |
| **G1 + G2**                               | **1.26** |     0.79 |
| G1 + G2 + studytime                       |     1.27 | **0.80** |
| G1 + G2 + studytime + failures + absences |     1.34 |     0.78 |

### Conclusion

The simplest feature set produced the **lowest MAE**.

Therefore, the final predictor uses:

```text
G1 + G2
```

rather than adding features that did not meaningfully improve prediction performance.

---

# 🔬 Algorithm Comparison

Three regression algorithms were tested using the same train/test split.

| Model                   |    MAE ↓ |     R² ↑ |
| ----------------------- | -------: | -------: |
| **Linear Regression**   | **1.26** | **0.79** |
| Decision Tree Regressor |     1.40 |     0.76 |
| Random Forest Regressor |     1.36 |     0.76 |

## 🏆 Selected Model

**Linear Regression**

It achieved the lowest MAE and the highest R² among the tested algorithms.

This demonstrates that a more complex algorithm does not necessarily perform better on a particular dataset.

---

# 📈 Model Evaluation

The model was evaluated on a **held-out 20% test set** that was not used during training.

## Mean Absolute Error — MAE

```text
MAE = 1.26 grade points
```

MAE represents the average absolute difference between the model's predicted final grade and the actual final grade.

A value of **1.26** means the predictions were off by approximately 1.26 grade points on average in absolute terms on the held-out test data.

## R² Score

```text
R² = 0.79
```

R² measures how well the model explains variation in the target values.

R² is reported together with MAE rather than being treated as a percentage accuracy measure.

---

# 📉 Prediction Visualization

The project includes an **actual vs predicted** graph for the final Linear Regression model.

The ideal relationship is:

```text
Actual Grade = Predicted Grade
```

Therefore, predictions closer to the diagonal relationship indicate better agreement between predicted and actual grades.

Most of the test predictions were reasonably close to this relationship.

---

# 🔎 Error Analysis

The project also examined individual prediction errors rather than relying only on overall metrics.

The error was calculated as:

```text
Error = Actual G3 - Predicted G3
```

The mean error was approximately:

```text
+0.16
```

The model showed several large errors for students whose recorded final grade was `0`.

For example, some students had relatively high previous grades but a final grade of `0`, causing the model to predict substantially higher values.

This revealed an important limitation:

> Previous grades alone cannot explain every unusual final outcome.

Rather than removing these observations simply because they were difficult to predict, they were retained and discussed as part of the model's limitations.

---

# 🖥️ Application

The final version of the project uses **Tkinter** to provide a simple graphical interface.

### User Input

```text
First-period grade (G1)
Second-period grade (G2)
```

### Application Output

```text
Predicted final grade
Predicted percentage
PASS / FAIL
Model name
MAE
R²
```

### Example

For:

```text
G1 = 12
G2 = 14
```

the trained model produces approximately:

```text
Predicted final grade : 13.83 / 20
Predicted percentage  : 69.13%
Predicted result      : PASS
Typical prediction error: ±1.26 grade points
```

The prediction is an estimate and should not be interpreted as a guaranteed individual result.

---

# 🖼️ Screenshots

## Application Dashboard

![Dashboard](screenshot/dashboard.PNG)

[🔗 Open Dashboard Screenshot](screenshot/dashboard.PNG)

## Prediction Result

![Prediction Result](screenshot/result.PNG)

[🔗 Open Prediction Result Screenshot](screenshot/result.PNG)

---

# 🛠️ Technologies Used

* **Python 3**
* **Pandas** — data loading and data manipulation
* **Scikit-learn** — machine learning and model evaluation
* **Matplotlib** — visualization and error analysis
* **Tkinter** — graphical user interface

---

# 📁 Project Structure

```text
student-final-performance-predictor/
│
├── main.py                  # Final Tkinter application
├── v1_model.py              # V1: basic ML predictor
├── v2_analysis.py           # V2: feature and algorithm experiments
├── v3_terminal.py           # V3: terminal-based application
│
├── requirements.txt         # Required Python packages
├── README.md                # Project documentation
├── LICENSE                  # MIT License
├── .gitignore
│
└── screenshot/
    ├── dashboard.PNG
    └── result.PNG
```

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/Tanishq9051/student-final-performance-predictor.git
```

## 2. Enter the project directory

```bash
cd student-final-performance-predictor
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Download the dataset

Download the mathematics student performance file:

```text
student-mat.csv
```

from the UCI Student Performance dataset and place it in the project directory.

## 5. Run the application

```bash
python main.py
```

---

# 🧪 Development Versions

The project was developed incrementally.

| Version   | Description                                                                  |
| --------- | ---------------------------------------------------------------------------- |
| **V1**    | Basic Linear Regression model, evaluation, and prediction                    |
| **V2**    | Feature experiments, algorithm comparison, visualization, and error analysis |
| **V3**    | Terminal-based prediction application                                        |
| **Final** | Tkinter graphical application                                                |

The previous versions are preserved in the repository to document the development process and experimentation.

---

# 💡 Key Findings

During development, several useful observations were made:

* `G1` and `G2` provided the strongest predictive signal among the tested feature sets.
* Adding `studytime` produced negligible improvement.
* Adding `failures` and `absences` reduced performance on the selected test split.
* Linear Regression performed better than the tested Decision Tree and Random Forest models.
* The more complex algorithms did not automatically improve prediction quality.
* Most predictions were reasonably close to the actual final grades.
* The model struggled with exceptional cases, particularly some students with `G3 = 0`.
* Previous grades alone cannot explain every factor affecting final examination performance.

---

# 📚 Learning Outcomes

Through this project, I learned and practiced:

* Fundamentals of supervised machine learning
* Regression and numerical prediction
* Features and target variables
* Dataset preparation using Pandas
* Train/test splitting
* Linear Regression
* Model coefficients and intercepts
* Model evaluation using MAE and R²
* Comparing machine learning algorithms
* Feature experimentation
* Prediction error analysis
* Data visualization using Matplotlib
* Building a GUI with Tkinter
* Input validation
* Understanding limitations of predictive models
* Turning a trained ML model into a usable application

---

# ⚠️ Limitations

This project is an educational machine learning application and should not be treated as a professional academic forecasting system.

The model was evaluated using a single held-out test split from the UCI dataset. Its performance may differ on other datasets, schools, populations, grading systems, or examination structures.

The model identifies statistical relationships in historical data. It does **not** establish causal relationships.

In particular, `G1` and `G2` cannot explain every exceptional final result.

The prediction should therefore be treated as:

> **An estimated outcome based on learned historical patterns, not a guaranteed future result.**

---

# 🔗 Project Links

* **GitHub Repository:**
  https://github.com/Tanishq9051/student-final-performance-predictor

* **Dataset:**
  https://archive.ics.uci.edu/dataset/320/student+performance

* **Related Project — Marks Analyser:**
  https://github.com/Tanishq9051/Marks-analyser

---

# 👤 Author

**Tanishq Hardeniya**

Python • Machine Learning • Data Analysis

---

# 📄 License

The source code in this repository is released under the **MIT License**.

The UCI Student Performance dataset is subject to its own licensing and attribution requirements.
