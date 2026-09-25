# Iris Flower Classification using K-Nearest Neighbors (KNN)

A beginner-friendly, end-to-end Machine Learning web application that predicts the species of an Iris flower (Setosa, Versicolor, or Virginica) from its sepal and petal measurements, using a K-Nearest Neighbors classifier served through a Flask web app.

---

## 1. Project Overview

This project demonstrates a complete, simple ML workflow: training a model on a well-known dataset, evaluating it properly, saving it to disk, and serving predictions through a small web application. It is intentionally kept free of unnecessary abstractions, frameworks, or databases so that it's easy to read from top to bottom and understand every step.

---

## 2. Objective

- Train a K-Nearest Neighbors (KNN) classifier to distinguish between three Iris species based on four flower measurements.
- Build a simple web interface where a user can enter measurements and instantly get a prediction.
- Keep ML training completely separate from the web application (train once, predict many times).

---

## 3. Technologies Used

| Layer            | Technology                          |
|-------------------|--------------------------------------|
| Backend           | Python, Flask                        |
| Machine Learning  | scikit-learn, pandas, NumPy, joblib  |
| Visualization     | Matplotlib, Seaborn                  |
| Frontend          | HTML, CSS, JavaScript (no frameworks)|

No React, Node.js, TypeScript, databases, or Docker are used — just Python + Flask + HTML/CSS/JS.

---

## 4. Dataset Explanation

The project uses the **Iris dataset**, a classic dataset built directly into scikit-learn (`sklearn.datasets.load_iris`). It contains **150 samples** of iris flowers, each described by 4 numeric features (in centimeters):

1. Sepal Length
2. Sepal Width
3. Petal Length
4. Petal Width

Each sample is labeled with one of 3 species (50 samples each):

- **Iris Setosa**
- **Iris Versicolor**
- **Iris Virginica**

---

## 5. Machine Learning Workflow

The workflow, implemented in `train_model.py`, follows these steps:

1. Load the Iris dataset from `sklearn.datasets`.
2. Separate the data into features (`X`) and target labels (`y`).
3. Split the data into training (80%) and testing (20%) sets.
4. Standardize the features using `StandardScaler`.
5. Train a `KNeighborsClassifier` with `K = 5`.
6. Evaluate the model on the held-out test set.
7. Calculate accuracy, a confusion matrix, and a classification report.
8. Save the trained model and scaler to disk using `joblib`.

The Flask app (`app.py`) then simply **loads** the saved model and scaler — it never retrains anything.

---

## 6. Explanation of KNN (K-Nearest Neighbors)

KNN is one of the simplest machine learning algorithms, and a great one for beginners to learn:

> To classify a new, unseen flower, KNN looks at the "K" most similar flowers (its nearest neighbors) in the training data — measured by distance between feature values — and assigns the new flower the species that is most common among those neighbors (a majority vote).

In this project, **K = 5**, meaning each prediction is based on a vote among the 5 most similar flowers in the training data.

KNN has no real "training" phase in the traditional sense — it just memorizes the training data and does all its work at prediction time by computing distances.

---

## 7. Why StandardScaler is Used

KNN relies entirely on **distance** between data points to find "nearest" neighbors. If features are on different scales (for example, one feature ranging 0–1 and another 0–100), the feature with the larger range would dominate the distance calculation, even if it's not actually more important.

`StandardScaler` transforms every feature so it has a **mean of 0** and a **standard deviation of 1**. This puts all four measurements (sepal length, sepal width, petal length, petal width) on equal footing, so the model treats each one fairly.

**Important:** the scaler is *fit* only on the training data, then used to *transform* both the training and test data (and later, any new user input) — this avoids leaking information from the test set into training.

---

## 8. Train / Test Split

The dataset is split using `train_test_split` from scikit-learn:

- **80%** of the data (120 samples) is used for **training**.
- **20%** of the data (30 samples) is held out for **testing**.
- `stratify=y` ensures each species is proportionally represented in both sets.
- `random_state=42` makes the split reproducible.

Testing on unseen data is essential — it tells us how well the model generalizes, rather than how well it simply memorized the training examples.

---

## 9. Model Evaluation

`train_model.py` prints the following when it runs:

- Dataset information (shape, feature names, class distribution)
- Training and testing set sizes
- **Accuracy** — overall percentage of correct predictions
- **Confusion Matrix** — a table of actual vs. predicted classes
- **Classification Report** — precision, recall, and F1-score per species

It also saves a visual heatmap of the confusion matrix as `confusion_matrix.png`.

With K=5 on this dataset, the model typically achieves around **95–100% accuracy** on the test set, since the Iris dataset is small, clean, and well-separated.

---

## 10. Project Structure

```
iris-classification/
│
├── app.py                    Flask web application (loads model, handles predictions)
├── train_model.py             Trains the KNN model and saves model.pkl + scaler.pkl
├── model.pkl                   Saved, trained KNN model (generated by train_model.py)
├── scaler.pkl                   Saved, fitted StandardScaler (generated by train_model.py)
├── confusion_matrix.png       Saved evaluation plot (generated by train_model.py)
├── requirements.txt             Python dependencies
├── README.md
├── .gitignore
│
├── templates/
│   └── index.html               Main web page (input form + result display)
│
└── static/
    ├── style.css                 Styling for the web page
    └── script.js                  Frontend logic (form validation, API calls)
```

---

## 11. How to Install

Make sure you have **Python 3.9+** installed. Then, from the project folder, install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 12. How to Run

**Step 1 — Train the model** (creates `model.pkl`, `scaler.pkl`, and `confusion_matrix.png`):

```bash
python train_model.py
```

**Step 2 — Start the Flask web app:**

```bash
python app.py
```

**Step 3 — Open your browser** and go to:

```
http://127.0.0.1:5000
```

---

## 13. Example Prediction

Enter the following values into the web form:

| Field         | Value |
|---------------|-------|
| Sepal Length  | 5.1   |
| Sepal Width   | 3.5   |
| Petal Length  | 1.4   |
| Petal Width   | 0.2   |

Expected result: **Iris Setosa** (these values are very close to a real Setosa sample from the dataset).

Try these values for a Virginica-like prediction:

| Field         | Value |
|---------------|-------|
| Sepal Length  | 6.7   |
| Sepal Width   | 3.0   |
| Petal Length  | 5.2   |
| Petal Width   | 2.3   |

Expected result: **Iris Virginica**.

---

## 14. Future Improvements

- Add a chart comparing user input against typical measurements for each species.
- Let the user experiment with different values of K directly from the UI.
- Add unit tests for the Flask API endpoints.
- Try other classifiers (e.g. Logistic Regression, Decision Tree) and compare accuracy.
- Add cross-validation to choose the best K more rigorously.
- Deploy the app publicly (e.g. Render, Railway, PythonAnywhere).

---

## License

This project is for educational purposes.
