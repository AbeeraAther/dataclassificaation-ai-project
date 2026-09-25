import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


iris = load_iris()

X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

iris_df = pd.DataFrame(X, columns=feature_names)
iris_df["species"] = [target_names[label] for label in y]

print("=" * 60)
print("IRIS DATASET INFORMATION")
print("=" * 60)
print(f"Total samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Feature names: {feature_names}")
print(f"Target classes: {list(target_names)}")
print("\nFirst 5 rows of the dataset:")
print(iris_df.head())
print("\nClass distribution (samples per species):")
print(iris_df["species"].value_counts())


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)
print("KNN classifier trained with K = 5 neighbors.")


y_pred = knn_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)

class_report = classification_report(y_test, y_pred, target_names=target_names)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)


plt.figure(figsize=(6, 5))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names,
)
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.title(f"Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("\nConfusion matrix plot saved as 'confusion_matrix.png'")


joblib.dump(knn_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n" + "=" * 60)
print("SAVED FILES")
print("=" * 60)
print("Trained model saved to: model.pkl")
print("Fitted scaler saved to: scaler.pkl")
print("\nTraining complete! You can now run the Flask app with: python app.py")
