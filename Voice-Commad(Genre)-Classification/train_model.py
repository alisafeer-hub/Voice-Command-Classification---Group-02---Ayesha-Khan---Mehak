import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Data Load Karna
print("Music Genre data load ho raha hai...")
df = pd.read_csv('dataset.csv')

# 2. Features (X) aur Target (y) ko alag karna
X = df[['tempo', 'instrument_loudness_db', 'vocal_pitch_hz']]
y = df['genre']

# 3. Data ko Train aur Test set mein divide karna (80/20 split)
# Ab yeh 800 training samples aur 200 testing samples honge
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Total samples: {len(df)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# 4. Model ko initialize karna
model = LogisticRegression(random_state=42, multi_class='ovr')

# 5. Model ko Train karna
print("\nModel train ho raha hai (800 samples par)...")
model.fit(X_train, y_train)
print("Model train ho gaya!")

# 6. Model ko Test (Evaluate) karna (200 samples par)
print("\nModel ko evaluate kiya ja raha hai...")
y_pred = model.predict(X_test)

# Accuracy check karna
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel ki Accuracy: {accuracy * 100:.2f}%")

# Detailed report dekhna
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Trained Model ko Save karna
model_filename = 'genre_model.pkl'
joblib.dump(model, model_filename)

print(f"\nModel save ho gaya hai '{model_filename}' file mein.")

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# --- Random Forest Classifier ---
print("\n==============================")
print("Random Forest Model Train ho raha hai...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, rf_pred))

# Model save karna
joblib.dump(rf_model, 'genre_model_randomforest.pkl')
print("Random Forest model 'genre_model_randomforest.pkl' mein save ho gaya hai.")


# --- Support Vector Machine (SVM) ---
print("\n==============================")
print("SVM Model Train ho raha hai...")
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_pred)
print(f"SVM Model Accuracy: {svm_accuracy * 100:.2f}%")
print("\nClassification Report (SVM):")
print(classification_report(y_test, svm_pred))

# Model save karna
joblib.dump(svm_model, 'genre_model_svm.pkl')
print("SVM model 'genre_model_svm.pkl' mein save ho gaya hai.")


# --- Comparison Summary ---
print("\n==============================")
print("Model Accuracy Comparison:")
print(f"1️⃣ Logistic Regression: {accuracy * 100:.2f}%")
print(f"2️⃣ Random Forest:       {rf_accuracy * 100:.2f}%")
print(f"3️⃣ SVM:                 {svm_accuracy * 100:.2f}%")
