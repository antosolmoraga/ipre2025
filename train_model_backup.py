import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # para guardar el modelo entrenado

# ------------------------------
# 1. Cargar dataset
# ------------------------------
df = pd.read_csv("dataset_final.csv")

# Separar características (X) y etiquetas (y)
canales = ['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8']
X = df[canales]
y = df['etiqueta']  # 0 = relajacion, 1 = contraccion

# ------------------------------
# 2. Dividir en train/test
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# 3. Crear modelo y entrenar
# ------------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ------------------------------
# 4. Evaluar modelo
# ------------------------------
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ------------------------------
# 5. Guardar modelo entrenado
# ------------------------------
joblib.dump(clf, "emg_model.pkl")
print("\nModelo guardado como 'emg_model.pkl'")
