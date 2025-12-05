import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # para guardar el modelo entrenado


# ------------------------------
# 1. Cargar dataset
# ------------------------------
df = pd.read_csv("dataset_final_canal2y4.csv")


# Verifica qué columnas tiene
print("Columnas del dataset:", df.columns.tolist())


# ------------------------------
# 2. Separar características (X) y etiquetas (y)
# ------------------------------
# Solo usaremos la columna procesada del canal 2
X = df[["ch2_proc", "ch4_proc", "ch6_proc"]]
y = df["etiqueta"]


# ------------------------------
# 3. Dividir en train/test
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ------------------------------
# 4. Crear modelo y entrenar
# ------------------------------
clf = RandomForestClassifier(n_estimators=8, random_state=42)
clf.fit(X_train, y_train)


# ------------------------------
# 5. Evaluar modelo
# ------------------------------
y_pred = clf.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

y_pred = clf.predict(X_train)
print("\nAccuracy:", accuracy_score(y_train, y_pred))
print("\nClassification Report:\n", classification_report(y_train, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_train, y_pred))


# ------------------------------
# 6. Guardar modelo entrenado
# ------------------------------
joblib.dump(clf, "emg_model.pkl")
print("\n Modelo guardado como 'emg_model.pkl'")
