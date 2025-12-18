import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

print("Chargement des données...")
try:
    X = pd.read_csv('X_processed.csv')
    y = pd.read_csv('y_labels.csv').iloc[:, 0]
except:
    print("Erreur de fichier.")
    exit()

X, poubelle1, y, poubelle2 = train_test_split(X, y, test_size=0.8, random_state=42, shuffle=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("\nEntraînement du AdaBoost ...")
start_time = time.time()

# n_estimators=100 : 100 weak learners
ada_clf = AdaBoostClassifier(
    n_estimators=150,
    learning_rate=1.5,
    random_state=42
)
ada_clf.fit(X_train, y_train)

print(f"Entraînement terminé en {time.time() - start_time:.2f} secondes.")

print("\nÉvaluation test")
y_pred = ada_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy AdaBoost test: {accuracy:.2%}")
print("-" * 30)
print(classification_report(y_test, y_pred))

print("\nÉvaluation entraînement")
y_pred_train = ada_clf.predict(X_train)

accuracy_train = accuracy_score(y_train, y_pred_train)
print(f"Accuracy AdaBoost entraînement: {accuracy_train:.2%}")
print("-" * 30)
print(classification_report(y_train, y_pred_train))


print("\nEvaluation Colorado")

try:
    X_co = pd.read_csv('X_processed_co.csv')
    y_co = pd.read_csv('y_labels_co.csv').iloc[:, 0]
except:
    print("Erreur de fichier.")
    exit()

y_pred = ada_clf.predict(X_co)

accuracy = accuracy_score(y_co, y_pred)
print(f"Accuracy Random Forest sur dataset Colorado: {accuracy:.2%}")
print("-" * 30)
print(classification_report(y_co, y_pred))

print("\nEvaluation Nevada")

try:
    X_ne = pd.read_csv('X_processed_ne.csv')
    y_ne = pd.read_csv('y_labels_ne.csv').iloc[:, 0]
except:
    print("Erreur de fichier.")
    exit()

y_pred = ada_clf.predict(X_ne)

accuracy = accuracy_score(y_ne, y_pred)
print(f"Accuracy Random Forest sur dataset Nevada: {accuracy:.2%}")
print("-" * 30)
print(classification_report(y_ne, y_pred))

# -------------------------------
# Importances des features (AdaBoost)
# -------------------------------

importances = ada_clf.feature_importances_
feature_names = X.columns

feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
plt.title('Top 10 des critères les plus importants (AdaBoost)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance_ada.png')