import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

print("Chargement des données...")
try:
    X = pd.read_csv('X_processed.csv')
    y = pd.read_csv('y_labels.csv').iloc[:, 0]
except:
    print("Erreur de fichier.")
    exit()

X, poubelle1, y, poubelle2 = train_test_split(X, y, test_size=0.5, random_state=42, shuffle=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print("\nEntraînement du Random Forest ...")
start_time = time.time()

# n_estimators=100 : On plante 100 arbres 
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=10, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train)

param_grid = {
    'n_estimators': [100,200],
    'max_features': ['sqrt', 'log2'],
    'max_depth' : [10,20, None],
    'min_samples_split' : [2,10]
}

#CV_rfc = GridSearchCV(estimator=RandomForestClassifier(random_state=42, n_jobs=-1), param_grid=param_grid, cv= 5)
#CV_rfc.fit(X_train, y_train)

#print("meilleurs paramètres")
#print(CV_rfc.best_params_)


print(f"Entraînement terminé en {time.time() - start_time:.2f} secondes.")

print("\nÉvaluation test")
y_pred = rf_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Random Forest test: {accuracy:.2%}")
print("-" * 30)
print(classification_report(y_test, y_pred))

print("\nÉvaluation entraînement")
y_pred = rf_clf.predict(X_train)

accuracy = accuracy_score(y_train, y_pred)
print(f"Accuracy Random Forest entraînement: {accuracy:.2%}")
print("-" * 30)
print(classification_report(y_train, y_pred))

print("\nEvaluation Colorado")

try:
    X_co = pd.read_csv('X_processed_co.csv')
    y_co = pd.read_csv('y_labels_co.csv').iloc[:, 0]
except:
    print("Erreur de fichier.")
    exit()

y_pred = rf_clf.predict(X_co)

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

y_pred = rf_clf.predict(X_ne)

accuracy = accuracy_score(y_ne, y_pred)
print(f"Accuracy Random Forest sur dataset Nevada: {accuracy:.2%}")
print("-" * 30)
print(classification_report(y_ne, y_pred))



# On regarde quelles colonnes le modèle a le plus utilisées
importances = rf_clf.feature_importances_
feature_names = X.columns

feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(10) # Top 10

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
plt.title('Top 10 des critères les plus importants (Random Forest)')
plt.xlabel('Importance (Score de Gini)')
plt.tight_layout()
plt.savefig('feature_importance_rf.png')
print(0.2*33263)
print(0.8*33263)
