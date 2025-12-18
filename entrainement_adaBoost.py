import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


try:
    X = pd.read_csv('X_processed.csv')
    y = pd.read_csv('y_labels.csv').iloc[:, 0]
except FileNotFoundError:
    print("ERREUR : Fichiers manquants.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# EXPE 1 : Paramètres par défaut
print("\n--- 1 : Paramètres par défaut ---")

# Par défaut, AdaBoost utilise 50 arbres et un learning rate de 1.0
ada_default = AdaBoostClassifier(random_state=42)

start_time = time.time()
ada_default.fit(X_train, y_train)
train_time_def = time.time() - start_time

acc_train_def = ada_default.score(X_train, y_train)
acc_test_def = ada_default.score(X_test, y_test)

print(f"Temps d'entraînement : {train_time_def:.4f} s")
print(f"Accuracy Train : {acc_train_def:.2%}")
print(f"Accuracy Test  : {acc_test_def:.2%}")

# EXPE 2 : Optimisation (GridSearch)

print("\n--- 2 : Optimisation des Hyperparamètres ---")

# On utilise des "Decision Stumps" (arbres de profondeur 1) comme base, c'est le standard pour AdaBoost
base_estimator = DecisionTreeClassifier(max_depth=1)

param_grid = {
    'n_estimators': [50, 100, 150],      # Combien de corrections successives ?
    'learning_rate': [0.1, 0.5, 1.0, 1.5] # Force de la correction
}

print(f"Grille testée : {param_grid}")
print("Lancement du GridSearch...")

# Note : AdaBoost ne supporte pas n_jobs=-1 directement pour le fit, 
# mais GridSearchCV peut paralléliser les tests de combinaisons.
grid = GridSearchCV(AdaBoostClassifier(estimator=base_estimator, random_state=42), 
                    param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)

start_time = time.time()
grid.fit(X_train, y_train)
train_time_opt = time.time() - start_time

best_ada = grid.best_estimator_
acc_train_opt = best_ada.score(X_train, y_train)
acc_test_opt = best_ada.score(X_test, y_test)

print("\n>>> RÉSULTATS OPTIMISÉS <<<")
print(f"Meilleurs paramètres : {grid.best_params_}")
print(f"Temps de recherche : {train_time_opt:.2f} s")
print(f"Accuracy Train (Best) : {acc_train_opt:.2%}")
print(f"Accuracy Test (Best)  : {acc_test_opt:.2%}")


print("\nEvaluation Colorado")

try:
    X_co = pd.read_csv('X_processed_co.csv')
    y_co = pd.read_csv('y_labels_co.csv').iloc[:, 0]
except:
    print("Erreur de fichier.")
    exit()

y_pred = ada_default.predict(X_co)

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

y_pred = ada_default.predict(X_ne)

accuracy = accuracy_score(y_ne, y_pred)
print(f"Accuracy Random Forest sur dataset Nevada: {accuracy:.2%}")
print("-" * 30)
print(classification_report(y_ne, y_pred))

# -------------------------------
# Importances des features (AdaBoost)
# -------------------------------


# Graphique Feature Importance
importances = best_ada.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='magma')
plt.title(f'Feature Importance (AdaBoost Opti - Acc: {acc_test_opt:.2%})')
plt.tight_layout()
plt.savefig('resultat_adaboost.png')
print("\nGraphique 'resultat_adaboost.png' sauvegardé.")

print("\n--- RÉSUMÉ POUR LE TABLEAU DU RAPPORT ---")
print(f"| Modèle | Train Acc | Train Time | Test Acc |")
print(f"| Default| {acc_train_def:.4f} | {train_time_def:.4f}s | {acc_test_def:.4f} |")
print(f"| Opti   | {acc_train_opt:.4f} | {train_time_opt:.4f}s | {acc_test_opt:.4f} |")


print("\nMatrice de Confusion (Défaut) :")
y_pred_def = ada_default.predict(X_test)
conf_matrix_def = confusion_matrix(y_test, y_pred_def)

print(conf_matrix_def)

