import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lime
import lime.lime_tabular

# --- 1. Préparation  ---
print("Chargement et entraînement rapide du modèle...")
try:
    X = pd.read_csv('X_processed.csv')
    y = pd.read_csv('y_labels.csv').iloc[:, 0]
except:
    print("Erreur : Lancez d'abord le script de pré-traitement pour avoir les fichiers CSV.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

#  notre Random Forest (Paramètres par défaut)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("Modèle prêt.")

# ==============================================================================
# APPROCHE 1 : EXPLICATION GLOBALE (Feature Importance) 
# ==============================================================================
print("\n--- 1. Explication Globale : Importance des Features ---")
print("Quels sont les critères les plus importants pour le modèle en général ?")

importances = rf_model.feature_importances_
feature_names = X.columns

# Création d'un tableau  pour l'affichage
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
plt.title('Top 10 des critères influençant le modèle (Global)')
plt.xlabel("Importance (Réduction d'impureté Gini)")
plt.tight_layout()
plt.savefig('explication_globale.png')
print("Graphique sauvegardé : 'explication_globale.png'")


# ==============================================================================
# APPROCHE 2 : EXPLICATION LOCALE (LIME)
# ==============================================================================
print("\n--- 2. Explication Locale : LIME ---")
print("Pourquoi le modèle a-t-il pris cette décision pour UNE personne précise ?")

# 1. Création de l'explainer LIME
# On lui donne les données d'entraînement pour qu'il comprenne les statistiques
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['<=50k', '>50k'],
    mode='classification'
)

# 2. Choix d'une personne à expliquer (prenons la personne n 10 du test set)
idx_personne = 10
#print(X_test[idx_personne]['RELP_Group_nan'])
personne_a_expliquer = X_test.iloc[idx_personne]
vraie_reponse = y_test.iloc[idx_personne]

print(f"\nAnalyse de la personne (Index {idx_personne}) :")
print(f"Vraie réponse : {vraie_reponse} (True = >50k, False = <=50k)")

# 3. Génération de l'explication
# LIME va perturber cette ligne (créer des variantes) pour voir comment le modèle réagit
# num_features=10 : on veut les 10 raisons principales
exp = explainer.explain_instance(
    data_row=personne_a_expliquer, 
    predict_fn=rf_model.predict_proba, 
    num_features=10
)

# 4. Affichage des raisons
print("\nPourquoi le modèle a prédit cela ? (Facteurs positifs vs négatifs)")
# On affiche la liste des règles (ex: AGEP > 0.5) et leur poids
reasons = exp.as_list()
for rule, weight in reasons:
    impact = "Augmente chance >50k" if weight > 0 else "Diminue chance >50k"
    print(f"- {rule} : {weight:.4f} ({impact})")

fig = exp.as_pyplot_figure()
plt.title(f"Explication Locale LIME (Index {idx_personne})")
plt.tight_layout()
plt.savefig('explication_locale_lime.png')
print("\nGraphique sauvegardé : 'explication_locale_lime.png'")