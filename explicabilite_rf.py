import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import lime
import lime.lime_tabular
import shap

try:
    X = pd.read_csv('X_processed.csv')
    y = pd.read_csv('y_labels.csv').iloc[:, 0]
except:
    print("Erreur : Fichiers manquants.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Ré-entraînement du Random Forest Optimisé...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# PERMUTATION FEATURE IMPORTANCE 
print("\n--- 3.1 Permutation Importance ---")

X_test_sample = X_test.sample(n=2000, random_state=42)
y_test_sample = y_test.loc[X_test_sample.index]

result = permutation_importance(
    rf_model, X_test_sample, y_test_sample, n_repeats=5, random_state=42, n_jobs=-1
)

perm_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': result.importances_mean  
})

perm_df = perm_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=perm_df, palette='viridis')

plt.title("Importance des Variables par Permutation (Top 10)")
plt.xlabel("Baisse de l'Accuracy (Moyenne)")
plt.ylabel("Attributs")
plt.tight_layout()
plt.savefig('explicabilite_1_permutation.png')
print("Image 'explicabilite_1_permutation.png' (Barplot) sauvegardée.")

# LIME (Explication Locale)
print("\n--- 3.2.1 LIME (Local Interpretation) ---")

idx_individu = 10
individu = X_test.iloc[idx_individu]
vrai_label = y_test.iloc[idx_individu]
print(f"Individu étudié (Index {idx_individu}) - Vrai Label : {vrai_label}")

explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['<=50k', '>50k'],
    mode='classification'
)

exp = explainer_lime.explain_instance(
    data_row=individu, 
    predict_fn=rf_model.predict_proba,
    num_features=10
)

plt.figure(figsize=(10, 6))
exp.as_pyplot_figure()
plt.title(f"Explication LIME (Individu {idx_individu})")
plt.tight_layout()
plt.savefig('explicabilite_2_lime.png')
print("Image 'explicabilite_2_lime.png' sauvegardée.")

#  SHAP (Global & Local)
print("\n--- 3.2.2 SHAP (Global & Local) ---")

explainer_shap = shap.TreeExplainer(rf_model)

X_shap_sample = X_test.sample(n=500, random_state=42)
shap_values = explainer_shap.shap_values(X_shap_sample)

vals_class_1 = shap_values[1]

shap_values_indiv = explainer_shap.shap_values(X_test.iloc[[idx_individu]])
print("Génération Waterfall Plot...")
plt.figure()
shap.plots.waterfall(
    shap.Explanation(values=shap_values_indiv[1][0], 
                     base_values=explainer_shap.expected_value[1], 
                     data=X_test.iloc[idx_individu], 
                     feature_names=X_test.columns),
    show=False
)
plt.savefig('explicabilite_3_shap_waterfall.png', bbox_inches='tight')
print("Image 'explicabilite_3_shap_waterfall.png' sauvegardée.")

print("Génération Summary Plot...")
plt.figure()
shap.summary_plot(vals_class_1, X_shap_sample, show=False)
plt.savefig('explicabilite_4_shap_summary.png', bbox_inches='tight')
print("Image 'explicabilite_4_shap_summary.png' sauvegardée.")

print("\nTerminé ! Toutes les images d'explicabilité sont prêtes.")