import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
import lime
import lime.lime_tabular
import shap


try:
    X = pd.read_csv('X_processed.csv')
    y = pd.read_csv('y_labels.csv').iloc[:, 0]
    X.columns = [c.replace('[', '').replace(']', '').replace('<', '') for c in X.columns]
except:
    print("Erreur : Fichiers manquants.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Entraînement du XGBoost Optimisé...")
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

#  PERMUTATION IMPORTANCE (Barplot Classique)
X_test_sample = X_test.sample(n=2000, random_state=42)
y_test_sample = y_test.loc[X_test_sample.index]

result = permutation_importance(
    xgb_model, X_test_sample, y_test_sample, n_repeats=5, random_state=42, n_jobs=-1
)

perm_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': result.importances_mean
})
perm_df = perm_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=perm_df, palette='viridis')
plt.title("Permutation Importance - XGBoost (Top 10)")
plt.xlabel("Baisse de l'Accuracy (Moyenne)")
plt.tight_layout()
plt.savefig('explicabilite_xgb_1_permutation.png')
print("Image 'explicabilite_xgb_1_permutation.png' sauvegardée.")


#  LIME (Explication Locale)

idx_individu = 10 # choix lde la personne n 10
individu = X_test.iloc[idx_individu]
print(f"Individu étudié (Index {idx_individu}) - Vraie classe : {y_test.iloc[idx_individu]}")

explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['<=50k', '>50k'],
    mode='classification'
)

exp = explainer_lime.explain_instance(
    data_row=individu, 
    predict_fn=xgb_model.predict_proba,
    num_features=10
)

plt.figure(figsize=(10, 6))
exp.as_pyplot_figure()
plt.title(f"Explication LIME (XGBoost - Individu {idx_individu})")
plt.tight_layout()
plt.savefig('explicabilite_xgb_2_lime.png')
print("Image 'explicabilite_xgb_2_lime.png' sauvegardée.")


#  SHAP (Global & Local)

explainer_shap = shap.TreeExplainer(xgb_model)

X_shap_sample = X_test.sample(n=500, random_state=42)
shap_values = explainer_shap.shap_values(X_shap_sample)

shap_values_indiv = explainer_shap(X_test.iloc[[idx_individu]])

print("Génération Waterfall Plot...")
plt.figure()
shap.plots.waterfall(shap_values_indiv[0], show=False)
plt.savefig('explicabilite_xgb_3_waterfall.png', bbox_inches='tight')
print("Image 'explicabilite_xgb_3_waterfall.png' sauvegardée.")

print("Génération Summary Plot...")
plt.figure()
shap.summary_plot(shap_values, X_shap_sample, show=False)
plt.savefig('explicabilite_xgb_4_summary.png', bbox_inches='tight')
print("Image 'explicabilite_xgb_4_summary.png' sauvegardée.")

