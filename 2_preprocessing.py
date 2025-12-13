import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

print("Début du nettoyage...")

# 1. Chargement
X = pd.read_csv('alt_acsincome_ca_features_85.csv')
y = pd.read_csv('alt_acsincome_ca_labels_85.csv')

# 2. Feature Engineering (Nos choix intelligents)

# A. Regroupement du Niveau Scolaire (SCHL) en 3 groupes
# 0-16: Basses, 17-20: Moyennes, 21-24: Hautes
X['SCHL_Group'] = pd.cut(X['SCHL'], bins=[0, 16, 20, 24], labels=['Bas', 'Moyen', 'Haut'], right=True)

# B. Regroupement du Lieu de Naissance (POBP)
# <=56: USA, >56: Etranger
X['POBP_Group'] = np.where(X['POBP'] <= 56, 'USA', 'Etranger')

# C. Suppression des colonnes inutiles ou trop complexes
# On enlève OCCP (trop complexe) et les anciennes colonnes SCHL/POBP
X = X.drop(columns=['OCCP', 'SCHL', 'POBP'])

print("Colonnes regroupées et nettoyées.")

# 3. Pipeline de Transformation (La machine à transformer)
# On définit quelles colonnes sont des chiffres et lesquelles sont des catégories
numerical_cols = ['AGEP', 'WKHP']
categorical_cols = ['SCHL_Group', 'POBP_Group', 'COW', 'MAR', 'RELP', 'SEX', 'RAC1P']

preprocessor = ColumnTransformer(
    transformers=[
        # Pour les chiffres : on standardise (moyenne 0, écart-type 1)
        ('num', StandardScaler(), numerical_cols),
        # Pour les catégories : on encode en 0 et 1 (OneHot)
        ('cat', OneHotEncoder(sparse_output=False), categorical_cols)
    ]
)

# 4. Exécution
print("Transformation en cours (encodage + standardisation)...")
X_clean = preprocessor.fit_transform(X)

# On remet les noms des colonnes pour que ce soit propre
feature_names = (numerical_cols + 
                 list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)))
X_final = pd.DataFrame(X_clean, columns=feature_names)

# 5. Sauvegarde
X_final.to_csv('X_clean.csv', index=False)
y.to_csv('y_clean.csv', index=False)

print("Terminé ! Fichiers 'X_clean.csv' et 'y_clean.csv' créés.")
print(f"Taille finale : {X_final.shape} (Lignes, Colonnes)")