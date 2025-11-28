import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import time

start_time = time.time()

# --- 1. Chargement des Données ---
try:
    features_df = pd.read_csv('alt_acsincome_ca_features_85.csv')
    labels_df = pd.read_csv('alt_acsincome_ca_labels_85.csv')
    print("Fichiers chargés.")
except FileNotFoundError:
    print("ERREUR: Fichiers CSV non trouvés. Vérifiez qu'ils sont dans le bon dossier.")
    exit()

# --- 2. Feature Engineering (Regroupement) ---
print("Application des regroupements (feature engineering)...")

# === Regroupement SCHL (Niveau d'études) ===
schl_bins = [0, 16, 20, 24]
schl_labels = ['Etudes_Basses', 'Etudes_Moyennes', 'Etudes_Hautes']
features_df['SCHL_Group'] = pd.cut(features_df['SCHL'], bins=schl_bins, labels=schl_labels, right=True)

# === Regroupement POBP (Lieu de naissance) ===
features_df['POBP_Group'] = np.where(features_df['POBP'] <= 56, 'Ne_USA', 'Ne_Etranger')



# --- 3. Définition des colonnes pour le Pre-processing ---

# Colonnes numériques (vrais chiffres)
numerical_cols = ['AGEP', 'WKHP']

# Colonnes catégorielles
# On prend nos NOUVEAUX groupes + les colonnes simples (SANS OCCP)
categorical_cols = [
    'SCHL_Group', 'POBP_Group', # Nos nouveaux groupes
    'COW', 'MAR', 'RELP', 'SEX', 'RAC1P'       # Les colonnes simples
]

# On supprime les colonnes originales qu'on a regroupées ET OCCP
features_df = features_df.drop(columns=['SCHL', 'OCCP', 'POBP'])

print("Regroupements et suppression de OCCP terminés.")

# --- 4. Création du "Preprocessor" (Le transformateur) ---
print("Configuration du preprocessor (OneHotEncoder + StandardScaler)...")

preprocessor = ColumnTransformer(
    transformers=[
        # Transformer 1: Pour les colonnes numériques
        ('num', StandardScaler(), numerical_cols),
        
        # Transformer 2: Pour les colonnes catégorielles
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

# --- 5. Application de la Transformation ---
print("Application de la transformation...")

X_processed = preprocessor.fit_transform(features_df)

print("Transformation terminée.")

# --- 6. Récupération des Noms de Colonnes ---
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_feature_names = list(numerical_cols) + list(cat_feature_names)

# Convertir le résultat (un array numpy) en DataFrame pandas
X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)

print(f"Dimensions des données AVANT transformation : {features_df.shape}")
print(f"Dimensions des données APRÈS transformation : {X_processed_df.shape}")
print("\nAperçu des données transformées :")
print(X_processed_df.head())

# --- 7. Sauvegarde des Fichiers "Propres" ---
print("Sauvegarde des fichiers 'X_processed.csv' et 'y_labels.csv'...")

X_processed_df.to_csv('X_processed.csv', index=False)
labels_df.to_csv('y_labels.csv', index=False) # On sauvegarde aussi les labels

end_time = time.time()
print(f"\nTerminé ! Pré-traitement effectué en {end_time - start_time:.2f} secondes.")
print("Vous avez maintenant 'X_processed.csv' et 'y_labels.csv' prêts pour l'entraînement.")