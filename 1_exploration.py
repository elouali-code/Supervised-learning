import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Chargement
X = pd.read_csv('alt_acsincome_ca_features_85.csv')
y = pd.read_csv('alt_acsincome_ca_labels_85.csv')
y.rename(columns={'PINCP': 'TARGET'}, inplace=True)
df = pd.concat([X, y], axis=1)

# 2. Analyse Numérique (Âge et Heures de travail)
# On regarde si l'âge et les heures sont liés au revenu
print("Génération de la matrice de corrélation...")
plt.figure(figsize=(8, 6))
sns.heatmap(df[['AGEP', 'WKHP', 'TARGET']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Corrélation : Âge, Heures, Revenu")
plt.savefig('analyse_correlation_numerique.png')

# 3. Analyse Catégorielle (Études, Mariage, Sexe)
# On regarde le % de riches dans chaque catégorie
cols_a_analyser = ['SCHL', 'MAR', 'SEX']
titles = ['Niveau Scolaire', 'Statut Marital', 'Sexe']

for col, title in zip(cols_a_analyser, titles):
    plt.figure(figsize=(10, 6))
    # On calcule la moyenne de la cible (0 ou 1), ce qui donne le % de riches
    sns.barplot(x=col, y='TARGET', data=df, palette='viridis', errorbar=None)
    plt.title(f"Probabilité d'avoir >50k$ selon : {title}")
    plt.ylabel("Probabilité (>50k)")
    plt.xlabel(f"Code {col}")
    plt.savefig(f'analyse_{col}.png')
