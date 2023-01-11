import pandas as pd
from src.data import make_dataset

df_train, df_test = make_dataset.read_data()
# print(df_train.head())

def eda(df):
    if isinstance(df, pd.DataFrame):
        # Toutes les valeurs manquantes
        total_na = df.isna().sum().sum()
        # Lignes, colonnes
        print("Dimension : %d Lignes, %d Colonnes" % (df.shape[0], df.shape[1]))
        # Toutes les valeurs manquantes
        print("Total NA valeurs : %d " % (total_na))
        print("%38s %10s   %10s %10s %8s" % ("Colonne", "Type de donnée", "Modalité", "NA valeurs", "isNaN"))
        # Nom de l'étiquette
        col_name = df.columns
        # Type de données
        dtyp = df.dtypes
        # Nombre de non-duplicatas
        uniq = df.nunique()
        # Nombre de valeurs manquantes par colonne
        na_val = df.isna().sum()
        # Pourcentage manquant par colonne
        percent_na = round((df.isna().sum())/len(df)*100, 2)
        for i in range(len(df.columns)):
            print("%38s %10s     %10s %10s %10s" % (col_name[i], dtyp[i], uniq[i], na_val[i], percent_na[i]))

eda(df_train)


# Supprimer les colonnes présentant un pourcentage élevé de valeurs manquantes (>30%)

# Création d'une liste contenant les colonnes avec un pourcentage élevé de valeurs manquantes
def drop_nan_features(df):
    col_name = df.columns
    na_val = df.isna().sum()
    percent_na = round((df.isna().sum()) / len(df) * 100, 2)
    to_drop = []
    for i in range(len(df.columns)):
        if percent_na[i] > 30:
            to_drop.append(col_name[i])

    return to_drop

high_nan_features = drop_nan_features(df_train)

# Suppression des colonnes avec un pourcentage élevé de valeurs manquantes
def drop_columns(df):
    return df.drop(columns=high_nan_features, inplace=True)

# drop_columns(df_train)

# Extraction des variables catégorielles et continues
def extract_cat_num(df):
    # Extrait les colonnes qui sont de type objet
    categorical=[col for col in df.columns if df[col].dtype=='object']
    # Extraction des colonnes qui ne sont pas de type objet
    numerical=[col for col in df.columns if df[col].dtype!='object']
    return categorical,numerical

categorical,numerical=extract_cat_num(df_train)

from sklearn.model_selection import train_test_split
def getdata():
    # X Toutes les caractéristiques
    # y Prix des logements
    X, y = df_train.iloc[:,:-1], df_train.iloc[:,-1]
    # 20% pour les ensemble d'essais
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def getfeatures():
    categorical_features = categorical
    numerical_features = numerical
    numerical_features.remove('SalePrice')
    return categorical_features, numerical_features

