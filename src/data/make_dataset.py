# Importer d'abord les paquets python requis
import pandas as pd

def read_data():
    # Ensemble d'entraînement de lecture
    df_train = pd.read_csv('../../data/train.csv')
    # Ensemble d'essais de lecture
    df_test = pd.read_csv('../../data/test.csv')

    # Remplacer la valeur ID originale
    df_train.set_index('Id', inplace=True)
    df_test.set_index('Id', inplace=True)

    return df_train, df_test
