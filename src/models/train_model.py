from fancyimpute import IterativeImputer as MICE

# Entraîner le modèle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from src.features import build_features

X_train, X_test, y_train, y_test = build_features.getdata()
categorical_features, numerical_features = build_features.getfeatures()

# Prétraitement des colonnes numériques,
# mise à l'échelle (normalisation des caractéristiques à un certain intervalle de valeurs)
# et imputation des valeurs manquantes (interpolation des valeurs manquantes dans les données par l'interpolateur).
numerical_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
    ('mice_imputer', MICE())])

# Prétraitement des variables catégorielles,
# leur codage (remplacement des caractéristiques catégorielles
# par des probabilités postérieures correspondant aux valeurs cibles)
# et imputation des valeurs manquantes.
categorical_transformer = Pipeline(steps=[
    ('target_encoder', ce.TargetEncoder(handle_unknown='ignore')),
    ('mice_imputer', MICE())])

# Consolidation des étapes de prétraitement
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)],
    )

# Créer un pipeline de prétraitement et de formation
# Formation sur les pipelines
# Convertisseurs : convertir les données dans le format requis pour les modèles d'apprentissage automatique Estimateurs :
# modèles d'apprentissage automatique (algorithme de forêt aléatoire)
pipeline1 = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestRegressor())])


pipeline2 = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor',DecisionTreeRegressor())])

pipeline3 = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor',GradientBoostingRegressor())])

# Random forest regression
def get_model1():
    # ajuster le pipeline pour entraîner un modèle de régression linéaire sur l'ensemble d'entraînement
    model_rf = pipeline1.fit(X_train, y_train)
    return model_rf

# Decision tree regression
def get_model2():
    # ajuster le pipeline pour entraîner un modèle de régression linéaire sur l'ensemble d'entraînement
    model_rf = pipeline2.fit(X_train, y_train)
    return model_rf

# Gradient boosting regression
def get_model3():
    # ajuster le pipeline pour entraîner un modèle de régression linéaire sur l'ensemble d'entraînement
    model_rf = pipeline2.fit(X_train, y_train)
    return model_rf