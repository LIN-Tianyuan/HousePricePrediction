from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from src.models import train_model
from src.features import build_features

X_train, X_test, y_train, y_test = build_features.getdata()

model_rf1 = train_model.get_model1()
# Random forest regression
predictions_rf1 = model_rf1.predict(X_test)

model_rf2 = train_model.get_model2()
# Decision tree regression
predictions_rf2 = model_rf2.predict(X_test)

model_rf3 = train_model.get_model3()
# Gradient boosting regression
predictions_rf3 = model_rf3.predict(X_test)

# Afficher les métriques
# Erreur quadratique moyenne (vraie, prédite)
mse = mean_squared_error(y_test, predictions_rf1)
# Racine carrée de l'erreur quadratique moyenne
rmse = np.sqrt(mse)
print("Random forest regression: ")
print("RMSE:", rmse)
# Erreur absolue moyenne
mae = mean_absolute_error(y_test, predictions_rf1)
print("MAE:", mae)
# R2 coefficient de détermination (qualité de l'ajustement)
r2 = r2_score(y_test, predictions_rf1)
print("R2:", r2)
print("----------------------------")
# Afficher les métriques
# Erreur quadratique moyenne (vraie, prédite)
mse = mean_squared_error(y_test, predictions_rf2)
# Racine carrée de l'erreur quadratique moyenne
rmse = np.sqrt(mse)
print("Decision tree regression: ")
print("RMSE:", rmse)
# Erreur absolue moyenne
mae = mean_absolute_error(y_test, predictions_rf2)
print("MAE:", mae)
# R2 coefficient de détermination (qualité de l'ajustement)
r2 = r2_score(y_test, predictions_rf2)
print("R2:", r2)
print("----------------------------")
# Afficher les métriques
# Erreur quadratique moyenne (vraie, prédite)
mse = mean_squared_error(y_test, predictions_rf3)
# Racine carrée de l'erreur quadratique moyenne
rmse = np.sqrt(mse)
print("Gradient boosting regression: ")
print("RMSE:", rmse)
# Erreur absolue moyenne
mae = mean_absolute_error(y_test, predictions_rf3)
print("MAE:", mae)
# R2 coefficient de détermination (qualité de l'ajustement)
r2 = r2_score(y_test, predictions_rf3)
print("R2:", r2)

