import matplotlib.pyplot as plt
import numpy as np
from src.features import build_features
from src.models import predict_model
from src.models.train_model import pipeline1

X_train, X_test, y_train, y_test = build_features.getdata()
predictions_rf = predict_model.predictions_rf1

# Plot predicted vs actual
plt.scatter(y_test, predictions_rf)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Sale Price Predictions')
z = np.polyfit(y_test, predictions_rf, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()

# On identifie les variables les plus pertinentes dans notre modèle
feature_importance = pipeline1.steps[1][1].feature_importances_
# Toutes les colonnes sont triées et la plus triée est la variable la plus pertinente.
sorted_idx = pipeline1.steps[1][1].feature_importances_.argsort()
print(sorted_idx)
plt.figure(figsize=(40,20))
plt.barh(X_train.columns[sorted_idx], feature_importance[sorted_idx])