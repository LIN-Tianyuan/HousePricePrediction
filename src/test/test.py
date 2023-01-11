from src.models import train_model
from src.data import make_dataset

df_train, df_test = make_dataset.read_data()

model_rf = train_model.get_model1()
# Appliquer à un ensemble de tests réels
predictions_rf = model_rf.predict(df_test)
df_test['SalePrice'] = predictions_rf
submission_rf = df_test['SalePrice'].to_frame()
submission_rf.to_csv('submission_rf.csv')