# Project
## House price forecast for Ames, Iowa

Developer: LIN Tianyuan

This project analyzed statistical data on Ames home prices and built an optimization model based on the analysis to evaluate the best selling price for the client

## Files

- data
    - test.csv
    - train.csv
- src (Source code for use in this project )
    - data (Scripts to download or generate data)
        - make_dataset.py
    - features (Scripts to turn raw data into features for modeling)
        - build_features.py
    - models (Scripts to train models and then use trained models to make redictions)
        - predict_model.py
        - train_model.py
    - test (Test actual data)
        - test.py 
    - visualization (Scripts to create exploratory and results oriented visualizations)
        - visualize.py


## Tech

The project uses the following techniques:

- numpy
- sklearn.metrics
- sklearn.compose
- sklearn.pipeline
- sklearn.preprocessing
- category_encoders
- sklearn.ensemble
- sklearn.tree
- fancyimpute


## Utilization

Running test.py will generate submisson_rf.csv, and you can see the predicted house prices

