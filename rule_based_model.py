import numpy as np
import pandas as pd
import argparse
import os
import json
from sklearn.model_selection import RandomizedSearchCV
import joblib
from scipy.stats import uniform, randint
from sklearn.base import BaseEstimator, TransformerMixin

def install(package):
    """Install packages missing from training container image 
    (make sure dependencies do not change or reinstall scipy, numpy or scikit-learn)"""
    os.system(f"python -m pip install {package}")
    return None


def model_fn(model_dir):
    """Required for sagemaker training if needed to deploy the trained model"""
    model = joblib.load(os.path.join(model_dir, 'model.mdl'))
    return model


def rule_based_models(df, recency_threshold=20, 
                      time_threshold=70, 
                      time_threshold_delta=10, 
                      frequency_threshold=15, 
                      use_simple_model=True):
    """Rule based models defined by observing data.
    These can form a good baseline"""
    
    ruler = CaseWhenRuler(default=0)
    
    if use_simple_model: # Rules are combined by OR condition
        ruler.add_rule(lambda d: d['Recency (months)'] <= recency_threshold, 1, name='recency')
        ruler.add_rule(lambda d: (d['Time (months)'] >= time_threshold) & 
                       (d['Time (months)'] <= time_threshold+time_threshold_delta), 1, name='time')
        ruler.add_rule(lambda d: d['Frequency (times)'] >= frequency_threshold,1, name='frequency')
    
    else: # Rules are combined by AND condition
        ruler.add_rule(lambda d: (d['Recency (months)'] <= recency_threshold) & 
                       (d['Time (months)'] >= time_threshold) & 
                       (d['Time (months)'] <= time_threshold+time_threshold_delta) &
                       (d['Frequency (times)'] >= frequency_threshold), 
                       1, name='recency&time&frequency')
        
    return ruler.predict(df)


if __name__ == '__main__':
    
    # Install additional required packages before importing them
    install('ipython')
    install('human-learn')
    from hulearn.classification import FunctionClassifier
    from hulearn.experimental import CaseWhenRuler
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', default='/opt/ml/input/data/training')
    parser.add_argument('--model-dir', default='/opt/ml/model')
    
    args = parser.parse_args()
    
    training_dir = args.training
    model_dir = args.model_dir

    
    df = pd.read_parquet(os.path.join(training_dir, 'train.parquet'))

    predictors = ['Recency (months)', 'Time (months)', 'Frequency (times)', 'Monetary (c.c. blood)']

    target = 'whether he/she donated blood in March 2007'

    X = df[predictors]
    y = df[target]
    
    clf = FunctionClassifier(rule_based_models)
    
    grid = {'time_threshold':randint(low=1, high=100), 
            'time_threshold_delta':randint(low=1, high=50),
            'recency_threshold':randint(low=1, high=80), 
            'frequency_threshold':randint(low=1, high=50),
            'use_simple_model':[True, False]}
    
    metrics = ['f1', 'precision', 'recall']
    
    model = RandomizedSearchCV(clf, grid, 
                               scoring=metrics, 
                               refit='f1', 
                               cv=5, 
                               n_jobs=-1, 
                               verbose=3, 
                               n_iter=4000)
    
    model.fit(X,y)
    
    results = pd.DataFrame(model.cv_results_)
    
    best_row = results.iloc[model.best_index_,:]
    
    print(f"Final results: {best_row}")
    print(f"f1_score={best_row['mean_test_f1']}")
    print(f"precision={best_row['mean_test_precision']}")
    print(f"recall={best_row['mean_test_recall']}")
    
    joblib.dump(model, os.path.join(model_dir, 'model.mdl'))
    