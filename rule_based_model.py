import numpy as np
import pandas as pd
import argparse
import os
import json
from sklearn.model_selection import RandomizedSearchCV
import joblib
from scipy.stats import uniform

def install(package):
    os.system(f"python -m pip install {package}")
    return None


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, 'model.mdl'))
    return model


def simple_rules(df, recency_threshold=20, time_threshold=70, frequency_threshold=15):
    ruler = CaseWhenRuler(default=0)
    ruler.add_rule(lambda d: d['Recency (months)'] <= recency_threshold, 1, name='recency')
    ruler.add_rule(lambda d: d['Time (months)'] <= time_threshold, 1, name='time')
    ruler.add_rule(lambda d: d['Frequency (times)'] >= frequency_threshold,1, name='frequency')
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
    
    clf = FunctionClassifier(simple_rules)
    
    grid = {'time_threshold':uniform(loc=1, scale=100), 'recency_threshold':uniform(loc=1, scale=80), 'frequency_threshold':uniform(loc=1, scale=50)}
    
    metrics = ['f1', 'precision', 'recall']
    model = RandomizedSearchCV(clf, grid, 
                               scoring=metrics, 
                               refit='f1', 
                               cv=5, 
                               n_jobs=-1, 
                               verbose=3, 
                               n_iter=2000)
    
    model.fit(X,y)
    
    results = pd.DataFrame(model.cv_results_)
    
    for i, row in results.iterrows():
        print(f"f1_score={row['mean_test_f1']}")
        print(f"precision={row['mean_test_precision']}")
        print(f"recall={row['mean_test_recall']}")
        print(f"epoch={i}")


    
    joblib.dump(model, os.path.join(model_dir, 'model.mdl'))
    