import numpy as np
import pandas as pd
import argparse
import os
import json
from sklearn.model_selection import cross_val_score
import joblib

def install(package):
    os.system(f"python -m pip install {package}")
    return None


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, 'model.mdl'))
    return model


def simple_rules(df, recency_threshold=20, time_threshold=70):
    ruler = CaseWhenRuler(default=0)
    ruler.add_rule(lambda d: d['Recency (months)'] <= recency_threshold, 1, name='recency')
    ruler.add_rule(lambda d: d['Time (months)'] <= time_threshold, 1, name='time')
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
    parser.add_argument('--recency_threshold', type=int, default=5)
    parser.add_argument('--time_threshold', type=int, default=15)
    
    args = parser.parse_args()
    
    training_dir = args.training
    model_dir = args.model_dir

    # Hyperparameters
    recency_threshold = args.recency_threshold
    time_threshold = args.time_threshold
    
    
    df = pd.read_parquet(os.path.join(training_dir, 'train.parquet'))

    predictors = ['Recency (months)', 'Time (months)', 'Frequency (times)', 'Monetary (c.c. blood)']

    target = 'whether he/she donated blood in March 2007'

    X = df[predictors]
    y = df[target]
    
    clf = FunctionClassifier(simple_rules, 
                             recency_threshold=recency_threshold, 
                             time_threshold=time_threshold)
        
    f1_score = cross_val_score(clf, X, y, scoring='f1', cv=4, n_jobs=2)
    print(f"f1_score={np.mean(f1_score)}")
    
    precision = cross_val_score(clf, X, y, scoring='precision', cv=4, n_jobs=2)
    print(f"precision={np.mean(precision)}")
    
    recall = cross_val_score(clf, X, y, scoring='recall', cv=4, n_jobs=2)
    print(f"recall={np.mean(recall)}")

    