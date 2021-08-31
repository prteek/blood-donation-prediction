import numpy as np
import pandas as pd
import argparse
import os
import json
from sklearn.model_selection import cross_val_score
import subprocess, sys


def install(package):
    subprocess.call([sys.executable, "-m","pip", "install", package])


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, 'model.mdl'))
    return model


def simple_rules(df, min_recency=20, min_time=70):
    ruler = CaseWhenRuler(default=0)
    ruler.add_rule(lambda d: d['Recency (months)'] <= min_recency, 1, name='recency')
    ruler.add_rule(lambda d: d['Time (months)'] <= min_time, 1, name='time')
    return ruler.predict(df)


if __name__ == '__main__':
    
    install('ipython')
    install('human-learn')
    from hulearn.classification import FunctionClassifier
    from hulearn.experimental import CaseWhenRuler

    parser = argparse.ArgumentParser()
    parser.add_argument('--training', default='/opt/ml/input/data/training')
    parser.add_argument('--model-dir', default='/opt/ml/model')
    parser.add_argument('--min_recency', type=int, default=5)
    parser.add_argument('--min_time', type=int, default=15)
    
    args = parser.parse_args()
    
    training_dir = args.training
    model_dir = args.model_dir

    # Hyperparameters
    min_recency = args.min_recency
    min_time = args.min_time
    
    
    df = pd.read_parquet(os.path.join(training_dir, 'train.parquet'))

    predictors = ['Recency (months)', 'Time (months)', 'Frequency (times)', 'Monetary (c.c. blood)']

    target = 'whether he/she donated blood in March 2007'

    X = df[predictors]
    y = df[target]
    
    clf = FunctionClassifier(simple_rules, 
                             min_recency=min_recency, 
                             min_time=min_time)
    
    f1_score = cross_val_score(clf, X, y, scoring='f1', cv=4, n_jobs=2)
    print(f"f1_score={np.mean(f1_score)};")
    
    precision = cross_val_score(clf, X, y, scoring='precision', cv=4, n_jobs=2)
    print(f"precision={np.mean(precision)};")
    
    recall = cross_val_score(clf, X, y, scoring='recall', cv=4, n_jobs=2)
    print(f"recall={np.mean(recall)};")

    