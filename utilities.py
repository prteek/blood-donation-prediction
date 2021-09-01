import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
import joblib
import tarfile

    
def load_model_from_tarfile(file_path):
    t = tarfile.open("model.tar.gz", "r")
    for filename in t.getnames():
        f = t.extractfile(filename)
        model = joblib.load(f)
    
    return model
    
    
