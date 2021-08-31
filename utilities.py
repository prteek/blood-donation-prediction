import numpy as np
import pandas as pd
import altair as alt
from sklearn.base import TransformerMixin, BaseEstimator


class CleanUpDataFrame(BaseEstimator, TransformerMixin):
    """Transformer to clean up data"""
    def __init__(self) -> None:
        return None
    
    def fit(self, df:pd.DataFrame()):
        self.df = df.copy()
        return self
    
    def transform(self, df) -> pd.DataFrame:
        """Remove problematic column and create a new one"""
        self.df = df.copy()
        self.df['Monetary (ml)'] = df['Monetary (c.c. blood)']*1.0
        self.df.drop(['Monetary (c.c. blood)'], axis=1,inplace=True)
        return self.df.values

    
    

def plot_data(df:pd.DataFrame)-> alt.Chart:
    scatter_matrix = (alt.Chart(df)
     .mark_circle()
     .encode(
        x=alt.X(alt.repeat("column"), type='quantitative'),
        y=alt.Y(alt.repeat("row"), type='quantitative'),
        color='whether he/she donated blood in March 2007:N')
     .properties(
        width=150,
        height=150)
     .repeat(
        row=df.columns.to_list(),
        column=df.columns.to_list())
     .interactive()
    )

    histograms = (alt.Chart(df)
     .transform_fold(df.columns.to_list(), as_=['column', 'value'])
     .mark_bar()
     .encode(x='value:Q',
             y='count():Q',
             column='column:N')
     .interactive()
    )
    
    return histograms, scatter_matrix




class SqrtTransform(BaseEstimator, TransformerMixin):
    def __init__(self, transform_columns:list=None) -> None:
        self.transform_columns = transform_columns
        return None
    def fit(self, X_train, y_train=None):
        return self
    def transform(self, X_train, y_train=None):
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
        else: pass
        
        if self.transform_columns is None:
            columns_to_transform = np.arange(X_train.shape[1])
        elif isinstance(self.transform_columns, int):
            columns_to_transform =  [self.transform_columns]
        else:
            columns_to_transform =  self.transform_columns
            
        X_train[:,columns_to_transform] = np.sqrt(X_train[:,columns_to_transform])
        return X_train


    
    
    
    
