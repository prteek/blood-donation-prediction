import os
import pandas as pd
from sagemaker import Session
from sklearn.model_selection import train_test_split
import argparse
from dotenv import load_dotenv
load_dotenv('local_credentials.env')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', default='.', help='s3 bucket name or local folder name where train and test dataframes will be saved e.g. "s3://blood-donation-prediction" or "./data"')
    
    args = parser.parse_args()
    
    bucket_ = args.bucket
    if bucket_.startswith('s3://'): bucket = bucket_.split('s3://')[1]
    else: bucket = bucket_
    
    
    df = pd.read_csv("./transfusion.data")

    predictors = ['Recency (months)', 'Time (months)', 'Frequency (times)', 'Monetary (c.c. blood)']

    target = 'whether he/she donated blood in March 2007'

    X = df[predictors]
    y = df[[target]] # to make sure y is a dataframe
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                                        test_size = 0.2, 
                                                        stratify=y, 
                                                        random_state=42)

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    
    train_file = 'train.parquet'
    test_file = 'test.parquet'
    
    if bucket_.startswith("s3://"):
        df_train.to_parquet(train_file)
        df_test.to_parquet(test_file)

        sess = Session()
        data_dir = sess.upload_data('transfusion.data', bucket=bucket, key_prefix='data')

        train_dir = sess.upload_data(train_file,
                                    bucket=bucket, key_prefix='data/train')

        print(f'training-data saved at:{train_dir}')

        test_dir = sess.upload_data(test_file,
                                    bucket=bucket, key_prefix='data/test')
        print(f'test-data saved at:{test_dir}')

        os.system(f"rm {train_file} {test_file}")
    
    else:
        df_train.to_parquet(os.path.join(bucket, train_file))
        df_test.to_parquet(os.path.join(bucket, test_file))
        
        