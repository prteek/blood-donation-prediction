# Blood donation prediction
Predictive modelling on dataset for likelihood of blood donation

This repo is an exercise in using both Cloud tools to quickly iterate and prototype on ML problem and well as demonstrating a logical approach to ML problem solving

### Setup 
Make sure you're using virtual environment

```shell
python3 -m pip install -r requirements.txt
```

Create a ```local_credentials.env``` file (do not share it or add to git or push to ECR) which should look something like below but with your own credentials:

---
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE  
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYzEXAMPLEKEY  
AWS_DEFAULT_REGION=eu-west-1  
SAGEMAKER_EXECUTION_ROLE=arn:aws:iam::432123456789:role/service-role/AmazonSageMaker-ExecutionRole-20210330T123456)

---
    
    (Optionally for above you can setup AWS credentials in cli)

Although the process can be accomplished locally it would be best to use AWS for data storage and training.
Setup on AWS requires:
1. An appropriate Role which can read/write to s3 and launch training/hpo jobs on sagemaker
2. An s3 bucket exclusively for the project


### Prepare training and test data
There is a helper ```data_prep.py``` which can help create train and test set data from ```transfusion.data``` on either s3 or locally

#### Usage
For s3
``` shell
python data_prep.py --bucket s3://bucket-name --test-size 0.2
```

Raw data is placed in the directory ```data``` in the bucket and train and test data in corresponding folders within ```data```


For local

```shell
python data_prep.py --bucket "." --test-size 0.2
```



### Train a model
There are many model specific files that can generate ML model by training either locally or in Sagemaker. The helper ```train``` can be used to start training using any model file

#### Train help 
```shell
./train -h
```

#### Usage
Train locally with:
```shell
./train --model-file rule_based_model --instance-type local
```

Train on sagemaker with:
```shell
./train --model-file rule_based_model --instance-type ml.m5.large --wait True
```

