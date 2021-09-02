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

