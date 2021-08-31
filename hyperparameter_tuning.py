import os
import sagemaker
import argparse
from sagemaker.sklearn import SKLearn
from sagemaker.tuner import HyperparameterTuner, IntegerParameter
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from smexperiments.search_expression import Filter, Operator, SearchExpression
from smexperiments.trial_component import TrialComponent
from sagemaker.analytics import HyperparameterTuningJobAnalytics
from botocore.exceptions import ClientError, ParamValidationError
from datetime import datetime
import time
from dotenv import load_dotenv
load_dotenv('local_credentials.env')


role = os.environ['SAGEMAKER_EXECUTION_ROLE']

def attach_tuner_outputs_to_trial(tuner, trial):
    """Attach trial components generated by a tuner to trial"""
    
    try:
        tuner_job  = tuner.describe()
    except ParamValidationError:
        raise Exception("The tuner job has not finished (or started yet), please try after the job has run")
        
    # the training job names contain the tuning job name (and the training job name is in the source arn)
    source_arn_filter = Filter(
        name="TrialComponentName", operator=Operator.CONTAINS, value=tuner_job["HyperParameterTuningJobName"]
    )

    search_expression = SearchExpression(
        filters=[source_arn_filter]
    )

    # search iterates over every page of results by default
    trial_component_search_results = list(
        TrialComponent.search(search_expression=search_expression)
    )
    
    print(f"Found {len(trial_component_search_results)} trial components.")
    
    # associate the trial components with the trial
    for tc in trial_component_search_results:
        print(f"Associating trial component {tc.trial_component_name} with trial {trial.trial_name}.")
        trial.add_trial_component(tc.trial_component_name)
        # sleep to avoid throttling
        time.sleep(0.5)
        
    return None




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', default='blood-donation')
    parser.add_argument('--trial-name-prefix', default='blood-donation')

    args = parser.parse_args()
    
    experiment_name = args.experiment_name
    trial_name_prefix = args.trial_name_prefix
    
    timestamp = datetime.now().strftime('%Y-%m-%dT%H%M')
    trial_name = f"{trial_name_prefix}--{timestamp}"
    
    bucket = 'blood-donation-prediction'
    
    try:
        experiment = Experiment.create(experiment_name=experiment_name)
    except:
        experiment = Experiment.load(experiment_name=experiment_name)
        
    trial = Trial.create(experiment.experiment_name, trial_name=trial_name)
    
    output_path = f"s3://{bucket}/model/artifacts"
    training_dir = f"s3://{bucket}/data/train"
#     hyperparams = {'min_recency':15, 'min_time':60}
    
    estimator = SKLearn('train.py',
                       role=role,
                       framework_version='0.23-1',
                       output_path=output_path,
                       instance_count=1,
                       instance_type='ml.m5.large',
                       use_spot_instances=True,
                       max_wait=1000,
                       max_run=300,
                       base_job_name='training'
                       )
    
    
    metric_definitions = [{'Name': 'f1_score', 'Regex': "f1_score=([0-9\\.]+)"},
                        {'Name': 'precision', 'Regex': "precision=([0-9\\.]+)"},
                        {'Name': 'recall', 'Regex': "recall=([0-9\\.]+)"}]
    
    
    hyperparams = {'min_recency':IntegerParameter(1,60, scaling_type='Logarithmic'),
                  'min_time':IntegerParameter(1,100, scaling_type='Logarithmic')}
    tuner = HyperparameterTuner(estimator, 
                                'f1_score', 
                                hyperparameter_ranges=hyperparams,
                                metric_definitions=metric_definitions,
                                max_jobs=20,
                                max_parallel_jobs=4,
                                base_tuning_job_name='tuning'
                               )
    
    tuner.fit(inputs={'training':training_dir}, 
              job_name=trial_name) # Same channel name as expected in train.py

    
    # Attach tuning job to an experiment    
    attach_tuner_outputs_to_trial(tuner, trial)
    
    