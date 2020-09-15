import pandas as pd
from sklearn.utils import shuffle
from imblearn.under_sampling import RandomUnderSampler


import datetime
import random
from datetime import timedelta

from sklearn.preprocessing import StandardScaler, RobustScaler

from core.composer.gp_composer.gp_composer import \
    GPComposer, GPComposerRequirements
from core.composer.visualisation import ComposerVisualiser
from core.repository.model_types_repository import ModelTypesRepository
from core.repository.quality_metrics_repository import \
    ClassificationMetricsEnum, MetricsRepository
from core.repository.tasks import Task, TaskTypesEnum
from core.utils import probs_to_labels
from examples.utils import create_multi_clf_examples_from_excel


import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix
from benchmark.benchmark_utils import get_scoring_case_data_paths
from core.composer.chain import Chain
from core.composer.node import PrimaryNode, SecondaryNode
from core.models.data import InputData

random.seed(1)
np.random.seed(1)



def get_model(train_file_path: str, cur_lead_time: datetime.timedelta = timedelta(minutes=5)):
    task = Task(task_type=TaskTypesEnum.classification)
    dataset_to_compose = InputData.from_csv(train_file_path, task=task)

    # the search of the models provided by the framework
    # that can be used as nodes in a chain for the selected task
    models_repo = ModelTypesRepository()
    available_model_types, _ = models_repo.suitable_model(task_type=task.task_type)

    metric_function = MetricsRepository(). \
        metric_by_id(ClassificationMetricsEnum.ROCAUC_penalty)

    composer_requirements = GPComposerRequirements(
        primary=available_model_types, secondary=available_model_types,
        max_lead_time=cur_lead_time, max_arity=3,
        max_depth=4, pop_size=20, num_of_generations=100, 
        crossover_prob = 0.8, mutation_prob = 0.8, 
        add_single_model_chains = True)

    # Create the genetic programming-based composer, that allow to find
    # the optimal structure of the composite model
    composer = GPComposer()

    # run the search of best suitable model
    chain_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                                initial_chain=None,
                                                composer_requirements=composer_requirements,
                                                metrics=metric_function, is_visualise=False)
    
    chain_evo_composed.fit(input_data=dataset_to_compose)

    return chain_evo_composed


def validate_model_quality(model: Chain, data_path: str):
    dataset_to_validate = InputData.from_csv(data_path)
    predicted_labels = model.predict(dataset_to_validate).predict

    
    roc_auc_st = roc_auc(y_true=test_data.target,y_score=predicted_labels)
                              
    p = precision_score(y_true=test_data.target,y_pred=predicted_labels.round())
    r = recall_score(y_true=test_data.target, y_pred=predicted_labels.round())
    a = accuracy_score(y_true=test_data.target, y_pred=predicted_labels.round())
    
    return roc_auc_st, p, r, a



def balance_class(file_path):
    df = pd.read_csv(file_path)
    
    X = df.drop(columns=['Class'])
    y = df.iloc[:,[-1]]

    rus = RandomUnderSampler(sampling_strategy = 'all', random_state=42)
    
    X_res, y_res = rus.fit_resample(X, y)
    X_res['Class'] = y_res
    
    df_balanced = shuffle(X_res, random_state = 42).reset_index().drop(columns='index')
    
    df_balanced.to_csv(r'./creditcard_overSample.csv', index=False)
    
    return r'./creditcard_overSample.csv'

if __name__ == "__main__":
    file_path = r'./creditcard.csv'
    
    file_path_first = balance_class(file_path)
    
    train_file_path, test_file_path = create_multi_clf_examples_from_excel(file_path_first)
    test_data = InputData.from_csv(test_file_path)
    
    fitted_model = get_model(train_file_path)
    
    ComposerVisualiser.visualise(fitted_model, save_path = f'./model_done.jpg')
    
    roc_auc, p, r, a = validate_model_quality(fitted_model, test_file_path)
    print(f'ROC AUC metric is {roc_auc}, \nPRECISION is {p}, \nRECALL is {r}, \nACCURACY is {a}')