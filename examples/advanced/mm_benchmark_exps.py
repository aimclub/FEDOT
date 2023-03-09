import numpy as np
from pathlib import Path
import os
from fedot.api.main import Fedot

from fedot.core.data.multi_modal import MultiModalData
from fedot.core.utils import fedot_project_root
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score

targets = {'product_sentiment_machine_hack': ('Sentiment', 'classification', 'accuracy'),
           'data_scientist_salary': ('salary', 'classification', 'accuracy'),
           'melbourne_airbnb': ('price_label', 'classification', 'accuracy'),
           'news_channel': ('channel', 'classification', 'accuracy'),
           'wine_reviews': ('variety', 'classification', 'accuracy'),
           'imdb_genre_prediction': ('Genre_is_Drama', 'classification', 'roc_auc'),
           'fake_job_postings2': ('fraudulent', 'classification', 'roc_auc'),
           'kick_starter_funding': ('final_status', 'classification', 'roc_auc'),
           'jigsaw_unintended_bias100K': ('target', 'classification', 'roc_auc'),
           'google_qa_answer_type_reason_explanation': ('answer_type_reason_explanation', 'regression', 'r2'),
           'google_qa_question_type_reason_explanation': ('question_type_reason_explanation', 'regression', 'r2'),
           'bookprice_prediction': ('Price', 'regression', 'r2'),
           'jc_penney_products': ('sale_price', 'regression', 'r2'),
           'women_clothing_review': ('Rating', 'regression', 'r2'),
           'ae_price_prediction': ('price', 'regression', 'r2'),
           'news_popularity2': ('log_shares', 'regression', 'r2'),
           'california_house_price': ('Sold Price', 'regression', 'r2'),
           'mercari_price_suggestion100K': ('price', 'regression', 'r2')}

melbourne_airbnb_columns_to_drop = ['listing_url', 'picture_url', 'host_url', 'host_thumbnail_url', 'host_picture_url']

columns_that_kill = ['Address']
columns_that_slow = ['Parking', 'Elementary School', 'Appliances included', 'Parking features', 'Listed On',
                     'Last Sold On'] # r2 = 0.917 with Summary
california_house_price_columns_to_drop = ['Address', 'Heating', 'Cooling', 'Parking', 'Bedrooms', 'Region',
                                          'Elementary School', 'Middle School', 'High School', 'Flooring',
                                          'Heating features', 'Cooling features', 'Appliances included',
                                          'Laundry features', 'Parking features', 'City', 'Listed On', 'Last Sold On',
                                          'Summary']

# california_house_price_columns_to_drop = ['Address', 'Heating', 'Cooling', 'Parking', 'Bedrooms', 'Region',
#                                           'Elementary School', 'Middle School', 'High School', 'Flooring',
#                                           'Heating features', 'Cooling features', 'Appliances included',
#                                           'Laundry features', 'Parking features', 'City', 'Listed On', 'Last Sold On',
#                                           'Summary']

columns_to_drop = columns_that_slow + columns_that_kill # r2 = 0.921 same as california_house_price_columns_to_drop


def get_text_sources_names(data: MultiModalData) -> list:
    text_sources = [source.split('/')[1] for source in list(data.keys()) if 'data_source_text' in source]
    return text_sources


def run_multimodal_dataset(dataset_name: str, timeout: int = 1, n_jobs: int = 4):
    print(f'Fit of dataset {dataset_name} is started')
    #try:
    target, task, metrics = targets.get(dataset_name)
    fit_path = Path(fedot_project_root(), 'examples/data/', dataset_name, 'train.csv')
    predict_path = Path(fedot_project_root(), 'examples/data/', dataset_name, 'test.csv')

    fit_data = MultiModalData.from_csv(file_path=fit_path, task=task, target_columns=target, index_col=None)
                                       #columns_to_drop=columns_to_drop)
    predict_data = MultiModalData.from_csv(file_path=predict_path, task=task, target_columns=target, index_col=None,
                                           text_columns=get_text_sources_names(fit_data))
                                           #columns_to_drop=columns_to_drop)

    automl_model = Fedot(problem=task, timeout=timeout, n_jobs=n_jobs, safe_mode=False, metric='acc')
    automl_model.fit(features=fit_data,
                     target=fit_data.target)

    prediction = automl_model.predict(predict_data)

    if metrics == 'accuracy':
        res_metrics = accuracy_score(predict_data.target, prediction)
    elif metrics == 'roc_auc':
        res_metrics = roc_auc_score(predict_data.target, prediction)
    elif metrics == 'r2':
        res_metrics = r2_score(predict_data.target, prediction)

    print(f'dataset {dataset_name} successfully finished with {metrics} = {np.round(res_metrics, 3)}')
    automl_model.history.save(f'history_{dataset_name}.json')
    automl_model.current_pipeline.save(f'pipeline_{dataset_name}')
    # except Exception as ex:
    #     print(f'dataset {dataset_name} failed')
    #     template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    #     message = template.format(type(ex).__name__, ex.args)
    #     print(message)


if __name__ == '__main__':
    # for dataset in os.listdir(Path(fedot_project_root(), 'examples', 'data', 'mm_benchmark')):
    #    if '_train' in dataset:
    #        run_multimodal_dataset(dataset_name=str(dataset.split('_train')[0]), timeout=1)
    run_multimodal_dataset(dataset_name='news_channel', timeout=10, n_jobs=6)
