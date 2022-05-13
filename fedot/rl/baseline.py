from os.path import join

from sklearn.metrics import roc_auc_score as roc_auc

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.utils import fedot_project_root

if __name__ == '__main__':
    path_to_train = join(fedot_project_root(), 'fedot/rl/data/train/')
    path_to_valid = join(fedot_project_root(), 'fedot/rl/data/valid/')

    # datasets = [file_name for (_, _, file_name) in walk(path_to_train)][0]

    datasets = [
        # 'adult.csv',
        'Amazon_employee_access.csv',
        'Australian.csv',
        'bank-marketing.csv',
        'blood-transfusion-service-center.csv',
        'jasmine.csv',
        'kc1.csv',
        # 'kr-vs-kp.csv',
        # 'phoneme.csv',
        # 'sylvine.csv'
    ]

    problem = 'classification'

    for dataset in datasets:
        train_data = InputData.from_csv(join(path_to_train, dataset))
        valid_data = InputData.from_csv(join(path_to_valid, dataset))

        baseline_model = Fedot(problem=problem, timeout=4)
        baseline_model.fit(train_data, predefined_model='rf')

        results = baseline_model.predict(valid_data)

        metric_value = roc_auc(
            y_true=valid_data.target,
            y_score=results,
            multi_class='ovo',
            average='macro'
        )

        print(dataset, '- roc_auc:', metric_value)
