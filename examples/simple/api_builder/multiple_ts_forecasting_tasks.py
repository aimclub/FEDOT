from fedot.api.builder import FedotBuilder
from fedot.core.utils import fedot_project_root


if __name__ == '__main__':
    SEED = 42

    builder = (FedotBuilder('ts_forecasting')
               .setup_composition(preset='fast_train', timeout=0.5, with_tuning=True, seed=SEED)
               .setup_evolution(num_of_generations=3)
               .setup_pipeline_evaluation(metric='mae'))

    datasets_path = fedot_project_root() / 'examples/data/ts'
    resulting_models = {}
    for data_path in datasets_path.iterdir():
        if data_path.name == 'ts_sea_level.csv':
            continue
        fedot = builder.build()
        fedot.fit(data_path, target='value')
        fedot.predict(features=fedot.train_data, validation_blocks=2)
        fedot.plot_prediction()
        fedot.current_pipeline.show()
        resulting_models[data_path.stem] = fedot
