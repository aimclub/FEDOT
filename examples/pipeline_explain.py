from examples.pipeline_tune import get_case_train_test_data
from examples.pipeline_tune import get_simple_pipeline


if __name__ == '__main__':
    train_data, test_data = get_case_train_test_data()

    # Pipeline composition
    pipeline = get_simple_pipeline()

    # Pipeline fitting
    pipeline.fit(train_data, use_fitted=False)

    # Pipeline explaining
    explainer = pipeline.explain(data=train_data, method='surrogate_dt', instant_output=False)

    print(f'Built surrogate model: {explainer.surrogate_str}')
    explainer.output()
