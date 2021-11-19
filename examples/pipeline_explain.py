from examples.pipeline_tune import get_case_train_test_data
from examples.pipeline_tune import get_simple_pipeline


if __name__ == '__main__':
    train_data, test_data = get_case_train_test_data()

    # Synthetic names for visualization
    feature_names = [f'feature {f}' for f in range(len(train_data.features[0]))]
    class_names = list({f'class {cl[0]}' for cl in train_data.target})

    # Pipeline composition
    pipeline = get_simple_pipeline()

    # Pipeline fitting
    pipeline.fit(train_data, use_fitted=False)

    # Pipeline explaining
    explainer = pipeline.explain(data=train_data, method='surrogate_dt', visualize=False)

    # Visualizing explanation
    print(f'Built surrogate model: {explainer.surrogate_str}')
    explainer.visualize(save_path='explanation.png', feature_names=feature_names, class_names=class_names, dpi=300)
