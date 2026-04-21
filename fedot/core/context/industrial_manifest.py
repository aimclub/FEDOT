from fedot.extensions.contracts import ExtensionManifest
from fedot.core.context.factories import ( industrial_context_factory, splitters_factory,
                                           data_merger_factory, image_merger_factory, ts_merger_factory,
                                           text_merger_factory, data_source_splitter_factory, tuner_class_factory,
                                           reproduction_factory, evaluator_factory, search_space_factory,
                                           mutations_factory, operation_predict_factory, lagged_transformer_factory,
                                           topo_features_factory, ts_smoothing_factory, api_composer_tune_factory)


FEDOT_INDUSTRIAL_MANIFEST = ExtensionManifest(
    name="industrial",
    version="1.0.0",
    models=(),
    description="Industrial extension for FEDOT with Dask support and optimized operations.",
    protocols={
        "context_factory": industrial_context_factory,
        "splitters": splitters_factory,
        "data_merger": data_merger_factory,
        "image_merger": image_merger_factory,
        "ts_merger": ts_merger_factory,
        "text_merger": text_merger_factory,
        "data_source_splitter": data_source_splitter_factory,
        "tuner_class": tuner_class_factory,
        "reproduction": reproduction_factory,
        "evaluator": evaluator_factory,
        "search_space": search_space_factory,
        "default_mutations": mutations_factory,
        "operation_predict": operation_predict_factory,
        "lagged_transformer": lagged_transformer_factory,
        "topological_features": topo_features_factory,
        "ts_smoothing": ts_smoothing_factory,
        "api_composer_tune": api_composer_tune_factory,
    }
)