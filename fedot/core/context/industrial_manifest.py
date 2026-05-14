from fedot.extensions.contracts import ExtensionManifest
from fedot.core.context.industrial_backend import (
    IndustrialSplitter,
    IndustrialDataMerger,
    IndustrialImageMerger,
    IndustrialTSMerger,
    IndustrialTextMerger,
    IndustrialDataSourceSplitterBuilder,
    IndustrialTunerClass,
    IndustrialReproduction,
    IndustrialEvaluator,
    IndustrialSearchSpace,
    IndustrialDefaultMutations,
    IndustrialOperationPredict,
    IndustrialLaggedTransformer,
    IndustrialTopologicalFeatures,
    IndustrialTsSmoothing,
    IndustrialApiComposerTune,
)

FEDOT_INDUSTRIAL_MANIFEST = ExtensionManifest(
    name="industrial",
    version="1.0.0",
    models=(),
    description="Industrial extension for FEDOT.",
    protocols={
        "splitter": IndustrialSplitter,
        "data_merger": IndustrialDataMerger,
        "image_merger": IndustrialImageMerger,
        "ts_merger": IndustrialTSMerger,
        "text_merger": IndustrialTextMerger,
        "data_source_splitter": IndustrialDataSourceSplitterBuilder,
        "tuner_class": IndustrialTunerClass,
        "reproduction": IndustrialReproduction,
        "evaluator": IndustrialEvaluator,
        "search_space": IndustrialSearchSpace,
        "default_mutations": IndustrialDefaultMutations,
        "operation_predict": IndustrialOperationPredict,
        "lagged_transformer": IndustrialLaggedTransformer,
        "topological_features": IndustrialTopologicalFeatures,
        "ts_smoothing": IndustrialTsSmoothing,
        "api_composer_tune": IndustrialApiComposerTune,
    }
)
