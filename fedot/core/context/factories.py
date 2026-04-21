from fedot.core.context.industrial_backend import (IndustrialSplitter, IndustrialDataMerger, IndustrialImageMerger,
                                                   IndustrialTSMerger, IndustrialTextMerger,
                                                   IndustrialDataSourceSplitterBuilder, IndustrialTunerClass,
                                                   IndustrialReproduction, IndustrialEvaluator, IndustrialSearchSpace,
                                                   IndustrialDefaultMutations,IndustrialOperationPredict,
                                                   IndustrialLaggedTransformer, IndustrialTopologicalFeatures,
                                                   IndustrialTsSmoothing, IndustrialApiComposerTune)



def industrial_context_factory(backend: str = "default"):
    return IndustrialContext(backend=backend)

def splitters_factory():
    return IndustrialSplitter()

def data_merger_factory():
    return IndustrialDataMerger()

def image_merger_factory():
    return IndustrialImageMerger()

def ts_merger_factory():
    return IndustrialTSMerger()

def text_merger_factory():
    return IndustrialTextMerger()

def data_source_splitter_factory():
    return IndustrialDataSourceSplitterBuilder()

def tuner_class_factory(backend: str = "default"):
    return IndustrialTunerClass(backend)

def reproduction_factory():
    return IndustrialReproduction()

def evaluator_factory():
    return IndustrialEvaluator()

def search_space_factory():
    return IndustrialSearchSpace()

def mutations_factory():
    return IndustrialDefaultMutations()

def operation_predict_factory():
    return IndustrialOperationPredict()

def lagged_transformer_factory():
    return IndustrialLaggedTransformer()

def topo_features_factory():
    return IndustrialTopologicalFeatures()

def ts_smoothing_factory():
    return IndustrialTsSmoothing()

def api_composer_tune_factory():
    return IndustrialApiComposerTune()