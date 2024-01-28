from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.operation_tags_n_repo_enums import DataOperationTagsEnum
from fedot.preprocessing.structure import PipelineStructureExplorer


def test_correct_pipeline_encoder_imputer_validation():
    """
    Validate correct unimodal pipeline. Both encoding and imputation operations
    take right places in the pipeline.
    """
    first_source = PipelineNode('data_source_table')
    second_imputer = PipelineNode('simple_imputation', nodes_from=[first_source])
    third_encoder = PipelineNode('one_hot_encoding', nodes_from=[second_imputer])
    fourth_imputer = PipelineNode('simple_imputation', nodes_from=[third_encoder])
    root = PipelineNode('linear', nodes_from=[fourth_imputer])
    pipeline = Pipeline(root)

    encoding_correct = PipelineStructureExplorer().check_structure_by_tag(pipeline,
                                                                          tag_to_check=DataOperationTagsEnum.encoding)
    imputer_correct = PipelineStructureExplorer().check_structure_by_tag(pipeline,
                                                                         tag_to_check=DataOperationTagsEnum.imputation)

    assert encoding_correct is True
    assert imputer_correct is True


def test_non_correct_pipeline_encoder_imputer_validation():
    """
    DataPreprocessor should correctly identify is pipeline has needed operations
    (encoding, imputation) in right order or not.
    In the case presented below incorrect unimodal pipeline generated.
    """
    first_imputation = PipelineNode('simple_imputation')
    first_encoder = PipelineNode('one_hot_encoding')

    second_rfr = PipelineNode('rfr', nodes_from=[first_imputation])
    second_ridge = PipelineNode('ridge', nodes_from=[first_encoder])
    second_encoder = PipelineNode('one_hot_encoding', nodes_from=[first_imputation])

    third_imputer = PipelineNode('simple_imputation', nodes_from=[second_ridge])
    root = PipelineNode('linear', nodes_from=[second_rfr, third_imputer, second_encoder])
    pipeline = Pipeline(root)

    encoding_correct = PipelineStructureExplorer().check_structure_by_tag(pipeline,
                                                                          tag_to_check=DataOperationTagsEnum.encoding)
    imputer_correct = PipelineStructureExplorer().check_structure_by_tag(pipeline,
                                                                         tag_to_check=DataOperationTagsEnum.imputation)

    assert encoding_correct is False
    assert imputer_correct is False
