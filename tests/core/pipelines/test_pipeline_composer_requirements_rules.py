import pytest

from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.pipeline_composer_requirements_rules import (
    PipelineEvaluationMode,
    resolve_pipeline_evaluation_mode,
)


@pytest.mark.unit
@pytest.mark.parametrize('mode', ['oof', PipelineEvaluationMode.oof])
def test_resolve_pipeline_evaluation_mode(mode):
    assert resolve_pipeline_evaluation_mode(mode) is PipelineEvaluationMode.oof


@pytest.mark.unit
def test_pipeline_composer_requirements_normalizes_evaluation_mode():
    requirements = PipelineComposerRequirements(evaluation_mode='oof')

    assert requirements.evaluation_mode is PipelineEvaluationMode.oof


@pytest.mark.unit
def test_resolve_pipeline_evaluation_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match='Unsupported pipeline evaluation mode'):
        resolve_pipeline_evaluation_mode('holdout')
