from types import SimpleNamespace

import pytest

from fedot.api.api_utils.predefined_model import PredefinedModel


@pytest.mark.unit
def test_predefined_model_fit_tensordata_uses_pipeline_tensor_runtime(monkeypatch):
    captured = {}

    class FakePipeline:
        def fit_tensordata(self, data):
            captured['data'] = data

    monkeypatch.setattr(
        PredefinedModel,
        '_get_pipeline',
        lambda self,
        use_input_preprocessing=True,
        api_preprocessor=None: FakePipeline())

    model = PredefinedModel(
        predefined_model='logit',
        data=SimpleNamespace(task=SimpleNamespace(task_type='classification')),
        log=SimpleNamespace(message=lambda *args, **kwargs: None),
    )

    result = model.fit_tensordata()

    assert isinstance(result, FakePipeline)
    assert captured['data'] == model.data
