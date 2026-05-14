import pytest
from unittest.mock import Mock
from fedot.core.context.context import ExecutionContext

@pytest.fixture
def ctx():
    return ExecutionContext()


def test_default_initialization(ctx):
    assert ctx.extra_params == {}
    assert ctx._instances == {}
    assert ctx._overridden == {}
    assert "splitter" in ctx._protocol_classes
    assert "evaluator" in ctx._protocol_classes


def test_extra_params_stored():
    extra = {"backend": "dask", "timeout": 30}
    ctx = ExecutionContext(extra_params=extra)
    assert ctx.extra_params == extra


def test_lazy_instantiation(ctx):
    assert "splitter" not in ctx._instances
    splitter1 = ctx.splitter
    splitter2 = ctx.splitter
    assert splitter1 is splitter2
    assert "splitter" in ctx._instances


def test_core_implementations_are_used_by_default(ctx):
    from fedot.core.context.default_backend import CoreSplitter, CoreEvaluator, CoreDataMerger

    assert isinstance(ctx.splitter, CoreSplitter)
    assert isinstance(ctx.evaluator, CoreEvaluator)
    assert isinstance(ctx.data_merger, CoreDataMerger)


def test_method_override():
    ctx_ind = ExecutionContext(extension_name="industrial")

    from fedot.core.context.industrial_backend import IndustrialSplitter
    assert isinstance(ctx_ind.splitter, IndustrialSplitter)

def test_missing(ctx):
    with pytest.raises(ValueError, match="No implementation for protocol 'unknown'"):
        ctx._get_protocol_class("unknown")


def test_attribute_override(ctx):
    ctx.custom_attr = 123
    assert ctx._overridden["custom_attr"] == 123
    assert ctx.custom_attr == 123

def test_multiple_contexts_independence():
    ctx1 = ExecutionContext(extra_params={"p": 1})
    ctx2 = ExecutionContext(extra_params={"p": 2})
    assert ctx1.extra_params["p"] == 1
    assert ctx2.extra_params["p"] == 2


def test_method_called_with_parameters(ctx):
    from unittest.mock import patch
    from fedot.core.data.data import InputData

    with patch('fedot.core.context.default_backend.CoreSplitter.split_any') as mock_split:
        mock_split.return_value = ("train_data", "test_data")

        mock_data = Mock(spec=InputData)

        splitter = ctx.splitter
        result = splitter.split_any(
            data=mock_data,
            split_ratio=0.75,
            shuffle=True,
            stratify=True,
            random_seed=42,
            extra_arg="custom_value"
        )

        mock_split.assert_called_once()

        args, kwargs = mock_split.call_args
        assert kwargs['data'] == mock_data
        assert kwargs['split_ratio'] == 0.75
        assert kwargs['shuffle'] is True
        assert kwargs['stratify'] is True
        assert kwargs['random_seed'] == 42
        assert kwargs['extra_arg'] == "custom_value"