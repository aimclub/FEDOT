import pytest

from fedot.core.operations.evaluation.model_engine_rules import (
    ModelEngine,
    ModelEngineCapabilities,
    ModelEngineRequest,
    RuntimePlatform,
    build_model_engine_plan,
    resolve_runtime_platform,
)


@pytest.mark.unit
def test_cuml_engine_is_selected_only_for_native_linux_cuda():
    capabilities = ModelEngineCapabilities(
        platform=RuntimePlatform.LINUX,
        cuda_available=True,
        installed_engines=frozenset({ModelEngine.CUML, ModelEngine.SKLEARN}),
    )

    plan = build_model_engine_plan(
        ModelEngineRequest(
            supported_engines=(ModelEngine.CUML, ModelEngine.SKLEARN),
            require_acceleration=True,
        ),
        capabilities,
    )

    assert plan.engine is ModelEngine.CUML
    assert plan.device_type == 'cuda'


@pytest.mark.unit
def test_auto_engine_plan_falls_back_in_declared_order():
    capabilities = ModelEngineCapabilities(
        platform=RuntimePlatform.LINUX,
        cuda_available=False,
        installed_engines=frozenset({ModelEngine.TORCH, ModelEngine.SKLEARN}),
    )

    plan = build_model_engine_plan(
        ModelEngineRequest(supported_engines=(ModelEngine.CUML, ModelEngine.TORCH, ModelEngine.SKLEARN)),
        capabilities,
    )

    assert plan.engine is ModelEngine.TORCH
    assert plan.device_type == 'cpu'


@pytest.mark.unit
@pytest.mark.parametrize(
    ('system_name', 'release', 'expected'),
    [
        ('Linux', '6.8.0', RuntimePlatform.LINUX),
        ('Linux', '5.15.0-microsoft-standard-WSL2', RuntimePlatform.WSL),
        ('Windows', '11', RuntimePlatform.WINDOWS),
        ('Darwin', '24.0', RuntimePlatform.MACOS),
    ],
)
def test_runtime_platform_resolution(system_name, release, expected):
    assert resolve_runtime_platform(system_name, release) is expected


@pytest.mark.unit
def test_explicit_cuml_plan_rejects_wsl_even_when_cuda_is_visible():
    capabilities = ModelEngineCapabilities(
        platform=RuntimePlatform.WSL,
        cuda_available=True,
        installed_engines=frozenset({ModelEngine.CUML}),
    )

    with pytest.raises(RuntimeError, match='native Linux'):
        build_model_engine_plan(
            ModelEngineRequest(
                supported_engines=(ModelEngine.CUML,),
                preferred_engine=ModelEngine.CUML,
            ),
            capabilities,
        )


@pytest.mark.unit
def test_accelerated_plan_does_not_silently_fall_back_to_sklearn():
    capabilities = ModelEngineCapabilities(
        platform=RuntimePlatform.LINUX,
        cuda_available=False,
        installed_engines=frozenset({ModelEngine.SKLEARN}),
    )

    with pytest.raises(RuntimeError, match='available only on CPU'):
        build_model_engine_plan(
            ModelEngineRequest(
                supported_engines=(ModelEngine.CUML, ModelEngine.SKLEARN),
                require_acceleration=True,
            ),
            capabilities,
        )
