from fedot.industrial.api.utils.api_init import ApiManager, IndustrialConfig, LearningConfig


def test_industrial_config_uses_rule_based_context_and_initial_problem(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        'fedot.industrial.api.utils.api_init.fedot_init_assumptions',
        lambda problem: captured.setdefault('problem', problem) or 'initial-assumption',
    )

    config = IndustrialConfig().build({
        'problem': 'classification',
        'strategy': 'tabular',
        'task_params': {},
        'initial_assumption': None,
        'optimizer': object,
        'use_input_preprocessing': False,
        'strategy_params': None,
    })

    assert config.is_default_fedot_context is True
    assert captured['problem'] == 'classification_tabular'


def test_learning_config_with_loss_uses_rule_based_loss_plan():
    config = LearningConfig().build({'optimisation_loss': {'quality_loss': 'f1', 'structural_loss': 'size'}})

    assert config.quality_loss == 'f1'
    assert config.structural_loss == 'size'
    assert config.computational_loss is None


def test_api_manager_null_state_object_uses_state_plan_defaults():
    manager = ApiManager()

    assert manager.solver is None
    assert manager.predicted_labels is None
    assert manager.predicted_probs is None
    assert manager.predict_data is None
    assert manager.dask_client is None
    assert manager.dask_cluster is None
    assert manager.target_encoder is None
    assert manager.is_finetuned is False
