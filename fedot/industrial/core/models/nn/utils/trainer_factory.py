"""
Trainer Factory for creating appropriate trainers based on task type
"""

from typing import Any, Dict, Optional, Type
import logging
from fedot.core.operations.operation_parameters import OperationParameters

from fedot.industrial.core.models.nn.network_impl.base_nn_model import (BaseNeuralModel, BaseNeuralForecaster)
from fedot.industrial.core.models.nn.network_impl.llm_trainer import LLMTrainer
from fedot.industrial.core.models.nn.utils.interfaces import ITrainer

logger = logging.getLogger(__name__)


def _analyze_model_architecture(model: Any) -> str:
    """
    Analyze model architecture to determine appropriate trainer type.

    Args:
        model: The model to analyze

    Returns:
        str: Trainer type ('llm', 'forecasting', 'general')
    """
    if model is None:
        return 'general'

    model_class_name = model.__class__.__name__.lower()

    llm_patterns = [
        'bert', 'gpt', 'transformer', 't5', 'roberta', 'distilbert',
        'albert', 'xlnet', 'electra', 'bart', 'llama', 'mistral',
        'bloom', 'opt', 'falcon', 'chatglm', 'qwen'
    ]

    for pattern in llm_patterns:
        if pattern in model_class_name:
            return 'llm'

    if hasattr(model, 'config'):
        config = model.config
        config_class_name = config.__class__.__name__.lower()

        transformer_config_patterns = [
            'bert', 'gpt', 'transformer', 't5', 'roberta', 'distilbert',
            'albert', 'xlnet', 'electra', 'bart', 'llama', 'vit'
        ]

        for pattern in transformer_config_patterns:
            if pattern in config_class_name:
                return 'llm'

        transformer_attrs = [
            'num_hidden_layers', 'num_attention_heads', 'hidden_size',
            'vocab_size', 'max_position_embeddings', 'type_vocab_size'
        ]

        if any(hasattr(config, attr) for attr in transformer_attrs):
            return 'llm'

    forecasting_patterns = [
        'lstm', 'gru', 'rnn', 'tcn', 'temporal', 'time', 'forecast',
        'arima', 'prophet', 'ets', 'nbeats', 'nhits', 'transformerforecast'
    ]

    for pattern in forecasting_patterns:
        if pattern in model_class_name:
            return 'forecasting'

    if hasattr(model, 'modules'):
        for module in model.modules():
            module_name = module.__class__.__name__.lower()
            if any(pattern in module_name for pattern in ['lstm', 'gru', 'rnn', 'tcn']):
                return 'forecasting'

    logger.debug("No specific architecture detected, returning 'general'")
    return 'general'


def _get_trainer_class(model: Any, task_type: str, params: Dict) -> Type[ITrainer]:
    """
    Determine the appropriate trainer class based on explicit config first, then model architecture and task type.
    Args:
        model: The model to train (type of input data, not a parameter)
        task_type: Task type from input data
        params: Training parameters

    Returns:
        Type[ITrainer]: Appropriate trainer class
    """
    is_llm = params.get('is_llm', None)

    if is_llm is True:
        logger.info("Creating LLMTrainer based on explicit is_llm=True parameter")
        return LLMTrainer
    elif is_llm is False:
        logger.info("Explicit is_llm=False, skipping LLM trainer")

    architecture_type = _analyze_model_architecture(model)
    task_type_lower = task_type.lower()

    logger.info(f"Determining trainer class for architecture: {architecture_type}, task: {task_type}")

    if architecture_type == 'llm' and is_llm is not False:
        logger.info("Creating LLMTrainer based on transformer architecture")
        return LLMTrainer

    elif architecture_type == 'forecasting':
        logger.info("Creating BaseNeuralForecaster based on forecasting architecture")
        return BaseNeuralForecaster

    elif 'forecasting' in task_type_lower:
        logger.info(f"Creating BaseNeuralForecaster based on task type: {task_type}")
        return BaseNeuralForecaster

    elif 'llm' in task_type_lower or 'transformer' in task_type_lower:
        logger.info(f"Creating LLMTrainer based on task type: {task_type}")
        return LLMTrainer

    else:
        logger.info(f"Creating BaseNeuralModel for general task: {task_type}")
        return BaseNeuralModel


def create_trainer(
        task_type: str,
        params: Optional[OperationParameters] = None,
        model: Any = None,
        **kwargs
) -> ITrainer:
    """
    Create appropriate trainer based on model architecture and task type.

    Model is treated as input data type, NOT as a parameter.
    Model is passed directly to trainer constructor, not through params dict.

    Args:
        task_type: Type of task ('forecasting', 'llm', 'classification', 'regression', etc.)
        params: Training parameters (WITHOUT model)
        model: Model to train (type of input data)
        **kwargs: Additional arguments

    Returns:
        ITrainer: Appropriate trainer instance
    """

    if params is not None and hasattr(params, 'to_dict'):
        params_dict = params.to_dict()
    else:
        params_dict = params or {}
    trainer_class = _get_trainer_class(model, task_type, params_dict)

    return trainer_class(params=params_dict, **kwargs)


def create_trainer_from_input_data(
        input_data: Any,
        params: Optional[OperationParameters] = None,
        model: Any = None,
        **kwargs
) -> ITrainer:
    """
    Create appropriate trainer based on input data.

    Model is extracted from input_data (input_data.target) as it represents
    the type of input data, not a parameter.

    Args:
        input_data: Input data with task information and model (in input_data.target)
        params: Training parameters (WITHOUT model)
        model: Optional model override (if not provided, extracted from input_data.target)
        **kwargs: Additional arguments

    Returns:
        ITrainer: Appropriate trainer instance
    """
    task_type = input_data.task.task_type.value

    if model is None and hasattr(input_data, 'target'):
        model = input_data.target
        logger.debug("Extracted model from input_data.target (model is input data type)")

    logger.debug(f"Creating trainer - task_type: {task_type}, model: {type(model).__name__ if model else 'None'}")

    return create_trainer(task_type, params, model, **kwargs)