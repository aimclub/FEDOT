import logging

from fedot.validation.context import ValidationContext


def test_validation_context_default_logger():
    context = ValidationContext()
    logger = context.get_logger()
    assert logger is not None
    assert hasattr(logger, 'warning')


def test_validation_context_custom_logger():
    custom_logger = logging.getLogger('test_custom_validation_logger')
    context = ValidationContext(logger=custom_logger)
    assert context.get_logger() is custom_logger
