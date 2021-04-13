import numpy as np

from fedot.core.data.data import InputData
from fedot.core.chains.node import PrimaryNode
from fedot.core.chains.chain import Chain
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def test_clean_text_preprocessing():
    test_text = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]

    input_data = InputData(features=test_text,
                           target=[0, 1, 1, 0],
                           idx=np.arange(0, len(test_text)),
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.text)

    preprocessing_chain = Chain(PrimaryNode('text_clean'))
    preprocessing_chain.fit(input_data)

    predicted_output = preprocessing_chain.predict(input_data)
    cleaned_text = predicted_output.predict

    assert len(test_text) == len(cleaned_text)
