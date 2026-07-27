import torch
import torch.nn as nn
from typing import Optional

from fedot.core.operations.operation_parameters import OperationParameters


class TorchLinearClassifier:
    """Linear classification model implemented with ``nn.Linear`` for TensorData runtime."""

    def __init__(self, params: Optional[OperationParameters] = None):
        self.params = params or OperationParameters()
        self.module: Optional[nn.Linear] = None
        self.device = torch.device('cpu')
        self.classes_: Optional[torch.Tensor] = None

    def fit(self, features: torch.Tensor, target: torch.Tensor) -> 'TorchLinearClassifier':
        """ Method fit model on a dataset

        :param features: feature tensor of shape (n_samples, n_features)
        :param target: target tensor of shape (n_samples,)
        """
        n_features = features.shape[1]
        classes = torch.unique(target)
        class_to_index = {label.item(): index for index, label in enumerate(classes)}
        remapped_target = torch.tensor(
            [class_to_index[label.item()] for label in target],
            dtype=torch.long,
            device=self.device,
        )

        n_classes = len(classes)
        self.module = nn.Linear(n_features, n_classes).to(self.device)
        self.classes_ = classes

        learning_rate = float(self.params.get('learning_rate') or 0.05)
        epochs = int(self.params.get('epochs') or 200)
        optimizer = torch.optim.Adam(self.module.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.module.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = criterion(self.module(features), remapped_target)
            loss.backward()
            optimizer.step()

        self.module.eval()
        return self

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """ Method make prediction with probabilities of classes

        :param features: data with features to process
        """
        if self.module is None:
            raise ValueError('TorchLinearClassifier is not fitted yet')

        with torch.no_grad():
            logits = self.module(features)
            probabilities = torch.softmax(logits, dim=-1)
        return probabilities

    def predict_labels(self, features: torch.Tensor) -> torch.Tensor:
        """ Method make prediction with labels of classes

        :param features: data with features to process
        """
        probabilities = self.predict_proba(features)
        class_indices = probabilities.argmax(dim=-1)
        return self.classes_[class_indices]
