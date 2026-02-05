from collections import OrderedDict
from typing import Optional

from fedot.core.operations.operation_parameters import OperationParameters
from torch import Tensor, cuda, device, no_grad
from torch.nn import Conv1d, ConvTranspose1d, Module, MSELoss, Sequential, ReLU
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from tqdm import tqdm

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.models.detection.anomaly.algorithms.autoencoder_detector import AutoEncoderDetector
from fedot.industrial.core.models.nn.network_modules.layers.special import EarlyStopping

device = device("cuda:0" if cuda.is_available() else "cpu")


class ConvolutionalAutoEncoderDetector(AutoEncoderDetector):
    """A reconstruction convolutional autoencoder model to detect anomalies
    in timeseries data using reconstruction error as an anomaly score.
    """

    def build_model(self):
        self.params.update(**{'n_steps': self.n_steps, 'learning_rate': self.learning_rate})
        return ConvolutionalAutoEncoder(self.params).to(device)


class ConvolutionalAutoEncoder(Module):
    def __init__(self, params: Optional[OperationParameters] = None):
        super(ConvolutionalAutoEncoder, self).__init__()
        self.learning_rate = params.get('learning_rate', 0.001)
        self.n_steps = params.get('n_steps', 10)
        self.encoder_layers = params.get('num_encoder_layers', 2)
        self.decoder_layers = params.get('num_decoder_layers', 2)
        self.latent_layer_params = params.get('latent_layer', 16)
        self.convolutional_params = params.get('convolutional_params', dict(kernel_size=3,
                                                                            stride=2,
                                                                            padding=1)
                                               )
        self.activation_func = params.get('act_func', ReLU)
        self.dropout_rate = params.get('dropout_rate', 0.5)

    def _init_model(self) -> tuple:
        self._build_encoder()
        self._build_decoder()
        self.loss_fn = MSELoss()
        self.optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return self.loss_fn, self.optimizer

    def _build_encoder(self):
        encoder_layer_dict = OrderedDict()
        for i in range(self.encoder_layers):
            if i == 0:
                in_channels = self.n_steps
                out_channels = self.latent_layer_params
            elif i == self.encoder_layers - 1:
                out_channels = self.latent_layer_params
            else:
                out_channels = 32
            encoder_layer_dict.update({f'conv{i}': Conv1d(in_channels=in_channels,
                                                          out_channels=out_channels,
                                                          **self.convolutional_params)})
            encoder_layer_dict.update({f'relu{i}': self.activation_func()})
            in_channels = out_channels
        self.encoder = Sequential(encoder_layer_dict)

    def _build_decoder(self):
        decoder_layer_dict = OrderedDict()
        for i in range(self.decoder_layers):
            if i == 0:
                in_channels = self.latent_layer_params
                out_channels = self.n_steps
            elif i == self.encoder_layers - 1:
                out_channels = self.n_steps
            else:
                out_channels = 32

            decoder_layer_dict.update({f'conv{i}':
                                       ConvTranspose1d(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       output_padding=1, **self.convolutional_params)})
            decoder_layer_dict.update({f'relu{i}': self.activation_func()})
            in_channels = out_channels
        self.decoder = Sequential(decoder_layer_dict)

    def _create_dataloader(self, input_data, batch_size, validation_split):
        dataset = TensorDataset(Tensor(input_data))
        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(validation_split * num_train))

        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
        return train_loader, valid_loader

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self,
            data,
            epochs: int = 100,
            batch_size: int = 32,
            validation_split: float = 0.1):
        self._init_model()
        train_loader, valid_loader = self._create_dataloader(data, batch_size, validation_split)
        train_steps, early_stopping, best_model, best_val_loss = max(1, len(train_loader)), EarlyStopping(), \
            None, float('inf')
        scheduler = lr_scheduler.OneCycleLR(optimizer=self.optimizer,
                                            steps_per_epoch=train_steps,
                                            epochs=epochs,
                                            max_lr=self.learning_rate)

        def train_one_batch(batch):
            batch_x = batch[0]
            self.optimizer.zero_grad()
            outputs = self.forward(batch_x)
            loss = self.loss_fn(outputs, batch_x)
            loss.backward()
            self.optimizer.step()
            return loss.item()

        def val_one_epoch(batch):
            inputs = batch[0]
            output = self.forward(inputs)
            loss = self.loss_fn(output, inputs)
            return loss.data.item() * inputs.size(0)

        for epoch in tqdm(range(epochs)):
            self.train()
            train_loss = list(map(lambda batch_tuple: train_one_batch(batch_tuple), train_loader))
            train_loss = np.average(train_loss)
            if valid_loader is not None:
                self.eval()
                valid_loss = list(map(lambda batch_tuple: val_one_epoch(batch_tuple), valid_loader))
                valid_loss = np.average(valid_loss)
            last_lr = scheduler.get_last_lr()[0]
            if epoch % 25 == 0:
                print(
                    "Epoch: {0}, Train Loss: {1} | Validation Loss: {2:.7f}".format(epoch + 1, train_loss, valid_loss))
                print('Updating learning rate to {}'.format(last_lr))
            if early_stopping.early_stop:
                print("Early stopping")
                break
        return self

    def predict(self, data):
        self.eval()
        with no_grad():
            data_torch = Tensor(data)
            predictions = self.forward(data_torch)
            return predictions.numpy()

    def score_samples(self, data):
        train_prediction = self.predict(data)
        residuals = np.abs(data - train_prediction)
        residuals = np.mean(residuals, axis=1).sum(axis=1)
        return residuals
