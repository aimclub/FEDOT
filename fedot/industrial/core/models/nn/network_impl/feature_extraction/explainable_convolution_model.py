from typing import Optional

import torch
from fastai.callback.hook import *
from fastai.layers import BatchNorm, LinBnDrop, SigmoidRange
from fastai.torch_core import Module
from fedot.core.operations.operation_parameters import OperationParameters
from torch import nn, optim

from fedot.industrial.core.architecture.abstraction.decorators import convert_inputdata_to_torch_dataset
from fedot.industrial.core.architecture.postprocessing.visualisation.gradcam_vis import visualise_gradcam
from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.architecture.settings.computational import default_device
from fedot.industrial.core.models.nn.network_impl.base_nn_model import BaseNeuralModel
from fedot.industrial.core.models.nn.network_modules.layers.conv_layers import Conv1d, Conv2d
from fedot.industrial.core.models.nn.network_modules.layers.linear_layers import Concat, Reshape, Squeeze, Flatten
from fedot.industrial.core.models.nn.network_modules.layers.pooling_layers import GACP1d, GAP1d


def torch_slice_by_dim(t, index, dim=-1, **kwargs):
    if not isinstance(index, torch.Tensor):
        index = torch.Tensor(index)
    assert t.ndim == index.ndim, "t and index must have the same ndim"
    index = index.long()
    return torch.gather(t, dim, index, **kwargs)


def create_head(
        nf,
        output_dim,
        seq_len=None,
        flatten=False,
        concat_pool=False,
        fc_dropout=0.,
        batch_norm=False,
        y_range=None):
    if flatten:
        nf *= seq_len
        layers = [Reshape()]
    else:
        if concat_pool:
            nf *= 2
        layers = [GACP1d(1) if concat_pool else GAP1d(1)]
    layers += [LinBnDrop(nf,
                         output_dim,
                         bn=batch_norm,
                         p=fc_dropout)]
    if y_range:
        layers += [SigmoidRange(*y_range)]
    return nn.Sequential(*layers)


class XCM(Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 seq_len: Optional[int] = None,
                 number_filters: int = 128,
                 window_perc: float = 1.,
                 flatten: bool = False,
                 custom_head: callable = None,
                 concat_pool: bool = False,
                 fc_dropout: float = 0.,
                 batch_norm: bool = False,
                 y_range: tuple = None,
                 **kwargs):

        window_size = int(round(seq_len * window_perc, 0))

        self.conv2dblock = nn.Sequential(*[Conv2d(input_dim,
                                                  number_filters,
                                                  kernel_size=(1, window_size),
                                                  padding='same'),
                                           BatchNorm(number_filters),
                                           nn.ReLU()])

        self.conv2d1x1block = nn.Sequential(*[nn.Conv2d(in_channels=number_filters,
                                                        out_channels=1,
                                                        kernel_size=1),
                                              nn.ReLU(),
                                              Squeeze(1)])

        self.conv1dblock = nn.Sequential(*[Flatten(),
                                           Conv1d(input_dim,
                                                  number_filters,
                                                  kernel_size=window_size,
                                                  padding='same'),
                                           BatchNorm(number_filters,
                                                     ndim=1),
                                           nn.ReLU()])

        self.conv1d1x1block = nn.Sequential(*[nn.Conv1d(number_filters,
                                                        1,
                                                        kernel_size=1),
                                              nn.ReLU()])
        self.flatten = Flatten()
        self.concat = Concat()
        self.conv1d = nn.Sequential(*[Conv1d(input_dim - 1,
                                             number_filters,
                                             kernel_size=window_size,
                                             padding='same'),
                                      BatchNorm(number_filters, ndim=1),
                                      nn.ReLU()])

        self.head_number_filters = number_filters
        self.output_dim = output_dim
        self.seq_len = seq_len
        if custom_head:
            self.head = custom_head(self.head_number_filters,
                                    output_dim,
                                    seq_len,
                                    **kwargs)
        else:
            self.head = create_head(
                self.head_number_filters,
                output_dim,
                seq_len,
                flatten=flatten,
                concat_pool=concat_pool,
                fc_dropout=fc_dropout,
                batch_norm=batch_norm,
                y_range=y_range)

    def forward(self, x):
        x1 = self.conv2dblock(x)
        x1 = self.conv2d1x1block(x1)
        x2 = self.conv1dblock(x)
        x2 = self.conv1d1x1block(x2)
        x1 = x1.reshape(x1.shape[0], 1, -1)

        out = self.concat((x2, x1))
        out = self.conv1d(out)
        out = self.head(out)
        return out

    def explain(self, input_data):
        target = input_data.target
        features = input_data.features
        median_dict = {}
        for class_number in input_data.class_labels:
            class_target_idx = np.where(target == class_number)[0]
            median_sample = np.median(features[class_target_idx], axis=0)
            median_dict.update({f'class_{class_number}': median_sample})
        input_data.supplementary_data = median_dict

        self._explain_by_gradcam(input_data)

    @convert_inputdata_to_torch_dataset
    def _explain_by_gradcam(self,
                            input_data,
                            detach=True,
                            cpu=True,
                            apply_relu=True,
                            cmap='inferno',
                            figsize=None,
                            **kwargs):

        att_maps = self.get_attribution_map(model=self,
                                            modules=[self.conv2dblock, self.conv1dblock],
                                            features=input_data.x,
                                            target=input_data.y,
                                            detach=detach,
                                            cpu=cpu,
                                            apply_relu=apply_relu)
        att_maps[0] = (att_maps[0] - att_maps[0].min()) / (att_maps[0].max() - att_maps[0].min())
        att_maps[1] = (att_maps[1] - att_maps[1].min()) / (att_maps[1].max() - att_maps[1].min())

        visualise_gradcam(att_maps,
                          input_data.supplementary_data,
                          figsize,
                          cmap,
                          **kwargs)

    def _get_acts_and_grads(self,
                            model,
                            modules,
                            x,
                            y=None,
                            detach=True,
                            cpu=False):
        """Returns activations and gradients for given modules in a model and a single input or a batch.
        Gradients require y value(s). If they are not provided, it will use the predictions.

        """
        if not isinstance(modules, list):
            modules = [modules]
        x = x[None, None] if x.ndim == 1 else x[None] if x.ndim == 2 else x

        with hook_outputs(modules, detach=detach, cpu=cpu) as h_act:
            with hook_outputs(modules, grad=True, detach=detach, cpu=cpu) as h_grad:
                preds = model.eval()(x)
                if y is None:
                    preds.max(dim=-1).values.mean().backward()
                else:
                    y = y.detach().cpu().numpy()
                    if preds.shape[0] == 1:
                        preds[0, y].backward()
                    else:
                        if y.ndim == 1:
                            y = y.reshape(-1, 1)
                        torch_slice_by_dim(preds, y).mean().backward()
        if len(modules) == 1:
            return h_act.stored[0].data, h_grad.stored[0][0].data
        else:
            return [h.data for h in h_act.stored], [h[0].data for h in h_grad.stored]

    def get_attribution_map(self,
                            model,
                            modules,
                            features,
                            target=None,
                            detach=True,
                            cpu=False,
                            apply_relu=True):
        def _get_attribution_map(A_k, w_ck):
            dim = (0, 2, 3) if A_k.ndim == 4 else (0, 2)
            w_ck = w_ck.mean(dim, keepdim=True)
            L_c = (w_ck * A_k).sum(1)
            if apply_relu:
                L_c = nn.ReLU()(L_c)
            if L_c.ndim == 3:
                return L_c.squeeze(0) if L_c.shape[0] == 1 else L_c
            else:
                return L_c.repeat(features.shape[1],
                                  1) if L_c.shape[0] == 1 else L_c.unsqueeze(1).repeat(1,
                                                                                       features.shape[1],
                                                                                       1)

        if features.ndim == 1:
            features = features[None, None]
        elif features.ndim == 2:
            features = features[None]
        A_k, w_ck = self._get_acts_and_grads(model,
                                             modules,
                                             features,
                                             target,
                                             detach=detach,
                                             cpu=cpu)
        if isinstance(A_k, list):
            return [_get_attribution_map(A_k[i], w_ck[i]) for i in range(len(A_k))]
        else:
            return _get_attribution_map(A_k, w_ck)


class XCModel(BaseNeuralModel):
    """Class responsible for Time series transformer (TST) model implementation.

    Attributes:
        self.num_features: int, the number of features.

    Example:
        To use this operation you can create pipeline as follows::
            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from examples.fedot.fedot_ex import init_input_data
            from fedot.industrial.tools.loader import DataLoader
            from fedot.industrial.core.repository.initializer_industrial_models import IndustrialModels

            train_data, test_data = DataLoader(dataset_name='Lightning7').load_data()
            input_data = init_input_data(train_data[0], train_data[1])
            val_data = init_input_data(test_data[0], test_data[1])

            with IndustrialModels():
                pipeline = PipelineBuilder().add_node('xcm_model', params={'epochs': 100,
                                                                           'batch_size': 10}).build()
                pipeline.fit(input_data)
                target = pipeline.predict(val_data).predict
                metric = evaluate_metric(target=test_data[1], prediction=target)

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.num_classes = self.params.get('num_classes', 1)
        self.epochs = self.params.get('epochs', 100)
        self.batch_size = self.params.get('batch_size', 32)
        self.is_regression_task = False

    def _init_model(self, ts):
        self.model = XCM(input_dim=ts.features.shape[1],
                         output_dim=self.num_classes,
                         seq_len=ts.features.shape[2]).to(default_device())
        self.model_for_inference = self.model
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = self._get_loss_metric(ts)
        return loss_fn, optimizer
