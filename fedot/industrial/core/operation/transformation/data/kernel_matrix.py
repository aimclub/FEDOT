from scipy.spatial.distance import pdist, squareform
from fedot.industrial.core.distance.pdist import torch_pdist
from fedot.industrial.core.architecture.preprocessing.data_convertor import DataConverter, TensorConverter
from fedot.industrial.core.architecture.settings.computational import backend_methods as np
import torch


def colorise(distance_matrix):
    """
    Instead of a binarization, we colorize the distance matrix by scaling the values to [0, 255].
    Also convert the matrix to uint8 type – this is the format that PIL uses to display images.
    """
    min_value, max_value = distance_matrix.min(), distance_matrix.max()
    return np.round((distance_matrix - min_value) / (max_value - min_value + 1e-8) * 255).astype('uint8')


class TSTransformer:
    def __init__(self, time_series, rec_metric=None):
        self.time_series = DataConverter(
            data=time_series).convert_to_2d_array()
        self.recurrence_matrix = None
        # TODO add threshold for other metrics
        self.threshold_baseline = [0.95, 0.7]
        self.min_signal_ratio = 0.6
        self.max_signal_ratio = 0.85
        if rec_metric is not None:
            self.rec_metric = rec_metric
        else:
            self.rec_metric = 'euclidean'

    def ts_to_recurrence_matrix(self,
                                threshold=None):
        distance_vec = pdist(metric=self.rec_metric, X=self.time_series.T)
        distance_matrix = squareform(distance_vec)
        distance_matrix = 1 - (distance_matrix / distance_matrix.max())
        self.recurrence_matrix = self.binarization(
            distance_matrix, threshold=threshold)
        return self.recurrence_matrix

    def ts_to_3d_recurrence_matrix(self):
        cosine_matrix = pdist(metric='cosine', X=self.time_series.T)
        euclidean_matrix = pdist(metric='euclidean', X=self.time_series.T)
        canberra_matrix = pdist(metric='canberra', X=self.time_series.T)
        squared_matrices = list(
            map(squareform, [cosine_matrix, euclidean_matrix, canberra_matrix]))
        dimensions = list(map(colorise, squared_matrices))
        self.recurrence_matrix = np.stack(dimensions, axis=0)
        return self.recurrence_matrix

    def binarization(self, distance_matrix, threshold):
        best_threshold_flag = False
        recurrence_matrix = None
        best_ratio = None
        if threshold is None:
            for threshold_baseline in self.threshold_baseline:
                tmp = (distance_matrix >= threshold_baseline).astype(float)
                signal_ratio = np.mean(tmp == 0)
                if self.min_signal_ratio < signal_ratio < self.max_signal_ratio:
                    if not best_threshold_flag or signal_ratio > best_ratio:
                        best_ratio = signal_ratio
                        recurrence_matrix = tmp
                        best_threshold_flag = True
            if not best_threshold_flag:
                threshold = self.threshold_baseline[0]
                recurrence_matrix = (distance_matrix >= threshold).astype(float)
        else:
            recurrence_matrix = (distance_matrix >= threshold).astype(float)
        return recurrence_matrix

    def get_recurrence_metrics(self):
        if self.recurrence_matrix is None:
            return self.ts_to_recurrence_matrix()


class TorchTSTransformer:
    def __init__(self, time_series, rec_metric='cosine', p=3, device=None):
        """
        Transformer for recurrence matrix generation.
        Args:
            time_series: torch.Tensor of shape (features, timesteps)
            rec_metric: one of ['cosine', 'euclidean', 'canberra']
        """
        self.time_series = TensorConverter(
            data=time_series).convert_to_2d_tensor()
        self.device = device or self.time_series.device
        self.time_series = self.time_series.to(self.device)
        self.recurrence_matrix = None
        self.threshold_baseline = [0.95, 0.7]
        self.min_signal_ratio = 0.6
        self.max_signal_ratio = 0.85
        self.rec_metric = rec_metric
        self.p = p

    def ts_to_recurrence_matrix(self, threshold=None):
        distance_matrix = torch_pdist(self.time_series.T, metric=self.rec_metric, p=self.p)
        distance_matrix = 1 - (distance_matrix / distance_matrix.max())
        self.recurrence_matrix = self.binarization(distance_matrix, threshold)
        return self.recurrence_matrix

    def binarization(self, distance_matrix: torch.Tensor, threshold: float = None):
        if threshold is not None:
            return (distance_matrix >= threshold).float()
        ths = torch.tensor(self.threshold_baseline,
                           device=distance_matrix.device,
                           dtype=distance_matrix.dtype)
        tmp = (distance_matrix.unsqueeze(0) >= ths[:, None, None]).float()
        signal_ratio = torch.mean((tmp == 0).float(), dim=(1, 2))

        mask = (signal_ratio > float(self.min_signal_ratio)) & (signal_ratio < float(self.max_signal_ratio))
        if mask.any():
            neg_one = torch.tensor(-1.0,
                                   device=distance_matrix.device,
                                   dtype=signal_ratio.dtype)
            candidate_scores = torch.where(mask, signal_ratio, neg_one)
            best_idx = torch.argmax(candidate_scores)
            return tmp[best_idx].float()
        else:
            return tmp[0].float()

    def _colorise_torch(self, matrix: torch.Tensor) -> torch.Tensor:
        matrix = matrix.to(torch.float64)
        min_val = matrix.min()
        max_val = matrix.max()
        normalized = (matrix - min_val) / (max_val - min_val + 1e-8)
        rounded = torch.floor(normalized * 255 + 0.5)
        return rounded.to(torch.uint8)

    def ts_to_3d_recurrence_matrix(self):
        cos = torch_pdist(self.time_series.T, metric='cosine')
        euc = torch_pdist(self.time_series.T, metric='euclidean')
        can = torch_pdist(self.time_series.T, metric='canberra')
        colorised = [self._colorise_torch(m) for m in [cos, euc, can]]
        stacked = torch.stack(colorised, dim=0)
        return stacked
