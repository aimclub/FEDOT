import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from tqdm import tqdm

from fedot.industrial.core.operation.transformation.data.kernel_matrix import colorise
from fedot.industrial.tools.explain.distances import DistanceTypes


class Explainer:
    def __init__(self, model, features, target):
        self.model = model
        self.features = features
        self.target = target

    def explain(self, **kwargs):
        pass

    @staticmethod
    def predict_proba(model, features, target):
        if hasattr(model, 'manager') and hasattr(model.manager, 'solver'):
            model.manager.solver.test_features = None
            base_proba_ = model.predict_proba(predict_data=(features, target))
        else:
            base_proba_ = model.predict_proba(X=features)
        return base_proba_


class RecurrenceExplainer(Explainer):
    def __init__(self, model, features, target):
        super().__init__(model, features, target)
        self.rec_matrix_by_cls = {}
        self.aggregate_func = {'mean': np.mean,
                               'sum': np.sum}

    def _get_recurrence_matrix(self):
        recurrence_extractor = [node.fitted_operation for node in self.model.solver.current_pipeline.nodes if
                                node.name.__contains__('recurrence_extractor')][0]
        return recurrence_extractor

    def explain(self, **kwargs):
        rec_matrix = self._get_recurrence_matrix().predict if len(self.features) <= 3 else self.features
        for classes in np.unique(self.target):
            cls_idx = np.where(self.target == classes)[0]
            self.rec_matrix_by_cls.update({classes: rec_matrix[cls_idx, :, :, :]})

    def visual(self, metric: str = 'mean', name: str = 'test', threshold: float = None):
        matplotlib.use('TkAgg')
        for classes, rec_matrix in self.rec_matrix_by_cls.items():
            aggregated_rec_matrix = self.aggregate_func[metric](rec_matrix, axis=0)
            aggregated_rec_matrix = colorise(aggregated_rec_matrix)
            plt.imshow(aggregated_rec_matrix.T)
            plt.colorbar()
            plt.savefig(f'recurrence_matrix_for_{name}_dataset_cls_{classes}.png')
            plt.close()


class PointExplainer(Explainer):
    def __init__(self, model, features, target):
        super().__init__(model, features, target)
        self.picked_target = None
        self.picked_feature = None

        self.scaled_vector = None
        self.window_length = None

    def explain(
            self,
            n_samples: int = 1,
            window: int = 5,
            method: str = 'rmse'):
        self.picked_feature, self.picked_target = self.select(
            self.features, self.target.flatten(), n_samples_=n_samples)
        self.scaled_vector, self.window_length = self.importance(window=window,
                                                                 method=method)

    def visual(self, threshold: int = 90, name='dataset', metric: str = None):
        self.plot_importance(thr=threshold, name=name)

    def importance(self, window=None, method='euclidean'):
        model = self.model
        part_feature_ = self.picked_feature
        part_target_ = self.picked_target
        distance_func = DistanceTypes[method]
        base_proba_ = self.predict_proba(model, part_feature_, part_target_)

        if not window:
            window_length = 0
            n_parts = part_feature_.shape[1]

            iv_scaled = self.get_vector(
                base_proba_,
                distance_func,
                model,
                n_parts,
                part_feature_,
                part_target_,
                window_length)

        else:
            window_length = part_feature_.shape[1] * window // 100
            n_parts = math.ceil(part_feature_.shape[1] / window_length)
            iv_scaled = self.get_vector(
                base_proba_,
                distance_func,
                model,
                n_parts,
                part_feature_,
                part_target_,
                window_length)

        return pd.DataFrame(iv_scaled), window_length

    def get_vector(
            self,
            base_proba_,
            distance_func,
            model,
            n_parts,
            part_feature_,
            part_target_,
            window_length):
        importance_vector_ = {cls: np.zeros(
            n_parts) for cls in np.unique(part_target_)}
        with tqdm(total=n_parts, desc='Processing points', unit='point') as pbar:
            for part in range(n_parts):
                feature_ = part_feature_.copy().values
                feature_ = self.replace_values(
                    feature_, window_len=window_length, i=part)
                proba_new = self.predict_proba(model, feature_, part_target_)

                distance_dict = {}
                for idx, cls in enumerate(part_target_):
                    if cls not in distance_dict:
                        distance_dict[cls] = []
                    distance = distance_func(
                        np.array(
                            base_proba_[idx]).ravel(), np.array(
                            proba_new[idx]).ravel())
                    distance_dict[cls].append(distance)
                for cls in distance_dict:
                    importance_vector_[cls][part] = np.mean(distance_dict[cls])
                pbar.update(1)
        return importance_vector_

    @staticmethod
    def replace_values(features: np.ndarray, window_len: int, i: int):
        if window_len:
            for idx, ts in enumerate(features):
                mean_ts = ts.mean()
                features[idx, i * window_len:(i + 1) * window_len] = mean_ts
        else:
            for idx, ts in enumerate(features):
                mean_ts = ts.mean()
                features[idx, i] = mean_ts
        return features

    @staticmethod
    def select(features_, target_, n_samples_: int = 3):
        selected_df = pd.DataFrame()
        selected_target = np.array([])
        if not isinstance(features_, pd.DataFrame):
            features_ = pd.DataFrame(features_)
        df = features_.copy()
        df['target'] = target_
        for class_label in np.unique(target_):
            class_samples = df[df['target'] == class_label].sample(
                n=n_samples_, replace=False)
            selected_df = pd.concat([selected_df, class_samples.iloc[:, :-1]])
            selected_target = np.concatenate(
                [selected_target, class_samples['target'].to_numpy()])

        return selected_df, selected_target

    def plot_importance(self, thr=90, name='dataset'):
        feature, target = self.picked_feature, self.picked_target
        vector_df = self.scaled_vector
        window = self.window_length
        # filter by threshold value for each class
        threshold_ = {cls: np.percentile(
            vector_df[cls], thr) for cls in np.unique(target)}
        importance_vector_filtered_ = {
            cls: np.where(
                vector_df[cls] > threshold_[cls],
                vector_df[cls],
                0) for cls in np.unique(target)}
        vector_df = pd.DataFrame(importance_vector_filtered_)
        n_classes = len(np.unique(target))
        fig, axs = plt.subplots(n_classes, 1, figsize=(
            10, 5 if n_classes < 6 else 5 * n_classes // 2))
        fig.suptitle(f'Importance of points for {name} dataset')

        # Color bar definition
        cbar_ax = fig.add_axes([1, 0.3, 0.01, 0.5])
        cmap = plt.get_cmap('Reds')
        norm = Normalize(vmin=vector_df.min().min(),
                         vmax=vector_df.max().max())

        scal_map = ScalarMappable(norm=norm, cmap='Reds')

        for idx, cls in enumerate(np.unique(target)):
            copy_vec = vector_df[cls].copy()
            if not window:
                # every 10% of length
                x_ticks = np.arange(0, len(feature.iloc[idx, :]), len(
                    feature.iloc[idx, :]) // 10)
                for dot_idx, dot in enumerate(copy_vec):
                    axs[idx].axvline(dot_idx, color=cmap(norm(dot)))
                    axs[idx].set_xticks(x_ticks)

            else:
                # ticks with window step
                x_ticks = [
                    *np.arange(0, len(feature.iloc[idx, :]), window), len(feature.iloc[idx, :])]
                for span_idx, dot in enumerate(copy_vec):
                    left = span_idx * window
                    right = span_idx * window + window if span_idx * window + \
                        window < len(feature.iloc[idx, :]) else len(
                            feature.iloc[idx, :])
                    axs[idx].axvspan(left, right, color=cmap(norm(dot)))
                    top = axs[idx].get_ylim()[1]
                    axs[idx].text(
                        left, top, f'{dot:.2f}', fontsize=8, color='white')
                    axs[idx].set_xticks(x_ticks)

            mean_value = feature.iloc[idx, :].mean()
            axs[idx].plot([0, len(feature.iloc[idx, :])], [
                mean_value, mean_value], color='black', linestyle='--', label='mean')
            class_indexes = np.where(target == cls)[0]
            for class_idx in class_indexes:
                axs[idx].plot(feature.iloc[class_idx, :],
                              color='dodgerblue', label=f'class-{cls}')
            axs[idx].text(len(feature.iloc[idx, :]) - 1, mean_value,
                          f'mean={mean_value:.2f}', fontsize=10)
            axs[idx].set_title(f'Class: {cls}')
        plt.colorbar(scal_map, cax=cbar_ax)
        plt.tight_layout()
        plt.show()
