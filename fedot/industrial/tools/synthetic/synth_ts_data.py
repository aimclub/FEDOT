from fedot.industrial.core.operation.transformation.splitter import TSTransformer
from fedot.industrial.tools.synthetic.anomaly_generator import AnomalyGenerator
from fedot.industrial.tools.synthetic.ts_generator import TimeSeriesGenerator


class SynthTimeSeriesData:
    def __init__(self, config: dict):
        """
        Args:
            config: dict with config for synthetic ts_data.
        """
        self.config = config

    def generate_ts(self, ):
        """
        Method to generate synthetic time series

        Returns:
            synthetic time series data.

        """
        return TimeSeriesGenerator(self.config).get_ts()

    def generate_anomaly_ts(self,
                            ts_data,
                            plot: bool = False,
                            overlap: float = 0.1):
        """
        Method to generate anomaly time series

        Args:
            ts_data: either np.ndarray or dict with config for synthetic ts_data.
            overlap: float, ``default=0.1``. Defines the maximum overlap between anomalies.
            plot: if True, plot initial and modified time series data with rectangle spans of anomalies.

        Returns:
            returns initial time series data, modified time series data and dict with anomaly intervals.

        """

        generator = AnomalyGenerator(config=self.config)
        init_synth_ts, mod_synth_ts, synth_inters = generator.generate(time_series_data=ts_data,
                                                                       plot=plot, overlap=overlap)

        return init_synth_ts, mod_synth_ts, synth_inters

    def split_ts(self,
                 time_series,
                 binarize: bool = False,
                 plot: bool = True) -> tuple:
        """
        Method to split time series with anomalies into features and target.

        Args:
            time_series (npp.array):
            binarize: if True, target will be binarized. Recommended for classification task if classes are imbalanced.
            plot: if True, plot initial and modified time series data with rectangle spans of anomalies.

        Returns:
            features (pd.DataFrame) and target (np.array).

        """

        features, target = TSTransformer().transform_for_fit(
            plot=plot, binarize=binarize, series=time_series, anomaly_dict=self.config)
        return features, target
