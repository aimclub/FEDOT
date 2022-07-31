import numpy as np
# Pipeline and nodes
import pandas as pd
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.utilities.ts_gapfilling import ModelGapFiller


# can still not handle gaps in first(fwrd-fill) and last(bidirect-fill) 7 rows
def fedot_frwd_bi(data_w_nan, fedot_window):
    # fill the nan with '-100' so fedot can work with it
    df_w_nan_copy = data_w_nan.fillna(-100)

    # Got univariate time series as numpy array
    time_series = np.array(df_w_nan_copy['Values'])

    # create a pipeline and defines the values which count as gaps
    pipeline = get_simple_ridge_pipeline(fedot_window)
    model_gapfiller = ModelGapFiller(gap_value=-100.0,
                                     pipeline=pipeline)

    # ----------
    # Filling in the gaps
    # ----------
    without_gap_forward = model_gapfiller.forward_filling(time_series)
    without_gap_bidirect = model_gapfiller.forward_inverse_filling(time_series)

    return without_gap_forward, without_gap_bidirect


def get_simple_ridge_pipeline(fedot_window):
    node_lagged = PrimaryNode('lagged')
    node_lagged.custom_params = {'window_size': fedot_window}

    node_final = SecondaryNode('ridge', nodes_from=[node_lagged])
    pipeline = Pipeline(node_final)

    return pipeline


if __name__ == '__main__':
    df = pd.read_csv('time_series_withgaps.csv', names=['Values'])
    df = df.head(150)
    fedot_frwd_bi(df, 10)