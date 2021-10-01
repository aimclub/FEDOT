import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def vis_predictions(path):
    """ Function plot predictions """
    folders = os.walk(path)
    print(folders)

vis_predictions('../../cases/data/time_series')
