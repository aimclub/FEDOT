import numpy as np
import pandas as pd


def filter_detecting_boundaries(detecting_boundaries):
    _detecting_boundaries = []
    for couple in detecting_boundaries.copy():
        if len(couple) != 0:
            _detecting_boundaries.append(couple)
    detecting_boundaries = _detecting_boundaries
    return detecting_boundaries


def single_detecting_boundaries(target_series,
                                target_list_ts,
                                predicted_labels,
                                share,
                                window_width,
                                anomaly_window_destination,
                                intersection_mode):
    """
    Extract detecting_boundaries from series or list of timestamps
    """

    if (target_series is not None) and (target_list_ts is not None):
        raise ValueError('Cannot perform boundaries extraction from both the series and the list of timestamps')
    elif target_series is not None:
        target_timestamps = target_series[target_series == 1].index
    elif target_list_ts is not None:
        if len(target_list_ts) == 0:
            return [[]]
        else:
            target_timestamps = target_list_ts
    else:
        raise ValueError('Cannot perform boundaries extraction: should extract from series or list of timestamps')

    detecting_boundaries = []
    td = pd.Timedelta(window_width) if window_width is not None else pd.Timedelta(
        (predicted_labels.index[-1] - predicted_labels.index[0]) / (len(target_timestamps) + 1) * share)
    for val in target_timestamps:
        if anomaly_window_destination == 'lefter':
            detecting_boundaries.append([val - td, val])
        elif anomaly_window_destination == 'righter':
            detecting_boundaries.append([val, val + td])
        elif anomaly_window_destination == 'center':
            detecting_boundaries.append([val - td / 2, val + td / 2])
        else:
            raise ValueError('Parameter anomaly_window_destination should be either "lefter", "righter" or "center"')

    # block for resolving intersection problem:
    # important to watch right boundary to be never included to avoid windows intersection
    if len(detecting_boundaries) == 0:
        return detecting_boundaries

    new_detecting_boundaries = detecting_boundaries.copy()
    intersection_count = 0
    for i in range(len(new_detecting_boundaries) - 1):
        if new_detecting_boundaries[i][1] >= new_detecting_boundaries[i + 1][0]:
            # transform print to list of intersections
            intersection_count += 1
            if intersection_mode == 'cut left window':
                new_detecting_boundaries[i][1] = new_detecting_boundaries[i + 1][0]
            elif intersection_mode == 'cut right window':
                new_detecting_boundaries[i +
                                         1][0] = new_detecting_boundaries[i][1]
            elif intersection_mode == 'cut both':
                _a = new_detecting_boundaries[i][1]
                new_detecting_boundaries[i][1] = new_detecting_boundaries[i + 1][0]
                new_detecting_boundaries[i + 1][0] = _a
            else:
                raise Exception("choose the intersection_mode")
    # print(f'There are {intersection_count} intersections of scoring windows')
    detecting_boundaries = new_detecting_boundaries.copy()
    return detecting_boundaries


def check_errors(my_list):
    """
    Check format of input true data

    Parameters
    ----------
    my_list - uniform format of true (See evaluating.evaluating)

    Returns
    ----------
    mx : depth of list, or variant of processing
    """
    assert isinstance(my_list, list)
    mx = 1
    #     ravel = []
    level_list = {}

    def check_error(my_list):
        return not (
            (all(isinstance(my_el, list) for my_el in my_list)) or
            (all(isinstance(my_el, pd.Series) for my_el in my_list)) or
            (all(isinstance(my_el, pd.Timestamp) for my_el in my_list))
        )

    def recurse(my_list, level=1):
        nonlocal mx
        nonlocal level_list

        if check_error(my_list):
            raise Exception(
                f"Non uniform data format in level {level}: {my_list}")

        if level not in level_list.keys():
            level_list[level] = []  # for checking format

        for my_el in my_list:
            level_list[level].append(my_el)
            if isinstance(my_el, list):
                mx = max([mx, level + 1])
                recurse(my_el, level + 1)

    recurse(my_list)
    for level in level_list:
        if check_error(level_list[level]):
            raise Exception(
                f"Non uniform data format in level {level}: {my_list}")

    if 3 in level_list:
        for el in level_list[2]:
            if not ((len(el) == 2) or (len(el) == 0)):
                raise Exception(
                    f"Non uniform data format in level {2}: {my_list}")
    return mx


def extract_cp_confusion_matrix(
        detecting_boundaries,
        predicted_labels,
        point=0,
        binary=False):
    """
    predicted_labels: pd.Series

    point=None for binary case
    Returns
    ----------
    dict: TPs: dict of numer window of [t1,t_cp,t2]
    FPs: list of timestamps
    FNs: list of numer window
    """
    _detecting_boundaries = []
    for couple in detecting_boundaries.copy():
        if len(couple) != 0:
            _detecting_boundaries.append(couple)
    detecting_boundaries = _detecting_boundaries

    times_pred = predicted_labels[predicted_labels.dropna(
    ) == 1].sort_index().index

    my_dict = {}
    my_dict['TPs'] = {}
    my_dict['FPs'] = []
    my_dict['FNs'] = []

    if len(detecting_boundaries) != 0:
        my_dict['FPs'].append(
            times_pred[times_pred < detecting_boundaries[0][0]])  # left
        for i in range(len(detecting_boundaries)):
            times_pred_window = times_pred[(times_pred >= detecting_boundaries[i][0]) &
                                           (times_pred <= detecting_boundaries[i][1])]
            times_predicted_labels_in_window = predicted_labels[
                detecting_boundaries[i][0]:detecting_boundaries[i][1]].index
            if len(times_pred_window) == 0:
                if not binary:
                    my_dict['FNs'].append(i)
                else:
                    my_dict['FNs'].append(times_predicted_labels_in_window)
            else:
                my_dict['TPs'][i] = [detecting_boundaries[i][0],
                                     # attention
                                     times_pred_window[point] if not binary else times_pred_window,
                                     detecting_boundaries[i][1]]
                if binary:
                    my_dict['FNs'].append(
                        times_predicted_labels_in_window[~times_predicted_labels_in_window.isin(times_pred_window)])
            if len(detecting_boundaries) > i + 1:
                my_dict['FPs'].append(times_pred[(times_pred > detecting_boundaries[i][1]) & (
                    times_pred < detecting_boundaries[i + 1][0])])

        my_dict['FPs'].append(
            times_pred[times_pred > detecting_boundaries[i][1]])  # right
    else:
        my_dict['FPs'].append(times_pred)

    if len(my_dict['FPs']) > 1:
        my_dict['FPs'] = np.concatenate(my_dict['FPs'])
    elif len(my_dict['FPs']) == 1:
        my_dict['FPs'] = my_dict['FPs'][0]
    if len(my_dict['FPs']) == 0:  # not elif on purpose
        my_dict['FPs'] = []

    if binary:
        if len(my_dict['FNs']) > 1:
            my_dict['FNs'] = np.concatenate(my_dict['FNs'])
        elif len(my_dict['FNs']) == 1:
            my_dict['FNs'] = my_dict['FNs'][0]
        if len(my_dict['FNs']) == 0:  # not elif on purpose
            my_dict['FNs'] = []
    return my_dict


def confusion_matrix(true, predicted_labels):
    target_ = true == 1
    predicted_labels_ = predicted_labels == 1
    TP = (target_ & predicted_labels_).sum()
    TN = (~target_ & ~predicted_labels_).sum()
    FP = (~target_ & predicted_labels_).sum()
    FN = (target_ & ~predicted_labels_).sum()
    return TP, TN, FP, FN


def single_average_delay(
        detecting_boundaries,
        predicted_labels,
        anomaly_window_destination,
        clear_anomalies_mode):
    """
    anomaly_window_destination: 'lefter', 'righter', 'center'. Default='right'
    """
    detecting_boundaries = filter_detecting_boundaries(detecting_boundaries)
    point = 0 if clear_anomalies_mode else -1
    dict_cp_confusion = extract_cp_confusion_matrix(
        detecting_boundaries, predicted_labels, point=point)

    missing = 0
    detectHistory = []
    all_target_anom = 0
    FP = 0

    FP += len(dict_cp_confusion['FPs'])
    missing += len(dict_cp_confusion['FNs'])
    all_target_anom += len(dict_cp_confusion['TPs']) + \
        len(dict_cp_confusion['FNs'])

    if anomaly_window_destination == 'lefter':
        def average_time(output_cp_cm_tp):
            return output_cp_cm_tp[2] - output_cp_cm_tp[1]
    elif anomaly_window_destination == 'righter':
        def average_time(output_cp_cm_tp):
            return output_cp_cm_tp[1] - output_cp_cm_tp[0]
    elif anomaly_window_destination == 'center':
        def average_time(output_cp_cm_tp):
            return output_cp_cm_tp[1] - (output_cp_cm_tp[0] +
                                         (output_cp_cm_tp[2] - output_cp_cm_tp[0]) / 2)
    else:
        raise Exception("Choose anomaly_window_destination")

    for fp_case_window in dict_cp_confusion['TPs']:
        detectHistory.append(
            average_time(
                dict_cp_confusion['TPs'][fp_case_window]))
    return missing, detectHistory, FP, all_target_anom


# def my_scale(fp_case_window=None,
#              A_tp=1,
#              A_fp=0,
#              koef=1,
#              detalization=1000,
#              clear_anomalies_mode=True,
#              plot_figure=False):
#     """
#     ts - segment on which the window is applied
#     """
#     x = np.linspace(-np.pi / 2, np.pi / 2, detalization)
#     x = x if clear_anomalies_mode else x[::-1]
#     y = (A_tp - A_fp) / 2 * -1 * np.tanh(koef * x) / \
#         (np.tanh(np.pi * koef / 2)) + (A_tp - A_fp) / 2 + A_fp
#     if not plot_figure:
#         event = int((fp_case_window[1] - fp_case_window[0]) /
#                     (fp_case_window[-1] - fp_case_window[0]) * detalization)
#         if event >= len(x):
#             event = len(x) - 1
#         score = y[event]
#         return score
#     else:
#         return y


def single_evaluate_nab(detecting_boundaries,
                        predicted_labels,
                        table_of_val=None,
                        clear_anomalies_mode=True,
                        scale="improved",
                        scale_val=1  # TODO
                        ):
    """

    detecting_boundaries: list of list of two float values
                The list of lists of left and right boundary indices
                for scoring results of labeling if empty. Can be [[]], or [[],[t1,t2],[]]
    table_of_coef: pandas array (3x4) of float values
                Table of coefficients for NAB score function
                indices: 'Standard','LowFP','LowFN'
                columns:'A_tp','A_fp','A_tn','A_fn'

    scale_func {default}, improved
    недостатки scale_func default  -
    1 - зависит от относительного шага, а это значит, что если
    слишком много точек в scoring window то перепад будет слишком
    жестким в середение.
    2-   то самая левая точка не равно  Atp, а права не равна Afp
    (особенно если пррименять расплывающую множитель)

    clear_anomalies_mode тогда слева от границы Atp срправа Afp,
    иначе fault mode, когда слева от границы Afp срправа Atp
    """

    #     def sigm_scale(len_ts, A_tp, A_fp, koef=1):
    #         x = np.arange(-int(len_ts/2), len_ts - int(len_ts/2))

    #         x = x if clear_anomalies_mode else x[::-1]
    #         y = (A_tp-A_fp)*(1/(1+np.exp(5*x*koef))) + A_fp
    #         return y
    #     def my_scale(len_ts,A_tp,A_fp,koef=1):
    #         """ts - участок на котором надо жахнуть окно """
    #         x = np.linspace(-np.pi/2,np.pi/2,len_ts)
    #         x = x if clear_anomalies_mode else x[::-1]
    #         # Приведение если неравномерный шаг.
    #         #x_new = x_old * ( np.pi / (x_old[-1]-x_old[0])) - x_old[0]*( np.pi / (x_old[-1]-x_old[0])) - np.pi/2
    #         y = (A_tp-A_fp)/2*-1*np.tanh(koef*x)/(np.tanh(np.pi*koef/2)) + (A_tp-A_fp)/2 + A_fp
    #         return y

    # if scale == "improved":
    #     scale_func = my_scale
    # #     elif scale_func == "default":
    # #         scale_func = sigm_scale
    # else:
    #     raise Exception("choose the scale_func")

    # filter
    detecting_boundaries = filter_detecting_boundaries(detecting_boundaries)

    if table_of_val is None:
        table_of_coef = pd.DataFrame([[1.0, -0.11, 1.0, -1.0],
                                      [1.0, -0.22, 1.0, -1.0],
                                      [1.0, -0.11, 1.0, -2.0]])
        table_of_coef.index = ['Standard', 'LowFP', 'LowFN']
        table_of_coef.index.name = "Metric"
        table_of_coef.columns = ['A_tp', 'A_fp', 'A_tn', 'A_fn']

    point = 0 if clear_anomalies_mode else -1
    dict_cp_confusion = extract_cp_confusion_matrix(
        detecting_boundaries, predicted_labels, point=point)

    Scores, Scores_perfect, Scores_null = [], [], []
    for profile in ['Standard', 'LowFP', 'LowFN']:
        A_tp = table_of_coef['A_tp'][profile]
        A_fp = table_of_coef['A_fp'][profile]
        A_fn = table_of_coef['A_fn'][profile]

        score = 0
        score += A_fp * len(dict_cp_confusion['FPs'])
        score += A_fn * len(dict_cp_confusion['FNs'])
        for fp_case_window in dict_cp_confusion['TPs']:
            set_times = dict_cp_confusion['TPs'][fp_case_window]
            score += scale(set_times, A_tp, A_fp, koef=scale_val)

        Scores.append(score)
        Scores_perfect.append(len(detecting_boundaries) * A_tp)
        Scores_null.append(len(detecting_boundaries) * A_fn)

    return np.array([np.array(Scores), np.array(
        Scores_null), np.array(Scores_perfect)])
