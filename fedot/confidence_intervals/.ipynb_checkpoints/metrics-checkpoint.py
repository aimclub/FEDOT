import numpy as np

def quantile_loss(y_true, y_pred, quantile=0.5):
    res = np.array(y_true)-np.array(y_pred)
    metrics=np.empty(shape = [0])
    for x in res:
        if x>=0:
            metrics = np.append(metrics, quantile*x)
        else:
            metrics = np.append(metrics,(quantile-1)*x)
    return np.mean(metrics)

def interval_metric(values,up,low):
    values = np.array(values)
    up = np.array(up)
    low = np.array(low)
    lv = len(values)
    if lv != len(up) or lv != len(low):
        print('ERROR! Arrays have different length!')
    for i in range(lv):
        if up[i]<low[i]:
            print('ERROR! Confidence intervals are not correct!')
            break
    else:    
        u = 0
        l = 0
        for i in range(lv):
            if values[i] > up[i]:
                u +=1
            if values[i] < low[i]:
                l+=1
        gap = up-low
        average_gap = gap.mean()
        ans = {'up_error': u,
               'low_error': l,
               'error' : l+u,
               'percent_up_error':u/lv,
               'percent_low_error': l/lv,
               'percent_error': (u+l)/lv,
               'average_gap': average_gap}
        
        return ans