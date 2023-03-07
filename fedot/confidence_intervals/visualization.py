import matplotlib.pyplot as plt


def plot_confidence_intervals(horizon,
         up_predictions,
         low_predictions,
         model_forecast,
         up,
         low,
         ts,
         up_int = True,
         low_int = True,
         forecast = True,
         history = True,
         up_train = True,
         low_train = True,
         ts_test = None):
        
    r = range(1,horizon+1)
                
    fig,ax = plt.subplots()
    fig.set(figwidth = 15,figheight = 7)
        
    for i in range(len(up_predictions)):
        if i==0:
            if up_train:
                ax.plot(r,up_predictions[i], color = 'yellow',label = 'preds for up train')
            if low_train:
                ax.plot(r,low_predictions[i], color = 'pink',label = 'preds for low train')
        else:
            if up_train:
                ax.plot(r,up_predictions[i], color = 'yellow')
            if low_train:
                ax.plot(r,low_predictions[i], color = 'pink')   
    if up_int:
        ax.plot(r,up, color = 'blue', label  = 'Up',marker= '.')
    if low_int:
        ax.plot(r,low, color = 'green', label  = 'Low',marker = '.')
    if forecast:
        ax.plot(r,model_forecast, color = 'red', label = 'Forecast')
    if ts_test is not None:
        ax.plot(r,ts_test,color = 'black', label = 'Actual TS')
    plt.legend()
        
    
    if history:
        fig1,ax1 = plt.subplots()
        
        fig1.set(figwidth = 15,figheight = 7)
       
        train_range = range(len(ts))
        test_range = range(len(ts),len(ts)+horizon)
        
        ax1.plot(train_range,ts, color = 'gray',label = 'Train ts')
        ax1.plot(test_range, up,color = 'blue', label = 'Up')
        ax1.plot(test_range, low,color = 'green', label = 'Low')
        ax1.plot(test_range,model_forecast, color = 'red', label = 'Forecast')
        if ts_test is not None:
            ax1.plot(test_range, ts_test,color = 'black', label = 'Actual TS')  
    plt.legend();     