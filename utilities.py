import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet

def unwrap(lst):
    ret = []
    for inLst in lst:
        ret.append(inLst[0])
    return ret

def fromSeriesToDF(df,series,IDX):
    series = df.iloc[IDX,:]
    dfSeries = series.to_frame()
    newdata = { 'ds':dfSeries.axes[0][1:].tolist(),'y':unwrap(dfSeries.values.tolist()[1:]) }
    newframe = pd.DataFrame(data=newdata)
    return newframe

def getTrainTest(df,sz,prc):
    proc = 100 - prc
    until = (proc * sz) // 100
    return df.iloc[:until],df.iloc[until:],until

def smape(actual,forecasted):
    if len(actual) != len(forecasted):
        raise Exception("Lists not equal")
    lst_size = len(actual)
    sum = 0
    for i in range(0,lst_size):
        sum += (np.absolute(forecasted[i] - actual[i]))/(np.absolute(forecasted[i]) + np.absolute(actual[i]))
    return sum / lst_size

def doForSeries(df,series,IDX,with_daily_seasonality = False,with_weekly_seasonality = False,with_yearly_seasonality = False):
    print("I AM HERE")
    newdf = fromSeriesToDF(df,series,IDX).dropna()
    df_size = newdf.shape[0]
    train_df,test_df,until = getTrainTest(newdf,df_size,30)

    my_model = Prophet(interval_width=0.95,daily_seasonality=with_daily_seasonality,weekly_seasonality=with_weekly_seasonality,yearly_seasonality=with_yearly_seasonality)
    my_model.fit(train_df)
    future_dates = my_model.make_future_dataframe(periods=df_size-until)
    forecast = my_model.predict(future_dates)
    
    forecasted_values = unwrap(forecast[['yhat']].values.tolist()[until:])
    test_values = unwrap(test_df[['y']].values.tolist())
    print("HERE")
    return test_values,forecasted_values,future_dates
    
def overlord(df,GO_UNTIL,with_daily_seasonality,with_weekly_seasonality,with_yearly_seasonality):
    ret = []
    FILE_TO_WRITE = "OUT/partial2.csv"
    FROM = 10000
    for i in range(FROM,GO_UNTIL):
        try:
            ret.append(doForSeries(df,df.iloc[i],i,with_daily_seasonality,with_weekly_seasonality,with_yearly_seasonality))
        except BaseException:
            ret.append("NaN")
        if i % 500 == 0:
            writeToFile(ret,FILE_TO_WRITE)
            print("wrote " + str(i))
    writeToFile(ret,FILE_TO_WRITE)
    return ret

def printListErrors(lst):
    for i in range(0,len(lst)):
        print(str(i+1) + " -> " + str(lst[i]))
        
def writeToFile(lst,filename):
    ln = len(lst)
    with open(filename,"w") as f:
        f.write(str(ln) + "\n")
        for nr in lst:
            f.write(str(nr) + "\n")