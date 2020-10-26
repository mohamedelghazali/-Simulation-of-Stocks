import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm

def get_simulation(ticker, name):
    data = pd.DataFrame()
    data[ticker] = wb.DataReader(ticker, data_source='yahoo', start='2007-1-1')['Adj Close']


    log_returns = np.log(1 + data.pct_change())

    u = log_returns.mean()

    # Essentially this is how far stock prices are spread out from the mean
    var = log_returns.var()

    # This is the change the average value in our stock prices over time.
    drift = u - (0.5 * var)

    # This is a measure of the dispersion of the stock prices. 
    stdev = log_returns.std()

    t_intervals = 365
    iterations = 10

    # create the random potential future daily returns for each day. 
       
    daily_returns = np.exp(drift.values + stdev.values * norm.ppf(np.random.rand(t_intervals, iterations)))
    S0 = data.iloc[-1]
    
    #  using np.zeros_like to create a numpy array 
     

    price_list = np.zeros_like(daily_returns)
    price_list[0] = S0

    for t in range(1, t_intervals):
        price_list[t] = price_list[t - 1] * daily_returns[t]
    plt.figure(figsize=(10,6))
    plt.title("1 Year Monte Carlo Simulation for " + name)
    plt.ylabel("Price (P)")
    plt.xlabel("Time (Days)")
    plt.plot(price_list)
    plt.savefig('tempplot.png')
    plt.show()

get_simulation("UU.L", "United Utilities")
