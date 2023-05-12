import yfinance as yf
import datetime
import pandas as pd
import numpy as np

def download_data(stock, start_date, end_date):
    data = {}
    ticker = yf.download(stock, start_date, end_date)
    data[stock] = ticker['Adj Close']
    return pd.DataFrame(data)

class VaRMonteCarlo: 
    def __init__(self, S, mu, sigma, c, n, iterations):
        # this is the value of our investment at t=0
        self.S = S
        self.mu = mu
        self.sigma = sigma
        self.c = c
        self.n = n
        self.iterations = iterations

    def simulation(self):
        rand = np.random.normal(0, 1, [1, self.iterations])
        
        # equation for the S(t) stock price
        # the random walk of our initial investment
        stock_price = self.S * np.exp(self.n * (self.mu-0.5*self.sigma**2)
                                      + self.sigma*np.sqrt(self.n)*rand)

        # we have to sort the stock prices to determine the percentile
        stock_price = np.sort(stock_price)

        # it depends on the confidence level: 95% -> 5 and 99% -> 1
        percentile = np.percentile(stock_price, (1-self.c)*100)

        return self.S - percentile

if __name__ == '__main__':
    S = 1e6 # this is the investment (stocks or whatever)
    c = 0.99 # confidenve level this time it is 95%
    n = 1 # 1 day
    iterations = 1000 # number of paths in the Monte-Carlo simulation

    # historical data to approximate mean and standard deviation
    start = datetime.datetime(2014, 1, 1)
    end = datetime.datetime(2018, 1, 1)

    stock_data = download_data('C', start, end)

    stock_data['returns'] = np.log(stock_data['C']/stock_data['C'].shift(1))
    stock_data = stock_data[1:]

    # we can assume daily returns to be normally distributed
    # mean and variance can describe the process
    mu = np.mean(stock_data['returns'])
    sigma = np.std(stock_data['returns'])

    model = VaRMonteCarlo(S, mu, sigma, c, n, iterations)
    print('Value at risk with Monte-Carlo simulation: $%0.2f' % model.simulation())

    