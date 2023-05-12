import numpy as np


class OptionPricing:

    def __init__(self, S0, K, T, rf, sigma, iterations):
        self.S0 = S0
        self.K = K
        self.T = T
        self.rf = rf
        self.sigma = sigma
        self.iterations = iterations

    def call_option_simulation(self):
        # we have 2 columns: first with 0s the second column will store the payoff
        # we need the first columns of 0s: payoff function is max(0, S-E) for call option
        option_data = np.zeros([self.iterations, 2])

        # dimensions: 1 dimensional array with as many items as the iterations
        rand = np.random.normal(0,1, [1, self.iterations])

        # equation for the S(t) stock price at T
        stock_price = self.S0*np.exp(self.T * (self.rf - 0.5*self.sigma**2)
                                     + self.sigma*np.sqrt(self.T)*rand)
        
        # we need S-K because we have to calculate the max(S-K,0)
        option_data[:,1]= stock_price - self.K

        # average for the Monte-Carlo simulation
        # max() returns the max(0,S-K) according to the formula
        average = np.sum(np.amax(option_data, axis=1))/float(self.iterations)

        # have to use the exp(-rT) discount factor
        return np.exp(-1.0*self.rf*self.T)*average
      
    def put_option_simulation(self):
        # we have 2 columns: first with 0s the second column will store the payoff
        # we need the first columns of 0s: payoff function is max(0, S-E) for call option
        option_data = np.zeros([self.iterations, 2])

        # dimensions: 1 dimensional array with as many items as the iterations
        rand = np.random.normal(0,1, [1, self.iterations])

        # equation for the S(t) stock price at T
        stock_price = self.S0*np.exp(self.T * (self.rf - 0.5*self.sigma**2)
                                     + self.sigma*np.sqrt(self.T)*rand)
        
        # we need S-K because we have to calculate the max(S-K,0)
        option_data[:,1]=  self.K - stock_price

        # average for the Monte-Carlo simulation
        # max() returns the max(0,K-S) according to the formula
        average = np.sum(np.amax(option_data, axis=1))/float(self.iterations)

        # have to use the exp(-rT) discount factor
        return np.exp(-1.0*self.rf*self.T)*average
      

if __name__=='__main__':
    model = OptionPricing(100, 100, 1, 0.05, 0.2, 10000000)
    print('Value of call option is $%.2f' %model.call_option_simulation())
    print('Value of put option is $%.2f' %model.put_option_simulation())