class CouponBond:

    def __init__(self, principal, rate, maturity, interest_rate):
        self.principal = principal
        # coupon rate, interest the bond pays annually
        self.rate = rate/100
        self.maturity = maturity
        # market interest rate
        self.interest_rate = interest_rate/100

    # present value of future cashflow
    def present_value(self, x, n):
        return x/(1+self.interest_rate)**n
        # continuous model 
        #  return x*exp(-self.interest_rate*n)

    def calculate_price(self):
        # discount the coupon payments
        price = 0
        for t in range(1,self.maturity+1):
            # present value of each cashflow
            price+= self.present_value(x=self.principal * self.rate, n=t)

        # discount principle amount
        price+= self.present_value(self.principal, self.maturity)

        return price
    
if __name__ == '__main__':

    bond = CouponBond(1000, 10, 3, 4)

    print("Price of the bond in dollars: %.2f" %bond.calculate_price())