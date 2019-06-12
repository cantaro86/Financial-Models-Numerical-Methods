class Option_param():
    """
    Option class wants the option parameters:
    S0 = current stock price
    K = Strike price
    T = time to maturity
    type_ = "Eu" for European or "Am" for American
    """
    def __init__(self, S0=15, K=15, T=1, type_="Eu"):
        self.S0 = S0
        self.K = K
        self.T = T
        if (type_=="Eu" or type_=="Am"):
            self.type_ = type_
        else: 
            raise ValueError("invalid type. Set 'Eu' or 'Am'")



class Diffusion_param():
    """
    Class collecting diffusion parameters (1-D):
    r = risk free constant rate
    sig = constant diffusion coefficient
    mu = constant drift 
    """
    def __init__(self, r=0.1, sig=0.2, mu=0.1):
        self.r = r
        self.mu = mu
        if (sig<=0):
            raise ValueError("sig must be positive")
        else:
            self.sig = sig
            
