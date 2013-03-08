import numpy as np

class Dropper:
    def __init__(self,dropout_rate=0):
        self.drop_mask = None
        self.dropout_rate = dropout_rate

    def apply_dropout(self,X):
        if (self.drop_mask is not None):
            return self.drop_mask*X
        else:
            return X

    #This will be overridden by a subclass.
    def set_dropout_mask(self,X):
        self.drop_mask = None

#Good ol' regular dropout.
class VanillaDropper(Dropper):
    def __init__(self,dropout_rate):
        Dropper.__init__(self,dropout_rate)

    def set_dropout_mask(self,X):
        if (self.dropout_rate > 0):
            self.drop_mask = np.random.rand(*X.shape) >= self.dropout_rate

#Plackett-Luce dropout.
class PLDropper(Dropper):
    def __init__(self,dropout_rate):
        Dropper.__init__(self,dropout_rate)

    def set_dropout_mask(self,X):
        if (self.dropout_rate > 0):
            self.drop_mask = np.zeros(X.shape)
            num_samples = X.shape[1] - int(np.floor(self.dropout_rate*X.shape[1]))
            for i in range(X.shape[0]):
                self.drop_mask[i] = self.__get_pl_drop_ind(X[i],num_samples)

    def __get_pl_drop_ind(self,x,num_samples):
        sx = x.sum()
        xx = x.copy()
        total_ind = np.arange(x.shape[0])
        sampled_ind = []
        for i in range(num_samples):
            px = xx / sx
            sampled_ind.append(np.random.multinomial(1,px).argmax())
            xx[sampled_ind[-1]] = 0
            sx = xx.sum()
        return xx == 0

#Plackett-Luce dropout with Gumbel sampling.
class PLGumbelDropper(Dropper):
    def __init__(self,dropout_rate):
        Dropper.__init__(self,dropout_rate)

    def set_dropout_mask(self,X):
        if (self.dropout_rate > 0):
            num_drop = int(np.floor(self.dropout_rate*X.shape[1]))
            self.drop_mask = np.zeros(X.shape)
            U = np.random.rand(*X.shape)
            G = np.log(-np.log(U))
            LX = np.log(X)
            LX[X == 0] = -np.Inf
            XX = G + LX
            ind = XX.argsort(1)[:,num_drop]
            self.drop_mask = XX > XX[np.arange(XX.shape[0]),ind][:,None]