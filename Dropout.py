import numpy as np

class Dropper:
    def __init__(self,dropout_rate=0):
        self.drop_mask = None
        self.dropout_rate = dropout_rate

    def apply_dropout(self,X,**kwargs): 
        if (self.drop_mask is not None):
            return self.drop_mask*X
        else:
            return X

    def compute_total_input_test(self,outputs,params,previous_layer_dropout_rate):
        return params.compute_total_input(outputs)

    #This will be overridden by a subclass.
    def set_dropout_mask(self,X,**kwargs):
        self.drop_mask = None

#Good ol' regular dropout.
class VanillaDropper(Dropper):
    def __init__(self,dropout_rate):
        Dropper.__init__(self,dropout_rate)

    def set_dropout_mask(self,X,**kwargs):
        test = kwargs.get('test',False)
        if (self.dropout_rate > 0 and not test):
            self.drop_mask = np.random.rand(*X.shape) >= self.dropout_rate
        elif (test):
            self.drop_mask = None

    def compute_total_input_test(self,outputs,params,previous_layer_dropout_rate):
        params.weights.param *= 1-previous_layer_dropout_rate
        params.biases.param *= 1-previous_layer_dropout_rate
        total_input = params.compute_total_input(outputs)
        params.weights.param /= 1-previous_layer_dropout_rate
        params.biases.param /= 1-previous_layer_dropout_rate
        return total_input

class PLGumbelDropper(Dropper):
    def __init__(self,dropout_rate):
        Dropper.__init__(self,dropout_rate)

    def set_dropout_mask(self,X,**kwargs):
        test = kwargs.get('test',False)
        if (self.dropout_rate > 0):
            num_drop = int(np.floor(self.dropout_rate*X.shape[1]))
            self.drop_mask = np.zeros(X.shape)
            if (not test):
                G = np.random.gumbel(size=X.shape)
                log_X = np.zeros(X.shape)
                log_X[X > 0] = np.log(X[X > 0])
                log_X[X <= 0] = -np.Inf
                XX = G + log_X
                ind = XX.argsort(1)[:,num_drop]
                self.drop_mask = XX >= XX[np.arange(XX.shape[0]),ind][:,None]
            else:
                self.drop_mask = None

"""
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
"""
