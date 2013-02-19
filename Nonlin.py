import numpy as np
import nnet_utils as nu

class Nonlin:
    def nonlin(self):
        return None

    def nonlin_grad_J(self):
        return None

class SigmoidNonlin(Nonlin):
    def nonlin(self,total_input):
        return nu.sigmoid(total_input)

    def nonlin_grad_J(self,total_input,backprop_tot):
        preds = self.nonlin(total_input)
        return preds*(1-preds)*backprop_tot

    def crossentropy_loss(self,total_input,targets,grad=False):
        log_preds = nu.log_sigmoid(total_input)
        log_one_minus_preds = nu.log_sigmoid(-total_input)
        preds = np.exp(log_preds)
        if (not grad):
            loss = -np.sum(targets*log_preds + (1-targets)*log_one_minus_preds,1).mean(0)
            return loss
        else:
            dloss = (preds - targets) / targets.shape[0]
            return dloss

class AbsNonlin(Nonlin):
    def nonlin(self,total_input):
        return np.abs(total_input)

    def nonlin_grad_J(self,total_input,backprop_tot):
        mult = np.ones(total_input)
        mult[total_input < 0] = -1
        return mult*backprop_tot

class SoftReluNonlin(Nonlin):
    def nonlin(self,total_input):
        return -nu.log_sigmoid(-total_input)

    def nonlin_grad_J(self,total_input,backprop_tot):
        preds = self.nonlin(total_input)
        return (1-(1/np.exp(preds)))*backprop_tot

class ReluNonlin(Nonlin):
    def nonlin(self,total_input):
        return np.maximum(0,total_input)

    def nonlin_grad_J(self,total_input,backprop_tot):
        preds = self.nonlin(total_input)
        return (preds > 0)*backprop_tot

class SoftmaxNonlin(Nonlin):
    def nonlin(self,total_input):
        return nu.softmax(total_input)

    def nonlin_grad_J(self,total_input,backprop_tot):
        preds = self.nonlin(total_input)
        return preds*(backprop_tot - np.sum(backprop_tot*preds,1)[:,None])

    def crossentropy_loss(self,total_input,targets,grad=False):
        preds = self.nonlin(total_input)
        if (not grad):
            loss = -np.mean(np.log(np.sum(targets*preds,1)))
            return loss
        else:
            dloss = (preds - targets) / targets.shape[0]
            return dloss