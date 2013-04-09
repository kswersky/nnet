##
# Copyright (C) 2012 Kevin Swersky
# This code is inspired by code originally written by Ilya Sutskever.
# 
# This code is written for research and educational purposes only.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import numpy as np
import nnet_utils as nu

class Nonlin:
    def nonlin(self):
        return None

    def nonlin_grad_J(self):
        return None

class LinearNonlin(Nonlin):
    def nonlin(self,total_input):
        return total_input

    def nonlin_grad_J(self,total_input,backprop_tot):
        return backprop_tot

    def squared_loss(self,total_input,targets,grad=False):
        if (not grad):
            loss = 0.5*np.sum((total_input-targets)**2) / total_input.shape[0]
            return loss
        else:
            dloss = (total_input - targets) / total_input.shape[0]
            return dloss

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

    def intersection_over_union_loss(self,total_input,targets,grad=False):
        preds = nu.sigmoid(total_input)
        t1 = targets*preds
        t2 = (1-targets)*preds
        if (not grad):
            loss = -np.mean(np.log(t1.sum(1)) - np.log((targets+t2).sum(1)))
            return loss
        else:
            dloss = -((1/t1.sum(1))[:,None]*t1 - (1/(targets+t2).sum(1))[:,None]*t2)*(1-preds) / targets.shape[0]
            return dloss

    def kl_sparsity_loss(self,total_input,target_prob,grad=False):
        preds = nu.sigmoid(total_input)
        q = preds.mean(0)
        if (not grad):
            loss = -np.sum(target_prob*np.log(q) + (1-target_prob)*np.log(1-q))
            return loss
        else:
            dloss = preds*(1-preds)*((q - target_prob)/(q*(1-q))) / total_input.shape[0]
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