##
# Copyright (C) 2012 Kevin Swersky
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
import optim as op
import build_net_examples as bne
import display_filters as d
import numpy as np
from scipy.io import loadmat
import pylab

def train_tied_autoencoder_dropout_sgd(X,num_hid,num_iters,eta,mo):
    net,input_dict,target_dict,param_dict = bne.binary_tied_autoencoder_dropout(X,num_hid)
    op.train_nnet_sgd(net,input_dict,target_dict,num_iters=num_iters,eta=eta,mo=mo)
    W = param_dict['W'].param
    pylab.ioff()
    d.print_aligned(W)
    pylab.show()
    return param_dict

def train_sparse_autoencoder_sgd(X,num_hid,num_iters,eta,mo,kl_target,kl_weight):
    net,input_dict,target_dict,param_dict = bne.binary_sparse_autoencoder(X,num_hid,kl_target,kl_weight)
    op.train_nnet_sgd(net,input_dict,target_dict,num_iters=num_iters,eta=eta,mo=mo)
    W = param_dict['W'].param
    pylab.ioff()
    d.print_aligned(W)
    pylab.show()
    return param_dict

if __name__ == '__main__':
    D = loadmat('mnist_small.mat')
    X = D['X']

    num_examples = X.shape[0]
    num_hid = 100
    num_iters = 50
    eta = 0.1
    mo = 0.9
    np.random.seed(1)

    ind = np.random.permutation(X.shape[0])[0:num_examples]
    X = X[ind]

    train_tied_autoencoder_dropout_sgd(X,num_hid,num_iters,eta,mo)