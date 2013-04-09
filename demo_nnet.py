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
import Dropout as dp
import display_filters as d
import numpy as np
from scipy.io import loadmat
import load_datasets as ld
import pylab

def train_tied_autoencoder_dropout_sgd(X,num_hid,num_iters,eta,mo):
    net,input_dict,target_dict,param_dict = bne.binary_tied_autoencoder_dropout(X,num_hid)
    op.train_nnet_sgd(net,input_dict,target_dict,num_iters=num_iters,eta=eta,mo=mo)
    W = param_dict['W'].param
    pylab.ioff()
    d.print_aligned_color(W)
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

def train_classifier_net(X,Y,Xtest,Ytest,num_hid,num_iters,eta,mo):
    net,input_dict,target_dict,param_dict = bne.one_layer_classifier_net(X,Y,num_hid)
    op.train_nnet_sgd(net,input_dict,target_dict,num_iters=num_iters,eta=eta,mo=mo,Xtest=Xtest,Ytest=Ytest)
    W = param_dict['W'].param
    pylab.ioff()
    d.print_aligned(W)
    pylab.show()
    return param_dict

def train_regression_net(num_hid,num_iters,eta,mo):
    X = np.array([[ 0.16534698],
      [ 0.33069396],
      [ 0.49604095],
      [ 0.66138793],
      [ 0.82673491],
      [ 0.99208189],
      [ 1.8188168 ],
      [ 1.98416378],
      [ 2.14951076],
      [ 2.31485774]])
    Y = np.array([ 0.59695623,  0.94320958,  0.91779631,  0.46602278, -0.15296497,
      -0.73801201,  0.84207673,  1.02725234,  0.71366892,  0.14607511])[:,None]
    Xtest = np.arange(0,3,0.1)[:,None]
    Ytest = Xtest
    net,input_dict,target_dict,param_dict = bne.one_layer_regression_net(X,Y,num_hid)
    net.set_input('input',X)
    net.set_target('squared loss',Y)
    layer1 = net.input_layers['input'].output_layers[0]
    layer2 = layer1.output_layers[0]
    layer1.dropper = dp.VanillaDropper(0)
    layer2.dropper = dp.VanillaDropper(0)
    op.train_nnet_sgd(net,input_dict,target_dict,num_iters=num_iters,eta=eta,mo=mo,batch_size=X.shape[0])
    import pickle
    #f = open('regression_net.dat','w')
    #pickle.dump({'net':net},f)
    #f.close()
    W = param_dict['W'].param
    pylab.ioff()
    net.set_input('input',Xtest)
    net.set_target('squared loss',Ytest)  
    net.forward_prop(test=True)
    pylab.plot(Xtest,net.target_layers['squared loss'].outputs)
    pylab.plot(X,Y)
    pylab.show()
    return param_dict

if __name__ == '__main__':
    #D = loadmat('mnist_small.mat')
    #D = loadmat('patches_16x16.mat')
    #X = D['patches']
    D = ld.load_mnist(range(10))
    X = D[0]
    Y = D[1]
    Xtest = D[2]
    Ytest = D[3]

    num_examples = X.shape[0]
    num_hid = 500
    num_iters = 100
    eta = 0.1
    mo = 0.9
    np.random.seed(1)

    ind = np.random.permutation(X.shape[0])[0:num_examples]
    X = X[ind]
    Y = Y[ind]

    #train_tied_autoencoder_dropout_sgd(X,num_hid,num_iters,eta,mo)
    train_classifier_net(X,Y,Xtest,Ytest,num_hid,num_iters,eta,mo)