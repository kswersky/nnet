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
import numpy as np
import nnet_utils as nu
from collections import deque
from collections import OrderedDict
import Nonlin as non
import Layer as la
import LayerParam as lp
import nnet
import Dropout as dp

reload(nnet)
reload(la)
reload(lp)
reload(non)
reload(nu)
reload(dp)

def crazy_net():
    X = np.random.rand(5,10).round()
    y2_2 = np.random.rand(5,1).round()
    y6 = np.random.randn(5,3)
    y6 = np.double(y6 == y6.max(1)[:,None])

    net = nnet.NeuralNet()

    input_layer = la.InputLayer(X.shape[1])
    layer1 = la.Layer(7,non.SoftmaxNonlin())
    layer2_1 = la.Layer(9,non.SoftReluNonlin())
    layer2_2 = la.Layer(1,non.SigmoidNonlin())
    layer3 = la.Layer(4,non.SoftReluNonlin())
    layer4_1 = la.Layer(4,non.SoftReluNonlin())
    layer4_2 = la.Layer(3,non.SoftmaxNonlin())
    layer5 = la.Layer(4,non.SoftReluNonlin()) 
    layer6 = la.Layer(10,non.SigmoidNonlin())
    
    net.add_input_layer(input_layer)
    net.add_layer(input_layer,layer1)
    net.add_layer(layer1,layer2_1)
    net.add_layer(layer1,layer2_2)
    net.add_layer(layer2_1,layer3)
    net.add_layer(layer3,layer4_1)
    net.add_layer(layer3,layer4_2)
    net.add_layer(layer4_1,layer5)
    net.add_layer(layer5,layer6)

    net.add_input('input',input_layer)
    net.add_loss_function('loss2_2',layer2_2,non.SigmoidNonlin.crossentropy_loss)
    net.add_loss_function('loss4_2',layer4_2,non.SoftmaxNonlin.crossentropy_loss,1)
    net.add_loss_function('loss6_v2',layer6,non.SigmoidNonlin.crossentropy_loss,2)

    net.set_input('input',X)
    net.set_target('loss2_2',y2_2)
    net.set_target('loss4_2',y6)
    net.set_target('loss6_v2',X)

    return net

def binary_tied_autoencoder_dropout(
        X=None,numhid=7,
        input_layer_dropout_type=dp.VanillaDropper,
        input_layer_dropout_rate=0.2,
        hidden_layer_dropout_type=dp.VanillaDropper,
        hidden_layer_dropout_rate=0.5):

    if (X is None):
        X = np.random.rand(5,10).round()
    input_layer = la.InputLayer(X.shape[1],dropper=input_layer_dropout_type(input_layer_dropout_rate))
    layer1 = la.Layer(numhid,non.SigmoidNonlin(),dropper=hidden_layer_dropout_type(hidden_layer_dropout_rate))
    output_layer = la.Layer(X.shape[1],non.SigmoidNonlin(),weight_type=lp.TransposeWeight)

    net = nnet.NeuralNet()
    net.add_input_layer(input_layer)
    net.add_layer(input_layer,layer1)
    net.add_layer(layer1,output_layer)

    net.tie_weights(layer1,output_layer)

    net.add_input('input',input_layer)
    net.add_loss_function('reconstruction loss',output_layer,non.SigmoidNonlin.crossentropy_loss)
    
    net.set_input('input',X)
    net.set_target('reconstruction loss',X)

    input_dict = {'input':X}
    target_dict = {'reconstruction loss':X}

    param_dict = {'W':layer1.params.weights,'b':layer1.params.biases,'c':output_layer.params.biases}

    return net,input_dict,target_dict,param_dict

def binary_sparse_autoencoder(X=None,numhid=7,kl_target=0.1,kl_weight=1):
    if (X is None):
        X = np.random.rand(5,10).round()
    input_layer = la.InputLayer(X.shape[1])
    layer1 = la.Layer(numhid,non.SigmoidNonlin())
    output_layer = la.Layer(X.shape[1],non.SigmoidNonlin())

    net = nnet.NeuralNet()
    net.add_input_layer(input_layer)
    net.add_layer(input_layer,layer1)
    net.add_layer(layer1,output_layer)

    net.add_input('input',input_layer)
    net.add_loss_function('reconstruction loss',output_layer,non.SigmoidNonlin.crossentropy_loss)
    net.add_loss_function('kl_sparsity_loss',layer1,non.SigmoidNonlin.kl_sparsity_loss,kl_weight)

    net.set_input('input',X)
    net.set_target('reconstruction loss',X)
    net.set_target('kl_sparsity_loss',kl_target)

    input_dict = {'input':X}
    target_dict = {'reconstruction loss':X, 'kl_sparsity_loss':kl_target}

    param_dict = {'W':layer1.params.weights,'b':layer1.params.biases,'c':output_layer.params.biases}

    return net,input_dict,target_dict,param_dict