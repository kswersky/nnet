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

reload(la)
reload(lp)
reload(non)
reload(nu)

def checkgrad_test(build_net,seed=1,eps=1e-5):
    np.random.seed(seed)
    net = build_net()[0]
    net.check_grad(eps)

def param_vec_test():
    net = build_net()[0]
    param_vec = net.get_vec()
    print param_vec
    net.set_param_vec(param_vec)
    print net.get_vec()

def checkgrad_vec_test(build_net,seed=1,eps=1e-5):
    np.random.seed(seed)
    net = build_net()

    param_vec = net.get_vec()
    f = net.forward_prop()
    net.backward_prop()
    g = net.get_vec(grad=True)
    g_est = np.zeros(g.shape)
    for i in range(net.total_count):
        param_vec[i] += eps
        net.set_param_vec(param_vec)
        f1 = net.forward_prop()
        param_vec[i] -= 2*eps
        net.set_param_vec(param_vec)
        f2 = net.forward_prop()
        param_vec[i] += eps
        net.set_param_vec(param_vec)
        g_est[i] = (f1-f2) / (2*eps)
        print [g[i], g_est[i]]
    return g,g_est

class NeuralNet:
    def __init__(self):
        self.first_layer = None
        self.layers = []
        self.input_layers = {}
        self.target_layers = {}

    def add_layer(self,a_layer,output_layer,weight_multiplier=0.1,bias_multiplier=0):
        self.__add_layer(a_layer,output_layer)

        input_size = a_layer.size
        output_layer.params.make_weights(input_size,output_layer.size,0.1)
        output_layer.params.make_biases(output_layer.size)

        self.__update_net_structure()

        return output_layer.params

    def add_input_layer(self,input_layer):
        self.first_layer = input_layer
        self.__update_net_structure()

    def tie_weights(self, layer,tied_layer):
        tied_layer.params.set_weights(layer.params.weights)
        self.__update_net_structure()

    def tie_biases(self, layer,tied_layer):
        tied_layer.params.set_biases(layer.params.biases)
        self.__update_net_structure()

    def tie_weights_and_biases(self, layer,tied_layer):
        self.tie_weights(layer,tied_layer)
        self.tie_biases(layer,tied_layer)

    def __add_layer(self,a_layer,output_layer):
        a_layer.output_layers.append(output_layer)
        output_layer.input_layer = a_layer
    
    def add_input(self,input_name,layer):
        self.input_layers[input_name] = layer

    def add_loss_function(self,loss_name,layer,loss_function,weight=1):
        #Bind the method to the nonlinearity object if not done already
        if (loss_function.im_self is None):
            loss_function = loss_function.__get__(layer.nonlin)

        layer.loss_functions[loss_name] = loss_function
        layer.loss_weights[loss_name] = weight
        self.target_layers[loss_name] = layer

    def set_input(self,input_name,inp):
        self.input_layers[input_name].data = inp

    def set_target(self,target_name,targ):
        self.target_layers[target_name].targets[target_name] = targ

    def set_inputs(self,*input_list):
        for (input_name,inp) in input_list:
            self.input_layers[input_name].data = inp

    def set_targets(self,*target_list):
        for (target_name,targ) in target_list:
            self.target_layers[target_name].targets[target_name] = targ

    def __update_net_structure(self):
        self.layers = []
        bfsqueue = deque()
        bfsqueue.appendleft(self.first_layer)

        while (len(bfsqueue) > 0):
            layer = bfsqueue.pop()
            self.layers.append(layer)
            for output_layer in layer.output_layers:
                bfsqueue.appendleft(output_layer)

        self.param_set,self.total_count = self.get_net_param_info()

    def forward_prop(self):
        total_loss = 0
        for layer in self.layers:
            loss = layer.forward_prop()
            if (loss is not None):
                total_loss += loss

        return total_loss

    def backward_prop(self):
        for layer in self.layers:
            if (not isinstance(layer,la.InputLayer)):
                layer.params.weights.reset_gradients()
                layer.params.biases.reset_gradients()

        for layer in self.layers[::-1]:
            layer.backward_prop()

    def take_gradient_step(self,eta,mo=0):
        f = self.forward_prop()
        self.backward_prop()
        for param in self.param_set:
            param.add_to_momentum(mo,-eta*param.dparam)
            param.take_step(param.moparam)
            
        return f

    def nnet_vec_obj(self,param_vec):
        self.set_param_vec(param_vec)
        return self.forward_prop()

    def nnet_vec_grad(self,param_vec):
        self.set_param_vec(param_vec)
        f = self.forward_prop()
        self.backward_prop()
        return self.get_vec(grad=True)

    def get_vec(self,grad=False):
        param_vec = np.zeros(self.total_count)
        index = 0
        for weight in self.param_set.keys():
            if (not grad):
                param_vec[index:index+weight.count] = weight.get_param_vec()
            else:
                param_vec[index:index+weight.count] = weight.get_grad_vec()
            index = index + weight.count

        return param_vec

    def set_param_vec(self,param_vec):
        index = 0
        for param in self.param_set.keys():
            param.set_params_from_vec(param_vec[index:index+param.count])
            index = index + param.count

    def get_net_param_info(self):
        param_set = OrderedDict()
        total_count = 0
        for layer in self.layers:
            if (not isinstance(layer,la.InputLayer)):
                if (not param_set.has_key(layer.params.weights)):
                    param_set[layer.params.weights] = True
                    total_count += layer.params.weights.count
                if (not param_set.has_key(layer.params.biases)):
                    param_set[layer.params.biases] = True
                    total_count += layer.params.biases.count

        return param_set,total_count

    def check_grad(self,eps):
        self.forward_prop()
        self.backward_prop()

        for layer in self.layers:
            if (not isinstance(layer,la.InputLayer)):
                dW_est = np.zeros(layer.params.weights.dparam.shape)
                db_est = np.zeros(layer.params.biases.dparam.shape)
                for i in range(layer.params.weights.param.shape[0]):
                    for j in range(layer.params.weights.param.shape[1]):
                        layer.params.weights.param[i,j] += eps
                        loss_1 = self.forward_prop()
                        layer.params.weights.param[i,j] -= 2*eps
                        loss_2 = self.forward_prop()
                        layer.params.weights.param[i,j] += eps
                        dW_est[i,j] = (loss_1 - loss_2) / (2*eps)

                for i in range(layer.params.biases.param.shape[0]):
                    for j in range(layer.params.biases.param.shape[1]):
                        layer.params.biases.param[i,j] += eps
                        loss_1 = self.forward_prop()
                        layer.params.biases.param[i,j] -= 2*eps
                        loss_2 = self.forward_prop()
                        layer.params.biases.param[i,j] += eps
                        db_est[i,j] = (loss_1 - loss_2) / (2*eps)

                print layer.params.weights.dparam
                print dW_est
                print layer.params.biases.dparam
                print db_est
                print 'dW error: %s' % np.linalg.norm(layer.params.weights.dparam.flatten() - dW_est.flatten())
                print 'db error: %s' % np.linalg.norm(layer.params.biases.dparam.flatten() - db_est.flatten())
