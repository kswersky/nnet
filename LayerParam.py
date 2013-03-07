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

class LinearWeight:
    @staticmethod
    def compute_total_input(inputs,W):
        return np.dot(inputs,W)

    @staticmethod
    def apply_backprop_gradient(inputs,backward_act,W):
        dW = np.dot(inputs.T,backward_act)
        backprop_grad = np.dot(backward_act,W.T)
        return dW, backprop_grad

class LinearBias:
    @staticmethod
    def compute_total_input(b):
        return b

    @staticmethod
    def apply_backprop_gradient(backward_act,b):
        db = np.sum(backward_act,0)[None,:]
        return db

class TransposeWeight:
    @staticmethod
    def compute_total_input(inputs,W):
        return np.dot(inputs,W.T)

    @staticmethod
    def apply_backprop_gradient(inputs,backward_act,W):
        dW = np.dot(inputs.T,backward_act).T
        backprop_grad = np.dot(backward_act,W)
        return dW, backprop_grad

class ExpWeight:
    @staticmethod
    def compute_total_input(inputs,W):
        return np.dot(inputs,np.exp(W))

    @staticmethod
    def apply_backprop_gradient(inputs,backward_act,W):
        dW = np.dot(inputs.T,backward_act)*np.exp(W)
        backprop_grad = np.dot(backward_act,np.exp(W).T)
        return dW, backprop_grad

class ExpBias:
    @staticmethod
    def compute_total_input(b):
        return np.exp(b)

    @staticmethod
    def apply_backprop_gradient(backward_act,b):
        db = np.exp(b)*np.sum(backward_act,0)[None,:]
        return db

class LayerParam:
    def __init__(self,weight_type=LinearWeight,bias_type=LinearBias):
        self.count = 0
        self.__weight_type = weight_type
        self.__bias_type = bias_type
        self.weights = None
        self.biases = None

    def set_params(self,weights,biases):
        self.set_weights(weights)
        self.set_biases(biases)

    def set_weights(self,weights):
        assert weights is not None, 'Weights should be initialized, perhaps you should build the network before calling this?'
        self.weights = weights
        self.count = self.weights.count
        if (self.biases is not None):
            self.count += self.biases.count

    def set_biases(self,biases):
        assert biases is not None, 'Biases should be initialized, perhaps you should build the network before calling this?'
        self.biases = biases
        self.count = self.biases.count
        if (self.weights is not None):
            self.count += self.weights.count

    def make_params(self,input_size,output_size,weight_multiplier=0.1,bias_multiplier=0):
        self.make_weights(input_size,output_size,weight_multiplier)
        self.make_biases(output_size,bias_multiplier)

    def make_weights(self,input_size,output_size,weight_multiplier):
        weights = Param((input_size,output_size),weight_multiplier)
        self.set_weights(weights)

    def make_biases(self,output_size,bias_multiplier=0):
        biases = Param((1,output_size),bias_multiplier)
        self.set_biases(biases)

    def compute_total_input(self,inputs):
        total_input = self.__weight_type.compute_total_input(inputs,self.weights.param)
        if (self.biases is not None):
            total_input += self.__bias_type.compute_total_input(self.biases.param)
        return total_input

    def apply_backprop_gradient(self,inputs,backward_act):
        dW, backprop_grad = self.__weight_type.apply_backprop_gradient(inputs,backward_act,self.weights.param)
        self.weights.add_to_gradients(dW)
        if (self.biases is not None):
            db = self.__bias_type.apply_backprop_gradient(backward_act,self.biases.param)
            self.biases.add_to_gradients(db)
        return backprop_grad

class Param:
    def __init__(self,weight_size,multiplier):
        self.param =  multiplier*np.random.randn(*weight_size)
        self.dparam = np.zeros(self.param.shape)
        self.moparam = np.zeros(self.param.shape)
        self.count = np.prod(self.param.shape)
        
    def add_to_gradients(self,dparam):
        self.dparam += dparam

    def add_to_momentum(self,mo,moparam):
        self.moparam = mo*self.moparam + moparam

    def take_step(self,direction):
        self.param += direction

    def reset_gradients(self):
        self.dparam = 0*self.dparam

    def get_param_vec(self):
        return self.param.flatten()

    def get_grad_vec(self):
        return self.dparam.flatten()

    def set_params_from_vec(self,param_vec):
        self.param = param_vec.reshape(self.param.shape)

    def count(self):
        return np.prod(self.param.shape)