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
import LayerParam as lp
import Dropout as dp

class Layer:
    def __init__(self,size,nonlin,**kwargs):
        weight_type = kwargs.get('weight_type',lp.LinearWeight)
        bias_type = kwargs.get('bias_type',lp.LinearBias)
        dropper = kwargs.get('dropper',dp.VanillaDropper(0))
        weight_constraints = kwargs.get('weight_constraints',None)
        self.size = size
        self.nonlin = nonlin
        self.output_layers = []
        self.input_layer = None
        self.inputs = None
        self.targets = {}
        self.params = lp.LayerParam(weight_type,bias_type,weight_constraints=weight_constraints)
        self.outputs = None
        self.backprop_grad = None
        self.loss_functions = {}
        self.loss_weights = {}
        self.total_input = None
        self.dropper = dropper

    def forward_prop(self,**kwargs):
        test = kwargs.get('test',False)
        if (not test):
            self.total_input = self.params.compute_total_input(self.input_layer.outputs)
        else:
            self.total_input = self.dropper.compute_total_input_test(self.input_layer.outputs,self.params,self.input_layer.dropper.dropout_rate)

        self.outputs = self.nonlin.nonlin(self.total_input)
        self.dropper.set_dropout_mask(self.outputs,test=test,total_input=self.total_input)
        self.outputs = self.dropper.apply_dropout(self.outputs,test=test)

        if (self.loss_functions):
            return self.apply_loss_functions()

    def backward_prop(self):
        total_backprop_grad_from_outputs = 0
        for ol in self.output_layers:
            total_backprop_grad_from_outputs = total_backprop_grad_from_outputs + ol.backprop_grad

        backward_act = self.nonlin.nonlin_grad_J(self.total_input,total_backprop_grad_from_outputs)
        backward_act = self.dropper.apply_dropout(backward_act)

        if (self.loss_functions):
            dloss = self.apply_loss_functions(grad=True)
            backward_act = backward_act + dloss

        self.backprop_grad = self.params.apply_backprop_gradient(self.input_layer.outputs,backward_act)

    def apply_loss_functions(self,**kwargs):
        grad = kwargs.get('grad',False)
        total_loss = 0
        for loss_name,loss in self.loss_functions.iteritems():
            total_loss = total_loss + self.loss_weights[loss_name]*loss(self.total_input,self.targets[loss_name],grad)
            
        return total_loss

class InputLayer(Layer):
    def __init__(self,size,dropper=dp.VanillaDropper(0)):
        Layer.__init__(self,size,None,dropper=dropper)
        self.data = None

    def forward_prop(self,**kwargs):
        test = kwargs.get('test',False)
        self.outputs = self.data
        self.dropper.set_dropout_mask(self.outputs,test=test)
        self.outputs = self.dropper.apply_dropout(self.outputs,test=test)

    def backward_prop(self):
        self.backprop_grad = None

    def apply_loss_function(self,**kwargs):
        return None
