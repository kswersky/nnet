import numpy as np

class Layer:
    def __init__(self,size,nonlin):
        self.size = size
        self.nonlin = nonlin
        self.output_layers = []
        self.input_layer = None
        self.inputs = None
        self.targets = {}
        self.params = None
        self.outputs = None
        self.backprop_grad = None
        self.loss_functions = {}
        self.loss_weights = {}
        self.total_input = None

    def forward_prop(self):
        if (self.inputs is None):
            inputs = self.input_layer.outputs
        else:
            inputs = self.inputs

        self.total_input = np.dot(inputs,self.params.W) + self.params.b
        self.outputs = self.nonlin.nonlin(self.total_input)

        if (self.loss_functions):
            return self.apply_loss_functions()

    def backward_prop(self):
        total_backprop_grad_from_outputs = 0
        for ol in self.output_layers:
            total_backprop_grad_from_outputs = total_backprop_grad_from_outputs + ol.backprop_grad

        backward_act = self.nonlin.nonlin_grad_J(self.total_input,total_backprop_grad_from_outputs)

        if (self.loss_functions):
            dloss = self.apply_loss_functions(grad=True)
            backward_act = backward_act + dloss

        if (self.inputs is None):
            inputs = self.input_layer.outputs
        else:
            inputs = self.inputs

        dW = np.dot(inputs.T,backward_act)
        db = np.sum(backward_act,0)[None,:]

        self.params.add_to_gradients(dW,db)

        self.backprop_grad = np.dot(backward_act,self.params.W.T)

        return dW,db

    def apply_loss_functions(self,**kwargs):
        grad = kwargs.get('grad',False)
        total_loss = 0
        for loss_name,loss in self.loss_functions.iteritems():
            total_loss = total_loss + self.loss_weights[loss_name]*loss(self.total_input,self.targets[loss_name],grad)
            
        return total_loss

class LayerParam:
    def __init__(self,input_size,output_size):
        #Parameters
        self.W = 0.1*np.random.randn(input_size,output_size)
        self.b = np.zeros((1,output_size))

        #Derivatives
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

        #Momentum terms for gradient descent
        self.moW = np.zeros(self.W.shape)
        self.mob = np.zeros(self.b.shape)
        self.count = input_size*output_size + output_size

    def apply_param(inputs):
        return self.param_func(inputs)

    def add_to_gradients(self,dW,db):
        self.dW += dW
        self.db += db

    def reset_gradients(self):
        self.dW = 0*self.dW
        self.db = 0*self.db

    def get_param_vec(self):
        return np.hstack((self.W.flatten(),self.b.flatten()))

    def get_grad_vec(self):
        return np.hstack((self.dW.flatten(),self.db.flatten()))

    def set_params_from_vec(self,param_vec):
        index = 0
        self.W = param_vec[index:index+np.prod(self.W.shape)].reshape(self.W.shape)
        index = index + np.prod(self.W.shape)
        self.b = param_vec[index:index+np.prod(self.b.shape)].reshape(self.b.shape)
        