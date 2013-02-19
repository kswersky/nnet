import numpy as np
import nnet_utils as nu
from collections import deque
from collections import OrderedDict
import Nonlin as non
import Layer as la

def checkgrad_test(seed=1,eps=1e-5):
    np.random.seed(seed)
    net = build_net()
    net.check_grad(eps)

def param_vec_test():
    net = build_net()
    param_vec = net.get_vec()
    print param_vec
    net.set_param_vec(param_vec)
    print net.get_vec()

def checkgrad_vec_test(seed=1,eps=1e-5):
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

def build_net():
    X = np.random.rand(5,10).round()
    y2_2 = np.random.rand(5,1).round()
    y6 = np.random.randn(5,3)
    y6 = np.double(y6 == y6.max(1)[:,None])

    net = NeuralNet()

    layer1 = la.Layer(7,non.SoftmaxNonlin())
    layer2_1 = la.Layer(9,non.SoftReluNonlin())
    layer2_2 = la.Layer(1,non.SigmoidNonlin())
    layer3 = la.Layer(4,non.SoftReluNonlin())
    layer4_1 = la.Layer(4,non.SoftReluNonlin())
    layer4_2 = la.Layer(3,non.SoftmaxNonlin())
    layer5 = la.Layer(4,non.SoftReluNonlin()) 
    layer6 = la.Layer(10,non.SigmoidNonlin())
    
    net.add_layer(10,layer1)
    net.add_layer(layer1,layer2_1)
    net.add_layer(layer1,layer2_2)
    net.add_layer(layer2_1,layer3)
    params = net.add_layer(layer3,layer4_1)
    net.add_layer(layer3,layer4_2)
    net.add_layer(layer4_1,layer5,params)
    net.add_layer(layer5,layer6)

    net.add_input('layer1_input',layer1)
    net.add_loss_function('loss2_2',layer2_2,non.SigmoidNonlin.crossentropy_loss)
    net.add_loss_function('loss4_2',layer4_2,non.SoftmaxNonlin.crossentropy_loss,1)
    net.add_loss_function('loss6_v2',layer6,non.SigmoidNonlin.crossentropy_loss,2)

    net.set_input('layer1_input',X)
    net.set_target('loss2_2',y2_2)
    net.set_target('loss4_2',y6)
    net.set_target('loss6_v2',X)

    return net

class NeuralNet:
    def __init__(self):
        self.first_layer = None
        self.layers = []
        self.input_layers = {}
        self.target_layers = {}

    def add_layer(self,an_input,output_layer,params=None):
        try:
            if (type(an_input) == int):
                self.first_layer = output_layer
                input_size = an_input
            elif (an_input.__class__ == la.Layer):
                an_input.output_layers.append(output_layer)
                output_layer.input_layer = an_input
                input_size = an_input.size
            else:
                raise Exception('Input must either be an int or a Layer object')

            if (params is None):
                output_layer.params = LayerParam(input_size,output_layer.size)
            else:
                assert params.W.shape[0] == input_size and params.W.shape[1] == output_layer.size, 'Input and output sizes must match the given parameter shape.'
                output_layer.params = params

        except Exception as e:
            print e

        self.__update_net_structure()

        return output_layer.params

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
        self.input_layers[input_name].inputs = inp

    def set_target(self,target_name,targ):
        self.target_layers[target_name].targets[target_name] = targ

    def set_inputs(self,*input_list):
        for (input_name,inp) in input_list:
            self.input_layers[input_name].inputs = inp

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
            layer.params.reset_gradients()

        for layer in self.layers[::-1]:
            layer.backward_prop()

    def gradient_step(self,eta,mo=0):
        f = self.forward_prop()
        self.backward_prop()
        for param in self.param_set:
            param.moW = param.moW - eta*param.dW
            param.mob = param.mob - eta*param.db
            param.W = param.W + param.moW
            param.b = param.b + param.mob
            
        return f

    def get_vec(self,grad=False):
        param_vec = np.zeros(self.total_count)
        index = 0
        for param in self.param_set.keys():
            if (not grad):
                param_vec[index:index+param.count] = param.get_param_vec()
            else:
                param_vec[index:index+param.count] = param.get_grad_vec()
            index = index + param.count

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
            if (not param_set.has_key(layer.params)):
                param_set[layer.params] = True
                total_count += layer.params.count

        return param_set,total_count

    def check_grad(self,eps):
        self.forward_prop()
        self.backward_prop()

        for layer in self.layers:
            dW_est = np.zeros(layer.params.dW.shape)
            db_est = np.zeros(layer.params.db.shape)
            for i in range(layer.params.W.shape[0]):
                for j in range(layer.params.W.shape[1]):
                    layer.params.W[i,j] += eps
                    loss_1 = self.forward_prop()
                    layer.params.W[i,j] -= 2*eps
                    loss_2 = self.forward_prop()
                    layer.params.W[i,j] += eps
                    dW_est[i,j] = (loss_1 - loss_2) / (2*eps)

            for i in range(layer.params.b.shape[0]):
                for j in range(layer.params.b.shape[1]):
                    layer.params.b[i,j] += eps
                    loss_1 = self.forward_prop()
                    layer.params.b[i,j] -= 2*eps
                    loss_2 = self.forward_prop()
                    layer.params.b[i,j] += eps
                    db_est[i,j] = (loss_1 - loss_2) / (2*eps)

            print layer.params.dW
            print dW_est
            print layer.params.db
            print db_est
            print 'dW error: %s' % np.linalg.norm(layer.params.dW.flatten() - dW_est.flatten())
            print 'db error: %s' % np.linalg.norm(layer.params.db.flatten() - db_est.flatten())
