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
from scipy.optimize import fmin_l_bfgs_b

def train_nnet_lbfgs(net,maxfun=1):
    params = fmin_l_bfgs_b(net.nnet_vec_obj, init_params, fprime=net.nnet_vec_grad, disp=1, maxfun=maxfun)
    return params

def train_nnet_sgd(net,input_dict,target_dict,**kwargs):
    num_examples = 0
    num_iters = kwargs.get('num_iters',1)
    batch_size = kwargs.get('batch_size',100)
    eta = kwargs.get('eta',0.1)
    mo = kwargs.get('mo',0)
    for input_name,inp in input_dict.iteritems():
        assert (num_examples == 0 or inp.shape[0] == num_examples), 'Number of examples must be consistent across inputs.'
        num_examples = inp.shape[0]

    num_batches = np.ceil(np.double(num_examples)/batch_size)
    for i in range(num_iters):
        obj = 0
        randIndices_data = np.random.permutation(num_examples)
        for batch in range(int(num_batches)):
            print "iteration " + str(i) + " batch " + str(batch) + " of " + str(int(num_batches))
            for input_name,inp in input_dict.iteritems():
                batch_inp = inp[randIndices_data[np.mod(range(batch*batch_size,(batch+1)*batch_size),num_examples)]]
                net.set_input(input_name,batch_inp)
            for target_name,targ in target_dict.iteritems():
                #Targets set at each minibatch should probably be the same size as the number of examples for a 1-1 mapping.
                #This might not always be the case, but it is sufficient for most purposes.
                #Other targets, like scalars, can be set before using SGD.
                if (isinstance(targ,np.ndarray) and targ.shape[0] == num_examples):
                    batch_targ = targ[randIndices_data[np.mod(range(batch*batch_size,(batch+1)*batch_size),num_examples)]]
                    net.set_target(target_name,batch_targ)

            f = net.take_gradient_step(eta,mo)
            obj += f/num_batches

        print 'Iteration %d complete. Object value: %s' % (i,obj)
