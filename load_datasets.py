from scipy.io import loadmat
from numpy import *
import pickle
import copy

def load_cifar(filename):
    data = loadmat(filename)
    Xtrain = double(data['train_data'])
    Xtest = data['test_data']
    
    ytrain = nto1ofk(data['train_labels'].flatten())
    ytest = nto1ofk(data['test_labels'].flatten())
    
    return Xtrain,ytrain,Xtest,ytest
    
def load_cifar_features(*batch_names):
    X = None
    y = None
    for i in batch_names:
        f = open(i)
        D = pickle.load(f)
        if (X is None):
            X = D['data']
            y = nto1ofk(array(D['labels']))
        else:
            X = vstack((X,D['data']))
            y = vstack((y,nto1ofk(array(D['labels']))))
        f.close()
    while (X.sum(1).max()>1e5):
        #Take care of problem entry
        row_ind = X.sum(1).argmax()
        column_ind = X[row_ind,:].argmax()
        X[row_ind,column_ind] = 0
        print row_ind
    return X,y
    
def load_montreal(filename):
    data = loadmat(filename)
    train = data['train']
    test = data['test']
    valid = data['valid']
    
    Xtrain = double(train[:,:-1])
    Xtest = double(test[:,:-1])
    Xvalid = double(valid[:,:-1])

    #Make sure data is normalized
    Xtrain = (Xtrain + Xtrain.min())/Xtrain.max()
    Xtest = (Xtest + Xtest.min())/Xtest.max()
    Xvalid = (Xvalid + Xvalid.min())/Xvalid.max()
    
    ytrain = nto1ofk(train[:,-1])
    ytest = nto1ofk(test[:,-1])
    yvalid = nto1ofk(valid[:,-1])
    
    return Xtrain,ytrain,Xtest,ytest,Xvalid,yvalid
    
def nto1ofk(y):
    #Assumes y starts at 0 and goes to K
    Y = zeros((y.shape[0],y.max()+1))
    Y[range(Y.shape[0]),int_(y)] = 1
    return Y
    
def load_mnist(digitsRange,binary=True):
	digitsRange = copy.copy(digitsRange)
	digits = loadmat('mnist_all.mat')
	d = digits['train'+str(digitsRange[0])]
	dTest = digits['test'+str(digitsRange[0])]
	if (len(digitsRange)<=2 and binary):
		targets = zeros((d.shape[0],))
		targetsTest = zeros((dTest.shape[0],))
	else:
		targets = zeros((d.shape[0],len(digitsRange)))
		targets[0:,0] = 1
		targetsTest = zeros((dTest.shape[0],len(digitsRange)))
		targetsTest[0:,0] = 1
	digitsRange.remove(digitsRange[0])
	dimIndex = 1
	for i in digitsRange:
		d = vstack((d,digits['train'+str(i)]))
		dTest = vstack((dTest,digits['test'+str(i)]))
		index = targets.shape[0]
		indexTest = targetsTest.shape[0]
		if (len(digitsRange) <= 1 and binary):
			targets = hstack((targets,ones((digits['train'+str(i)].shape[0],))))
			targetsTest = hstack((targetsTest,ones((digits['test'+str(i)].shape[0]))))
		else:
			targets = vstack((targets,zeros((digits['train'+str(i)].shape[0],len(digitsRange)+1))))
			targets[index:,dimIndex] = 1
			
			targetsTest = vstack((targetsTest,zeros((digits['test'+str(i)].shape[0],len(digitsRange)+1))))
			targetsTest[indexTest:,dimIndex] = 1
			dimIndex = dimIndex + 1
	d = double(d)/255
	dTest = double(dTest)/255
	return d,targets,dTest,targetsTest,dTest,targetsTest