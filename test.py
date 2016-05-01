# pyVolve package test
import numpy as np
import pyVolve

def evaluateA(org):
	return pyVolve.l2(org.sum(0),np.array([1,2,3,4,5]))

def evaluateB(org):
	return pyVolve.l2(org.T.sum(0),np.array([1,2,3,4,5]))

pop = pyVolve.Population()
pop.populateRandom(1000,(5,5))
pop.batchGeneration(1000,[evaluateA],[0.9],[1])
