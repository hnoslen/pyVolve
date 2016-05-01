# python genetic algorithms library
import numpy as np

def l2(ar1,ar2):
	return ((ar1.reshape((ar1.size,))-ar2.reshape((ar2.size,)))**2).sum()**0.5

class Population:
	"""Class to hold organism objects and perform common operations
	Organisms are represented as numpy arrays. Specifications of organisms
	should be done before adding to the herd. For fitness, 0 is optimal"""

	herd = np.array([])
	fitness = np.array([])
	organismShape = ()
	mutation_rate = 0.05


	def addToHerd(self, organism):
		if self.herd.size!=0:
			self.herd = np.vstack((self.herd,organism.reshape((1,organism.size))))
		else:
			self.herd = organism.reshape((1,organism.size))
		return self.herd.shape[0]

	def cullHerd(self):
		self.herd = np.array([])
		return self.herd.shape[0]

	def cullOrganism(self, ind_to_cull):
		if ind_to_cull <self.herd.shape[0]:
			l = range(self.herd)
			l.remove(ind_to_cull)
			if self.herd.shape[0]==self.fitness.size:
				self.fitness = self.fitness[l]
			self.herd = self.herd[l,:]
		else:
			print "Specified organism is not in herd"
		return self.herd.shape[0]

	def replaceOrganism(self,ind_to_replace, new_org):
		if ((ind_to_replace >= 0) and (ind_to_replace<self.herd.shape[0])):
			self.herd[ind_to_replace,:] = new_org
		else:
			print "Invalid organism index for replacement"
		return self.herd.shape[0]

	def populateRandom(self, num, shape):
		if self.herd.size == 0:
			self.organismShape = shape
			self.herd = np.random.random((num,np.array(shape).prod()))
			return self.herd.shape[0]
		else:
			print "Herd is non-empty. Aborting populateRandom"
		return self.herd.shape[0]

	def fitnessTest(self,f):
		self.fitness = np.zeros(self.herd.shape[0])
		for org_ind in range(self.herd.shape[0]):
			self.fitness[org_ind] = f(self.herd[org_ind,:].reshape(self.organismShape))
		self.fitness = self.fitness/self.fitness.max()
		return None

	def averageMate(self,org_inds):
		if (np.array(org_inds)>=self.herd.shape[0]).any():
			print "Mating index out of bounds"
			return np.array([])
		elif len(org_inds)<2:
			print "Not enough organisms to mate"
			return np.array([])
		else:
			return self.herd[org_inds,:].mean(0)

	def crossoverMate(self,org_inds):
		if (np.array(org_inds)>=self.herd.shape[0]).any():
			print "Mating index out of bounds"
			return np.array([])
		elif len(org_inds)<2:
			print "Not enough organisms to mate"
			return np.array([])
		else:
			new_org = np.zeros(self.herd[org_inds[0],:].shape)
			crossoverArray = np.floor(len(org_inds)*np.random.random(new_org.shape))
			for place in range(new_org.size):
				new_org[place] = self.herd[crossoverArray[place],:][place]
			return new_org

	@staticmethod
	def spotMutate(num,rate):
		if np.random.random() < rate:
			return np.random.random()
		else:
			return num

	def mutateOrganism(self,org):
		orgshape = org.shape
		org = org.reshape(org.size)
		for place in range(org.size):
			org[place] = self.spotMutate(org[place],self.mutation_rate)
		return org.reshape(orgshape)

	def generation(self,fitnessFunc, cullThreashold=0.9,mateType=0):
		self.fitnessTest(fitnessFunc)
		worst = np.where(self.fitness>cullThreashold)[0].astype(int)
		# for this implementation, all that dont suck are best
		best = np.setdiff1d(np.array(range(self.fitness.size)),worst).astype(int)
		if best.size==0:
			print "No organisms fit enough to mate"
			fitnessesCopy = self.fitness
			b1 = np.argmin(fitnessesCopy)
			fitnessesCopy[b1] = fitnessesCopy.max()
			b2 = np.argmin(fitnessesCopy)
			print "Mating best we've got instead"
			best = np.array([b1, b2])
		new_org = []
		for i in range(worst.size):
			if mateType != 1:
				new = self.crossoverMate(best.tolist())
			else:
				new = self.averageMate(best.tolist())
			new_org.append(self.mutateOrganism(new))
		for i in range(len(new_org)):
			self.replaceOrganism(worst[i],new_org[i])
		print "%d replacements" % len(new_org)
		return self.herd.shape[0]

	def batchGeneration(self, batches, fitFunc, cullThreashold=[0.9],mateType=[0]):
		"""Takes all arguments as lists. Length of fitFunc is batch size.
		All other inputs are lengths of the same length or just a single value"""
		batchsize = len(fitFunc)
		if len(cullThreashold)!=batchsize:
			cullThreashold = cullThreashold*np.ones(batchsize)
		if len(mateType)!=batchsize:
			mateType = mateType*np.ones(batchsize)

		for batch in range(batches):
			for func_ind in range(batchsize):
				self.generation(fitFunc[func_ind%batchsize], cullThreashold[func_ind%batchsize], mateType[func_ind%batchsize])
				print "Batch: {0}, Test: {1}, AverageFitness: {2}".format(batch, func_ind, self.fitness.mean())
		return self.herd.shape[0]


