import numpy as np
from math import log
import matplotlib.pyplot as plt
import copy
import random
import Tools
from tqdm import tqdm


class HMC:

    def __init__(self, sim, tools, count, L, sigma) -> None:
        self.sim = sim
        self.tools = tools
        self.start = np.zeros(sim.steps)
        self.sigma = sigma
        self.L = L
        self.count = count
        self.N = sim.N
        self.steps = sim.steps

    def toMin(self, inputs):
        data = np.reshape(np.repeat(inputs, self.N), (self.N, self.steps))
        temp = (self.sim.spikes, self.sim.counts,
                Tools.modified_hist(data, self.sim.inputs, self.N,
                                    self.steps, copy.copy(self.sim.histo),
                                    self.sim.receptors))
        result = self.tools.logp(temp) + self.tools.logapriori(inputs[0])
        return result

    def H(self, x, z):
        return (1/2) * np.linalg.norm(copy.copy(z)) - self.toMin(copy.copy(x))

    def RunChain(self):
        chaine = np.zeros((self.count + 1, self.steps))
        chaine[0] = self.tools.bestWithGradientDescent
        step = 0
        zs = np.random.multivariate_normal(np.zeros(self.steps),
                                           np.identity(self.steps),
                                           self.count+1)

        print("running HMC chain")
        for i in tqdm(range(self.count)):

            xtest = np.copy(chaine[i])
            xinit = copy.copy(xtest)
            ztest = zs[i]
            zinit = copy.copy(ztest)
            for j in range(self.L):
                ztest += (self.sigma / 2) * self.tools.gradtot(copy.copy(xtest))
                xtest += self.sigma * ztest
                ztest += (self.sigma / 2) * self.tools.gradtot(copy.copy(xtest))

            u = log(random.random())
            if u < -(self.H(xtest, ztest) - self.H(xinit, zinit)):
                chaine[i+1] = xtest
                with open('HMC/' + str(i) + '.npy', 'wb') as f:
                    np.save(f, xtest)
                step += 1
            else:
                chaine[i+1] = xinit
                with open('HMC/' + str(i) + '.npy', 'wb') as f:
                    np.save(f, xinit)
            print(step*100/(i+1))
        print("ratio = ", step*100/self.count, "%")
        self.chaine = chaine

        temp = copy.copy([np.mean(chaine[10000:19999, i]) for i in range(self.sim.steps)])
        temp2 = copy.copy(temp)
        for i in range(41, self.sim.steps-32):
            temp[i] = np.mean(temp2[i-30:i+30])
        self.continuous = temp
        return chaine

    def plot(self):
        plt.plot(range(self.sim.steps), self.sim.inputs[0], color='red')
        plt.plot(range(self.sim.steps), [np.mean(self.chaine[10000:19999, i]) for i in range(self.sim.steps)], color='black')
        plt.show()

        plt.plot(range(self.sim.steps), self.sim.inputs[0], color='red')
        plt.plot(range(self.sim.steps), self.continuous, color='black')
        plt.show()
