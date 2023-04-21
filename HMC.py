import numpy as np
import random as rnd
from math import exp, log, isnan, sin, pi, sqrt
import networkx as nx
import matplotlib.pyplot as plt
import copy
import numba
import random
import itertools
import simu
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
        chaine[0] = self.start
        step = 0

        print("running HMC chain")
        for i in tqdm(range(self.count)):
            # plt.plot(range(steps), chaine[count])
            # plt.show()
            # print(i, step)
            # print(chaine[i][460])

            xtest = np.copy(chaine[i])
            xinit = copy.copy(xtest)
            ztest = 0.5 * np.random.multivariate_normal(np.zeros(self.steps),
                                                        np.identity(self.steps))
            zinit = copy.copy(ztest)
            for j in range(self.L):
                ztest += (self.sigma / 2) * self.tools.gradtot(copy.copy(xtest))
                xtest += self.sigma * ztest
                ztest += (self.sigma / 2) * self.tools.gradtot(copy.copy(xtest))

            u = log(random.random())

            if u < -(self.H(xtest, ztest) - self.H(xinit, zinit)):
                chaine[i+1] = xtest
                step += 1
            else:
                chaine[i+1] = xinit
        print("ratio = ", step*100/self.count, "%")
        self.chain = chaine
        return chaine
