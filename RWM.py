import numpy as np
from math import log
import matplotlib.pyplot as plt
import copy
import random
import Tools
from tqdm import tqdm


class RWM:

    def __init__(self, sim, tools, sigma, count) -> None:
        self.sim = sim
        self.tools = tools
        self.sigma = sigma
        self.count = count
        self.Mat = np.linalg.cholesky(tools.hess)

    def stable(self, x):
        temp = (self.tools.logp((copy.copy(self.sim.spikes),
                                 self.sim.counts,
                                 Tools.modified_hist(copy.copy(x),
                                                     np.zeros((self.sim.N,
                                                               self.sim.steps)),
                                                    self.sim.N, self.sim.steps,
                                                    copy.copy(self.tools.lambdas),
                                                    self.sim.receptors)))
                + self.tools.logapriori(copy.copy(x)))
        return temp

    def next(self, x):
        return x + np.dot(self.sigma * self.Mat,
                          np.random.multivariate_normal(np.zeros(self.sim.steps),
                                                        np.identity(self.sim.steps)))

    def q(self, x, y, m):
        temp = (-1/2) * np.dot(np.dot((x-y).T, self.tools.hess), (x-y))
        return temp

    def ComputeChain(self):
        chaine = np.zeros((self.count + 1, self.sim.steps))
        chaine[0] = self.tools.bestWithGradientDescent
        step = 0

        print("running RWM chain")
        for i in tqdm(range(self.count)):
            u = random.random()
            x = chaine[i]
            y = self.next(x)
            sx = self.stable(x)
            sy = self.stable(y)
            test = (sy - sx)

            if log(u) <= test:
                chaine[i+1] = y
                with open('RWM2/' + str(i) + '.npy', 'wb') as f:
                    np.save(f, y)
                step += 1
            else:
                chaine[i+1] = x
                with open('RWM2/' + str(i) + '.npy', 'wb') as f:
                    np.save(f, y)
            print(step/(i+1))
        print("ratio = ", step*100/self.count, "%")
        print("simulation terminée")
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
