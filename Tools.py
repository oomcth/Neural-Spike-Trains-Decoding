import numpy as np
import random as rnd
from math import exp, log, isnan, sin, pi, sqrt
import networkx as nx
import matplotlib.pyplot as plt
import copy
import numba
import random
import simu
from tqdm import tqdm


@numba.njit()
def modified_hist(x, xreal, N, steps, c, prop):
    vect = (x-xreal)
    temp = c
    for i in range(int(prop*N)):
        for j in range(steps):
            temp[i][j] = temp[i][j] * exp(vect[i][j])
    return temp


class Tools:

    def __init__(self, simulation, save=False) -> None:
        self.save = save
        self.N = simulation.N
        self.steps = simulation.steps
        self.T = simulation.T
        self.receptors = simulation.receptors
        self.inputs = simulation.inputs
        self.relevant = [[i] for i in range(simulation.N)]
        self.base = log(20)
        self.simu = simulation
        self.graddata = []
        self.delta = self.T/self.steps

        # important pour numba
        modified_hist(np.zeros((1, 1)), np.zeros(self.steps), 1,
                      1, copy.copy(simulation.histo), self.receptors)

        NormalM = 10 * np.identity(self.steps)
        self.mu = np.zeros(self.steps)
        for i in range(self.steps):
            if i < self.steps-20 and i > 20:
                for t in range(1, 20):
                    NormalM[i][i+t] = 0.4
                    NormalM[i+t][i] = 0.4
            if i > 20:
                for t in range(1, 20):
                    NormalM[i][i-t] = 0.4
                    NormalM[i-t][i] = 0.4

        print("inverting Normal Matrix")
        NormalM = np.linalg.inv(NormalM)
        print("done")
        self.NormalMatrixInv = NormalM

    def gradlogapriori(self, x):
        temp = - np.dot(self.NormalMatrixInv, (x-self.mu))
        return temp

    def logapriori(self, x):
        return -np.dot(np.transpose(x-self.mu), np.dot(self.NormalMatrixInv, (x-self.mu)))

    def ComputeLambdas(self):
        print("computing lambdas...")
        lambdas = np.zeros((self.N, self.steps))
        for i in tqdm(range(self.N)):
            for j in range(self.steps):
                lambdas[i][j] = self.simu.lam(i, j*self.T/self.steps,
                                              self.simu.spikes,
                                              np.zeros((self.N, self.steps)))
        # plt.plot(range(self.steps), lambdas[0])
        # plt.plot(range(self.steps), self.simu.histo[0])
        # plt.show()
        self.lambdas = lambdas
        print("done")

    def logp(self, simulation):
        spikes = simulation[0]
        histo = simulation[2]
        temp = 0
        for i in range(self.N):
            temp += sum([log(max(histo[i][int(t*self.steps/self.T)], 0.01)) for t in spikes[i]])
            temp -= sum([f for f in histo[i]]) / self.steps
        return temp

    def gradientDescent(self, x0=[], eta=0.1, max_iter=400, tol=0.001):
        grad_f = self.gradtot
        if x0 == []:
            x = np.zeros(self.steps)
        else:
            x = copy.copy(x0)
        print("Gradient descent...")
        for i in tqdm(range(max_iter)):
            grad = -grad_f(x)
            norm_grad = np.linalg.norm(grad)
            if norm_grad < tol:
                break
            x -= eta * grad / sqrt(i+1)
            for i in range(len(x)):
                if x[i] < -5:
                    x[i] = -5
                elif x[i] > 5:
                    x[i] = 5
            # plt.plot(range(len(x)), grad_f(x))
            # plt.title("x")
            # plt.show()
        self.hess = self.approxhessian(x)
        self.bestWithGradientDescent = x

        continuousSol = copy.copy(x)

        for i in range(16, self.steps-16):
            continuousSol[i] = np.median(x[i-15:i+15])
        self.continuousSol = continuousSol
        return x

    def gradSimu(self, x):
        temp = np.zeros(self.steps)
        if self.graddata == []:
            for spikes in self.simu.spikes:
                for spike in spikes:
                    loc = min(int(spike*self.steps/self.T), self.steps-1)
                    temp[loc] += 1
            self.graddata = copy.copy(temp)
        else:
            temp = copy.copy(self.graddata)
        for i in range(self.steps):
            for j in range(int(self.receptors * self.N)):
                temp[i] -= self.lambdas[j][i] * self.delta * exp(x[i])
            for j in range(int(self.receptors * self.N), self.N):
                temp[i] -= self.lambdas[j][i] * self.delta
            if isnan(temp[i]) or temp[i] < -20:
                temp[i] = -20
            if temp[i] > 20:
                temp[i] = 20
        return temp

    def gradtot(self, x):
        return self.gradSimu(x) + 5 * self.gradlogapriori(x)

    def approxhessian(self, x):
        data = (x-self.simu.inputs)[0]

        def hessloggaus():
            return self.NormalMatrixInv

        def hesslogp(x):
            return np.diag([sum([self.simu.histo[j][i] *
                                 self.delta * exp(data[i])
                                 for j in range(self.N)])
                           for i in range(self.steps)])

        return hessloggaus() + hesslogp(x)

    def plotBestGrad(self):
        plt.plot(range(self.steps), self.bestWithGradientDescent)
        plt.plot(range(self.steps), self.simu.inputs[0])
        plt.show()
        plt.plot(range(self.steps), self.continuousSol)
        plt.plot(range(self.steps), self.simu.inputs[0])
        plt.show()
