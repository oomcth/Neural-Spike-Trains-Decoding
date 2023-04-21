import numpy as np
import random as rnd
from math import exp, log, isnan, sin, pi, sqrt
import networkx as nx
import matplotlib.pyplot as plt
import copy
import numba
import random
import itertools


class Simulator:

    def __init__(self, N, steps, receptors, inputs, T, save=False, e=0) -> None:
        self.save = save
        self.N = N
        self.steps = steps
        self.T = T
        self.receptors = receptors
        self.inputs = inputs
        self.relevant = [[i] for i in range(N)]
        self.base = log(20)
        self.e = e

    def createAdjacency(self):
        graph = nx.Graph()
        graph.add_nodes_from([2, 3])
        print("calcul de la matrice")
        test = 0
        while not (nx.is_connected(graph)):
            test += 1
            print("try : ", test)
            graph = nx.generators.random_graphs.dense_gnm_random_graph(self.N, 0.05*self.N**2)
        print("fait")
        M = nx.convert_matrix.to_numpy_array(graph)

        for i in range(self.N):
            M[i][i] = -1.5
            self.relevant[i].append(i)
            for j in range(self.N):
                if M[i][j] == 1:
                    self.relevant[j].append(i)
                    M[i][j] = np.random.normal(0.05 + self.e, 0.25, 1)
                    if M[i][j] >= 0.5:
                        M[i][j] = 0.5
                    if M[i][j] <= -0.5:
                        M[i][j] = -0.5
        if self.save:
            with open('simData/adjMat.npy', 'wb') as f:
                np.save(f, M)
        self.M = M

    def LoadAdjacency(self):

        M = np.load('simData/adjMat.npy', allow_pickle=True, fix_imports=True)
        print(self.e)
        for i in range(self.N):
            self.relevant[i].append(i)
            for j in range(self.N):
                if M[i][j] != 0 and i != j:
                    self.relevant[j].append(i)
                    M[i][j] += self.e
                    if M[i][j] >= 0.5:
                        M[i][j] = 0.5
                    if M[i][j] <= -0.5:
                        M[i][j] = -0.5
        self.M = M

    def h(self, i, j, t, tjs):
        if self.M[i][j] != 0:
            temp = 0
            for tj in tjs[1:]:
                if t-tj < 1 and t > tj:
                    temp += exp(-(t-tj))
            return self.M[i][j] * temp
        else:
            return 0

    def K(self, i, t, input):
        if i < self.N * self.receptors:
            return input[i][min(self.steps-1, int(t*self.steps/self.T))]
        else:
            return 0

    def lam(self, k, t, spikes, inputs):
        temp = (self.base + self.K(k, t, inputs) +
                sum(self.h(k, j, t, spikes[k]) for j in self.relevant[k]))
        return exp(temp)

    def normalSimu(self, echo=True):
        s = 0
        spikes = [[1] for i in range(self.N)]
        count = np.zeros(self.N)
        histo = np.zeros((self.N, self.steps))
        histoStart = 0
        minw = 10**(-10)
        while s < self.T:
            if echo:
                print(s)
            lambda1 = sum([self.lam(i, s, spikes, self.inputs) for i in range(self.N)])
            u = rnd.random()
            w = - log(u) / lambda1
            if w <= minw:
                return -1
            s += w
            d = rnd.random()
            mem = [self.lam(i, s, spikes, self.inputs) for i in range(self.N)]
            lambda2 = sum(mem)
            if d * lambda1 < lambda2 and s < self.T:
                k = 0
                temp = mem[k]
                for neuron in range(self.N):
                    for time in range(histoStart, int(self.steps * s / self.T) + 1):
                        histo[neuron][time] = mem[neuron]
                histoStart = int(self.steps * s / self.T)
                while temp < d * lambda1:
                    k += 1
                    temp += mem[k]
                count[k] += 1
                spikes[k].append(s)
        self.spikes = spikes
        self.counts = count
        self.histo = histo
        if self.save:
            with open('simData/spikes.npy', 'wb') as f:
                np.save(f, spikes)
            with open('simData/counts.npy', 'wb') as f:
                np.save(f, count)
            with open('simData/histo.npy', 'wb') as f:
                np.save(f, histo)

        return 0

    def plotHisto(self, num=0):
        _, ax = plt.subplots(1, 1, figsize=[15, 15])

        ax.plot(range(self.steps), self.histo[num])
        for i in self.spikes[num]:
            ax.vlines(i*self.steps/self.T, 0, 10, colors="red")
        ax.plot(range(self.steps), 5 * self.inputs[0])

        plt.show()

    def plotSpikes(self, min=0, max=10):
        _, ax = plt.subplots(1, 1, figsize=[5, 5])

        for i in range(min, max):
            ax.vlines(self.spikes[i], i - 0.5, i + 0.5)
        ax.set_xlim([-1, self.T])
        ax.set_ylabel('Neuron')
        ax.set_title('Neuronal Spike Times (sec)')

        plt.show()

    def plotAgregatedSpikes(self):
        fig, ax = plt.subplots(1, 1, figsize=[15, 15])

        ax.hist(list(itertools.chain(*self.spikes[1:80])),
                density=False, bins=500)
        plt.xlabel("time in sec")
        plt.ylabel("number of spikes")
        ax.vlines([2], -10, 100, colors=["black"])

        plt.show()
