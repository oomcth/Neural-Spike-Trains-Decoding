import numpy as np
import random as rnd
from math import exp, log
import networkx as nx
import matplotlib.pyplot as plt
import itertools


np.random.seed = 42

N = 100
steps = 1000
T = 10
delta = T/steps
x = 0 * np.random.random(steps*N)


b = log(20) * np.ones(N)

graph = nx.Graph()
graph.add_nodes_from([2, 3])

while not (nx.is_connected(graph)):
    print("calcul de la matrice")
    graph = nx.generators.random_graphs.dense_gnm_random_graph(N, 0.05*N**2)
print("fait")
M = nx.convert_matrix.to_numpy_array(graph)

for i in range(N):
    M[i][i] = -1.5
    for j in range(N):
        if M[i][j] == 1:
            M[i][j] = np.random.normal(0.01, 0.3, 1)

# print(max([abs(e) for e in np.linalg.eigvals(M)]))


def h(i, j, t, tjs):
    if M[i][j] != 0:
        temp = 0
        for tj in tjs[1:]:
            if t-tj < 1 and t > tj:
                temp += exp(-3*(t-tj))
        return M[i][j] * temp
    else:
        return 0


def K(i, t, N):
    #  if x[t*N+i] != 0:
    #      return int(i <= N*0.05) * x[t*N+i]
    return 0


def lam(k, t, spikes):
    temp = (b[k] + K(k, t, N) +
            sum(h(k, j, t, spikes[j]) for j in range(N)))
    if temp > log(200):
        print("aie")
    if temp > log(500):
        print(k, t)
        raise "lambda trop grand"
    return exp(temp)  # min(temp, log(50)))


def sim():
    s = 0
    spikes = [[1] for i in range(N)]
    count = np.zeros(N)
    histo = np.zeros((N, steps))
    histoStart = 0
    while s < T:
        lambda1 = sum([lam(i, s, spikes) for i in range(N)])
        u = rnd.random()
        w = - log(u) / lambda1
        s += w
        d = rnd.random()
        mem = [lam(i, s, spikes) for i in range(N)]
        lambda2 = sum(mem)
        if d * lambda1 < lambda2 and s < T:
            k = 0
            temp = mem[k]
            for neuron in range(N):
                for time in range(histoStart, int(steps * s / T)):
                    histo[neuron][time] = mem[neuron]
            histoStart = int(steps * s / T)
            while temp < d * lambda1:
                k += 1
                temp += mem[k]
            count[k] += 1
            spikes[k].append(s)
    return spikes, count, histo


print("simulation...")
(a, b, c) = sim()
print("ok")


fig, ax = plt.subplots(1, 1, figsize=[15, 15])
ax.plot(range(len(c[0])), c[0])
for i in a[0]:
    ax.vlines(i*steps/T, 0, 10, colors="red")
plt.show()

fig, ax = plt.subplots(1, 1, figsize=[15, 15])

# ax.axvspan(light_onset_time, light_offset_time, alpha=0.5, color='greenyellow')

for i in range(10):
    ax.vlines(a[i], i - 0.5, i + 0.5)

ax.set_xlim([-1, T])

ax.set_ylabel('Neuron')

ax.set_title('Neuronal Spike Times (sec)')

plt.show()


plt.hist(list(itertools.chain(*a[1:80])), density=False, bins=500)
plt.xlabel("time in sec")
plt.ylabel("number of spikes")
plt.vlines([2], -10, 100, colors=["black"])
plt.show()


def logp(sim):
    spikes, count, histo = sim
    temp = 0
    for i in range(N):
        temp += sum([log(logl) for logl in spikes[i]])
        temp -= sum([f for f in histo[i]]) / steps
    return temp


print("logp :")
print(logp((a, b, c)))
