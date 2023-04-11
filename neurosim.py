import numpy as np
import random as rnd
from math import exp, log
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import copy
from scipy.optimize import minimize
import numba


np.random.seed = 42

N = 100
steps = 1000
T = 10
delta = T/steps
x = np.zeros((N, steps))


sensitive = 0.2

xinf = 3
xmax = 5
intensity = 4

xreal = copy.copy(x)


for i in range(int(sensitive*N)):
    for j in range(int(xinf*steps/T), int(xmax*steps/T)):
        xreal[i][j] = intensity


for i in range(int(sensitive*N)):
    for j in range(int(xinf*steps/T), int(xmax*steps/T)):
        x[i][j] = intensity

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


def K(i, t, N, input=[]):
    if input == []:
        if t > xinf and t < xmax:
            return intensity * int(i <= N*sensitive)
    else:
        return input[i][min(steps-1, int(t*steps/T))]
    return 0


def lam(k, t, spikes, input=[]):
    temp = (b[k] + K(k, t, N, input) +
            sum(h(k, j, t, spikes[j]) for j in range(N)))
    if temp > log(200):
        pass
    if temp > log(1000):
        pass
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
                for time in range(histoStart, int(steps * s / T) + 1):
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


def logp(sim):
    spikes, _, histo = sim
    temp = 0
    for i in range(N):
        temp += sum([log(max(histo[i][int(t*steps/T)], 0.01)) for t in spikes[i]])
        temp -= sum([f for f in histo[i]]) / steps
    return temp


d = []

for i in range(steps):
    if c[0][i] != 0:
        d.append(log(c[0][i] / (exp(log(20) + sum(h(0, j, i*T/steps, a[j]) for j in range(N))))))
    else:
        d.append(0)
# plt.plot(range(steps), d)
# plt.plot(range(len(c[0])), [m for m in c[0]])
# plt.show()



@numba.njit()
def modified_hist(x, xreal, N, steps, c):
    vect = (x-xreal)
    temp = c
    for i in range(N):
        for j in range(steps):
            temp[i][j] = temp[i][j] * exp(vect[i][j])   #Knjit(i, j*T/steps, N, x-xreal, steps, T))
    return temp


# important pour numba
useless = modified_hist(np.zeros((1, 1)), xreal, 1, 1, copy.copy(c))

print(modified_hist(np.zeros((N, steps)), xreal, N, steps, copy.copy(c)))


print("logp :")
print(logp((a, b, c)))
print(logp((a, b, modified_hist(np.zeros((N, steps)), xreal, N, steps, copy.copy(c)))))
print(logp((a, b, modified_hist(5 * np.ones((N, steps)), xreal, N, steps, copy.copy(c)))))
print("ok")


def toMin(input):

    print(np.random.random(1))
    input = np.reshape(input, (N, steps))
    temp = (a, b, modified_hist(input, xreal, N, steps, c))
    return -logp(temp)


print("minimisation...")
bounds = np.array([(0, 5) for i in range(N*steps)])
min_sd_results = minimize(fun=toMin,
                          x0=np.zeros((N * steps)),
                          bounds=bounds)
print("fait")

print(toMin(min_sd_results.x))
print(toMin(np.zeros(N*steps)))
print(min_sd_results.x)

a = copy.copy(min_sd_results.x)
a = np.reshape(a, (N, steps))
plt.plot(range(len(a[1])), a[1])
plt.plot(range(len(x[1])), x[1])
plt.show()
