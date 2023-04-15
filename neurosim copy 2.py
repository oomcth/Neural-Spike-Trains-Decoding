import numpy as np
import random as rnd
from math import exp, log, isnan, sin, pi
import networkx as nx
import matplotlib.pyplot as plt
import copy
import numba


np.random.seed = 42

N = 100
steps = 10000
T = 10
delta = T/steps
x = np.zeros((N, steps))

Istudy = range(200, 900)
liminf = -5
limsup = 5

sensitive = 0.2

xinf = 3
xmax = 5
intensity = 4

xreal = copy.copy(x)

relevant = [[i] for i in range(N)]


for i in range(int(sensitive*N)):
    for j in range(int(xinf*steps/T), int(xmax*steps/T)):
        xreal[i][j] = 4 * sin(2*pi*(j - int(xinf*steps/T)) / (int(xmax*steps/T) - int(xinf*steps/T)))
    for j in range(int(xmax*steps/T), 7000):
        x[i][j] = 2
        xreal[i][j] = 2

for i in range(int(sensitive*N)):
    for j in range(int(xinf*steps/T), int(xmax*steps/T)):
        x[i][j] = intensity * sin(2*pi*(j - int(xinf*steps/T)) / (int(xmax*steps/T) - int(xinf*steps/T)))

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
    relevant[i].append(i)
    for j in range(N):
        if M[i][j] == 1:
            relevant[j].append(i)
            M[i][j] = np.random.normal(0.01, 0.25, 1)


# print(max([abs(e) for e in np.linalg.eigvals(M)]))

NormalM = 3 * np.identity(steps)
mu = np.zeros(steps)
for i in range(steps):
    if i < steps-1:
        NormalM[i][i+1] = 1.4
    if i > 0:
        NormalM[i][i-1] = 1.4
print("invertion")
NormalM = np.linalg.inv(NormalM)
print("ok")


def gradlogapriori(x):
    temp = - np.dot(NormalM, (x-mu))
    return temp


def logapriori(x):
    return -np.dot(np.transpose(x-mu), np.dot(NormalM, (x-mu)))


def h(i, j, t, tjs):
    if M[i][j] != 0:
        temp = 0
        for tj in tjs[1:]:
            if t-tj < 1 and t > tj:
                temp += exp(-3*(t-tj))
        return M[i][j] * temp
    else:
        return 0


def K(i, t, input):
    return input[i][min(steps-1, int(t*steps/T))]


def lam(k, t, spikes, inputs):
    temp = (b[k] + K(k, t, inputs) +
            sum(h(k, j, t, spikes[k]) for j in relevant[k]))
    return exp(temp)  # min(temp, log(50)))


def sim():
    s = 0
    spikes = [[1] for i in range(N)]
    count = np.zeros(N)
    histo = np.zeros((N, steps))
    histoStart = 0
    while s < T:
        print(s)
        lambda1 = sum([lam(i, s, spikes, xreal) for i in range(N)])
        u = rnd.random()
        w = - log(u) / lambda1
        s += w
        d = rnd.random()
        mem = [lam(i, s, spikes, xreal) for i in range(N)]
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
(a, _, c) = sim()
print("ok")


print("calcul des lambdas")
lambdas = np.zeros((N, steps))
for i in range(N):
    print(i)
    for j in range(steps):
        lambdas[i][j] = lam(i, j*T/steps, a, np.zeros((N, steps)))

plt.plot(range(len(c[0])), c[0])
print(sum(c[0]))
plt.plot(range(len(c[0])), lambdas[0], color='black')
plt.show()
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
def modified_hist(x, xreal, N, steps, c, prop):
    vect = (x-xreal)
    temp = c
    for i in range(int(prop*N)):
        for j in range(steps):
            temp[i][j] = temp[i][j] * exp(vect[i][j])   #Knjit(i, j*T/steps, N, x-xreal, steps, T))
    return temp


# important pour numba
useless = modified_hist(np.zeros((1, 1)), xreal, 1, 1, copy.copy(lambdas), sensitive)

# print(modified_hist(np.zeros((N, steps)), xreal, N, steps, copy.copy(c)))


# print("logp :")
# print(logp((a, b, c)))
# print(logp((a, b, modified_hist(np.zeros((N, steps)), xreal, N, steps, copy.copy(c)))))
# print(logp((a, b, modified_hist(5 * np.ones((N, steps)), xreal, N, steps, copy.copy(c)))))
# print("ok")


def toMin(inputs):
    data = np.reshape(np.repeat(inputs, N), (N, steps))
    temp = (a, b, modified_hist(data, xreal, N, steps, copy.copy(lambdas), sensitive))
    result = -logp(temp) - logapriori(inputs[0])
    return result


def minimize(f, grad_f, x0, eta=0.01, alpha=0.5, beta=0.5, max_iter=100, tol=0.1):
    """
    Performs variable step gradient descent to minimize a convex function f given its gradient grad_f, 
    a starting point x0, a learning rate eta, and other parameters alpha, beta, max_iter, and tol.

    Parameters:
    f (function): The convex function to minimize.
    grad_f (function): The gradient of the convex function f.
    x0 (ndarray): The starting point for the optimization.
    eta (float): The initial learning rate. Default is 1.0.
    alpha (float): The decrease factor for the learning rate. Default is 0.5.
    beta (float): The increase factor for the learning rate. Default is 0.5.
    max_iter (int): The maximum number of iterations. Default is 1000.
    tol (float): The tolerance for convergence. Default is 1e-6.

    Returns:
    ndarray: The optimal point that minimizes the function f.
    """
    x = x0
    for i in range(max_iter):
        print(i)
        grad = -grad_f(x)
        norm_grad = np.linalg.norm(grad)
        if norm_grad < tol:
            break
        eta_k = eta
        while f(x - eta_k * grad) > f(x) - alpha * eta_k * norm_grad**2:
            eta_k *= beta
        
        
        # plt.plot(range(steps), - eta_k * grad)
        # plt.plot(range(steps), grad)
        # plt.plot(range(steps), gradlogapriori(x))
        # plt.title(x[10])
        # plt.legend()
        # plt.show()
        # plt.plot(range(len(x)), grad)
        # plt.title("grad")
        # plt.show()
        x = x - 0.2 * grad
        for i in range(len(x)):
            if x[i] < -5:
                x[i] = -5
            elif x[i] > 5:
                x[i] = 5
        # plt.plot(range(len(x)), x)
        # plt.title("x")
        # plt.show()
    return x


def grad(x):
    data = (x - xreal)[0]
    step = T/steps
    temp = np.zeros(steps)
    for spikes in a:
        for spike in spikes:
            loc = min(int(spike*steps/T), steps-1)
            temp[loc] += 1
    for i in range(steps):
        for j in range(N):
            temp[i] -= c[j][i] * step * exp(data[i])
        if isnan(temp[i]) or temp[i] < -20:
            temp[i] = -20
        if temp[i] > 20:
            temp[i] = 20
    return temp


def realsearch():
    data = (-xreal)[0]
    step = T/steps
    temp = np.zeros(steps)
    for spikes in a:
        for spike in spikes:
            loc = min(int(spike*steps/T), steps-1)
            temp[loc] += 1
    for i in range(steps):
        z = 0
        for j in range(N):
            z += c[j][i] * step * exp(data[i])
        temp[i] /= z
    for i in range(steps):
        print(temp[i])
    return [log(max(tttt, 0.001)) for tttt in temp]


def gradtot(x):
    return grad(x) + gradlogapriori(x)


# temp = gradtot(np.zeros(steps))
# plt.plot(range(len(temp)), temp)
# temp = grad(np.zeros(steps))
# plt.plot(range(len(temp)), temp)
# plt.show()
# print(toMin(x[0]))
# print(logp((a, b, c)))
# print("tomin")


print("minimisation...")
min_sd_results = minimize(toMin, gradtot, np.zeros(steps))
print("fait")

print(min_sd_results)

aa = copy.copy(min_sd_results)
plt.plot(range(len(aa)), aa)
plt.plot(range(steps), xreal[0])
plt.show()


for i in range(20, steps-22):
    if sum(aa[i+j] for j in range(-15, 15)) > 20:
        aa[i] = aa[i-1]
for i in range(20, steps-22):
    if sum(aa[i+j] for j in range(-15, 15)) > 20:
        aa[i] = aa[i-1]
for i in range(20, steps-22):
    if sum(aa[i+j] for j in range(-15, 15)) > 20:
        aa[i] = aa[i-1]
plt.plot(range(len(aa)), aa)
plt.plot(range(steps), xreal[0])
plt.show()

othertest = copy.copy(min_sd_results)


for i in range(41, steps-22):
    othertest[i] = np.mean(othertest[i-30:i+30])

plt.plot(range(len(othertest)), othertest)
plt.plot(range(steps), xreal[0])
plt.show()
