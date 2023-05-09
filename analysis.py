import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from math import sin, pi
from simu import Simulator
from Tools import Tools
import copy


N = 100  # Number of neurons
steps = 1000  # in how many 'steps' is the time divided
T = 10  # how long does the simulation run for (in s or an arbitrary unit)
xreal = np.zeros((N, steps))  # real stimulus received by the neurons
receptor = 0.4  # proportion of neurons that receive the input
intensity = 1  # intensity of the input


# creating the input xreal
for i in range(int(receptor*N)):
    for j in range(200, steps-200):
        xreal[i][j] = intensity * (2*(j-200)/steps + sin(2*pi*5*j/steps))
    for j in range(steps-200, steps-100):
        xreal[i][j] = xreal[i][steps - 200 - 1] * (1 - (j-steps+200) / 100)
    for j in range(steps-100, steps):
        xreal[i][j] = 0

count = 100000
steps = 1000

HMC = np.zeros((count, steps))

# Global variable

N = 100  # Number of neurons
steps = 1000  # in how many 'steps' is the time divided
T = 10  # how long does the simulation run for (in s or an arbitrary unit)
xreal = np.zeros((N, steps))  # real stimulus received by the neurons
receptor = 0.4  # proportion of neurons that receive the input
intensity = 1  # intensity of the input


# creating the input xreal
for i in range(int(receptor*N)):
    for j in range(200, steps-200):
        xreal[i][j] = intensity * (2*(j-200)/steps + sin(2*pi*5*j/steps))
    for j in range(steps-200, steps-100):
        xreal[i][j] = xreal[i][steps - 200 - 1] * (1 - (j-steps+200) / 100)
    for j in range(steps-100, steps):
        xreal[i][j] = 0


print("loading data...")
for i in tqdm(range(count)):
    if os.path.exists('HMC/' + str(i) + '.npy'):
        HMC[i] = np.load('HMC/' + str(i) + '.npy',
                         allow_pickle=True,
                         fix_imports=True)
    else:
        HMC[i] = HMC[i-1]

oui = copy.copy(HMC)

# def corr(M):
#     temp = np.zeros(count)
#     mean = np.mean(temp[:])
#     M -= mean
#     for i in tqdm(range(count)):
#         temp[i] = sum([np.dot(M[t+i], M[t]) for t in range(count - i - 1)])
#     return (temp - min(temp)) / max(temp)

if os.path.exists('simData/spikes.npy'):
     spikes = np.load('simData/spikes.npy',
                     allow_pickle=True,
                     fix_imports=True)
if os.path.exists('simData/histo.npy'):
    hist = np.load('simData/histo.npy',
                   allow_pickle=True,
                   fix_imports=True)
if os.path.exists('simData/adjMat.npy'):
    adjMat = np.load('simData/adjMat.npy',
                   allow_pickle=True,
                   fix_imports=True)
if os.path.exists('simData/counts.npy'):
    c = np.load('simData/counts.npy',
                   allow_pickle=True,
                   fix_imports=True)

sim = Simulator(N, steps, 0.4, xreal, 10, False)
sim.spikes = spikes
sim.histo = hist
sim.M = adjMat
t = Tools(sim)
#t.ComputeLambdas()
#t.gradientDescent()


print("computing autocorrelation...")
# temp = corr(HMC)
temp = np.correlate([HMC[i][553] for i in range(count)], [HMC[i][600] for i in range(count)], mode='full')
temp = temp[len(temp)//2:]
temp /= temp[0]
plt.plot(range(len(temp)), temp)
plt.show()

plt.plot(range(count-1), [HMC[i][500] for i in range(1, count)])
plt.plot(range(count-1), [HMC[i][600] for i in range(1, count)], color='orange')
# plt.plot(range(count-1), [xreal[0][600] for i in range(count-1)])

plt.show()

plt.plot(range(steps), oui[99000][:])
plt.plot(range(steps), xreal[0])
plt.show()

#a = [-0.5 + np.mean(copy.copy(oui)[90000:99999][i]) for i in range(steps)]
a = [sum([oui[i][:] for i in range(90000, count)])][0] / 10000


plt.plot(range(steps), [i-0.5 for i in a], color='black')
# plt.plot(range(steps), oui[4000], color='orange')
plt.plot(range(steps), xreal[0])
plt.show()

continuousSol = copy.copy(a)
for i in range(16, steps-16):
    continuousSol[i] = np.mean(a[i-15:i+15])
azgezgz= copy.copy(continuousSol)
for i in range(16, steps-16):
    azgezgz[i] = np.mean(continuousSol[i-15:i+15])
plt.plot(range(steps), [i - 0.5 for i in continuousSol])
plt.plot(range(steps), xreal[0])
plt.show()
plt.plot(range(steps), [i - 0.5 for i in azgezgz ])
plt.plot(range(steps), xreal[0])
plt.show()

# RWM : 440 400 500
