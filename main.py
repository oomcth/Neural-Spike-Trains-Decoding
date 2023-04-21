from simu import Simulator
from Tools import Tools
import numpy as np
from math import sin, pi
from RWM import RWM
from HMC import HMC
import copy


np.random.seed = 42
N = 100
steps = 1000
T = 10
xreal = np.zeros((N, steps))
receptor = 0.4
intensity = 1


for i in range(int(receptor*N)):
    for j in range(200, steps-200):
        xreal[i][j] = intensity * (2*(j-200)/steps + sin(2*pi*5*j/steps))
    for j in range(steps-200, steps-100):
        xreal[i][j] = xreal[i][steps - 200 - 1] * (1 - (j-steps+200) / 100)
    for j in range(steps-100, steps):
        xreal[i][j] = 0


if True:
    simu = Simulator(N, steps, receptor, xreal, T, e=0.07037)
    simu.LoadAdjacency()
    simu.normalSimu()

if False:
    simu = Simulator(N, steps, receptor, xreal, T, e=0, save=True)
    simu.createAdjacency()
    simu.normalSimu()

if False:
    simu.plotAgregatedSpikes()
    simu.plotHisto(50)
    simu.plotSpikes(30, 50)

tools = Tools(simu)
tools.ComputeLambdas()
tools.gradientDescent()

if False:
    tools.plotBestGrad()

rwm = RWM(simu, tools, 0.1, 20000)
rwm.ComputeChain()
rwm.plot()

hmc = HMC(simu, tools, 10, 3, 0.01)
hmc.RunChain()
