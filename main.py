from simu import Simulator
from Tools import Tools
import numpy as np
from math import sin, pi
from RWM import RWM
from HMC import HMC


np.random.seed = 42


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


# runs a simulation
if True:
    simu = Simulator(N, steps, receptor, xreal, T, e=0.07037)
    simu.LoadAdjacency()
    simu.normalSimu()


# runs a simulation and saves our result in the computer
if False:
    simu = Simulator(N, steps, receptor, xreal, T, e=0, save=True)
    simu.createAdjacency()
    simu.normalSimu()


# plots data about the simulation
if False:
    simu.plotAgregatedSpikes()
    simu.plotHisto(50)
    simu.plotSpikes(30, 50)


tools = Tools(simu)
# Calculates each neuron's lambda for each time period if there is no input
tools.ComputeLambdas()
# Does a gradient descent to estimate the input
tools.gradientDescent()


# plots the gradient descent approximation of the Maximum A Posteriori (xMAP)
if True:
    tools.plotBestGrad()

# runs a Random Walk Monte Carlo Chain to estimate the bayesian
# estimator of the input
if False:
    rwm = RWM(simu, tools, 0.1, 20000)
    rwm.ComputeChain()
    rwm.plot()

# runs a Hamiltonian Monte Carlo Chain to estimate the bayesian
# estimator of the input
if True:
    hmc = HMC(simu, tools, 20000, 1, 0.008)
    hmc.RunChain()
    hmc.plot()
