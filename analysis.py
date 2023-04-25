import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


count = 20000
steps = 1000

HMC = np.zeros((count, steps))


print("loading data...")
for i in tqdm(range(count)):
    if os.path.exists('HMC/' + str(i) + '.npy'):
        HMC[i] = np.load('HMC/' + str(i) + '.npy',
                         allow_pickle=True,
                         fix_imports=True)
    else:
        HMC[i] = HMC[i-1]


# def corr(M):
#     temp = np.zeros(count)
#     mean = np.mean(temp[:])
#     M -= mean
#     for i in tqdm(range(count)):
#         temp[i] = sum([np.dot(M[t+i], M[t]) for t in range(count - i - 1)])
#     return (temp - min(temp)) / max(temp)


print("computing autocorrelation...")
# temp = corr(HMC)
temp = np.correlate([HMC[i][400] for i in range(count)], [HMC[i][400] for i in range(count)], mode='full')
temp = temp[len(temp)//2:]
plt.plot(range(len(temp)), temp)
plt.show()

plt.plot(range(count-1), [HMC[i][400] for i in range(1, count)])
plt.show()


# RWM : 440 400 500