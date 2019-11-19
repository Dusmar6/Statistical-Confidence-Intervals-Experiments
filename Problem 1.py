import numpy as np
import matplotlib.pyplot as plt
import math
import statistics

################ Problem 1

N = 1200000 #total number of bearings

mu = 45 #grams

std = 3 #grams

n = 180
#sample sizes = 1 to 180

small_x = np.zeros((n,1))

sigma_small_x = np.zeros((n,1))

sample = np.zeros((n,1))

mu_small_x = np.full((n,1), mu)

for i in range(1, n+1):
    X = np.random.normal(mu, std, i)
    small_x[i-1] = np.sum(X) / i
    sigma_small_x[i-1] = std/math.sqrt(i)
    sample[i-1] = i
    
    
    
print(small_x)
#95
conf_95_pos = np.zeros((n,1))
conf_95_neg = np.zeros((n,1))

for i in range(0,n):
    conf_95_pos[i] = mu + (sigma_small_x[i]*1.96)
    conf_95_neg[i] = mu - (sigma_small_x[i]*1.96)
    plt.plot(sample[i], small_x[i], 'b', marker = 'x')
    
plt.title("Sample mean and confidence interval: 95%")
plt.xlabel("Size")
plt.ylabel("mu")
plt.plot(sample, conf_95_pos, 'r')
plt.plot(sample, mu_small_x, 'k')
plt.plot(sample, conf_95_neg, 'r')
plt.show()

#95
conf_99_pos = np.zeros((n,1))
conf_99_neg = np.zeros((n,1))

for i in range(0,n):
    conf_99_pos[i] = mu + (sigma_small_x[i]*2.58)
    conf_99_neg[i] = mu - (sigma_small_x[i]*2.58)
    plt.plot(sample[i], small_x[i], 'b', marker = 'x')
    
plt.title("Sample mean and confidence interval: 99%")
plt.xlabel("Size")
plt.ylabel("mu")
plt.plot(sample, conf_99_pos, 'g')
plt.plot(sample, mu_small_x, 'k')
plt.plot(sample, conf_99_neg, 'g')
plt.show()










############## Problem 2

n = 5

small_x = np.zeros((n,1))

#sigma_small_x = np.zeros((n,1))
#
#sample = np.zeros((n,1))
#
#mu_small_x = np.full((n,1), mu)


for i in range(1, n+1):
    X = np.random.normal(mu, std, i)
    small_x[i-1] = np.sum(X) / i
#    sigma_small_x[i-1] = std/math.sqrt(i)
#    sample[i-1] = i

    
small_x_mean = sum(small_x)/len(small_x)
small_x_std = np.std(small_x)

conf_95_pos = small_x_mean + (small_x_std*1.96)
conf_95_neg = small_x_mean - (small_x_std*1.96)

t_conf_95_pos = small_x_mean + (small_x_std*2.78)
t_conf_95_neg = small_x_mean - (small_x_std*2.78)








