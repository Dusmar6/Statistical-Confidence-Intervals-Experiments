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
M = 10000


n = 5

conf_95_succ=0
conf_99_succ=0

t_conf_95_succ=0
t_conf_99_succ=0
for num in range(M):
    
    small_x = np.zeros((n,1))
    for i in range(1, n+1):
        X = np.random.normal(mu, std, i)
        small_x[i-1] = np.sum(X) / i
        
    small_x_mean = sum(small_x)/len(small_x)
    small_x_std = np.std(small_x)
    
    
    conf_95_pos = small_x_mean + (small_x_std*1.96)
    conf_95_neg = small_x_mean - (small_x_std*1.96)
    if (mu<conf_95_pos and mu>conf_95_neg):
        conf_95_succ+=1
    
    conf_99_pos = small_x_mean + (small_x_std*2.58)
    conf_99_neg = small_x_mean - (small_x_std*2.58)
    if (mu<conf_99_pos and mu>conf_99_neg):
        conf_99_succ+=1
    
    t_conf_95_pos = small_x_mean + (small_x_std*2.78)
    t_conf_95_neg = small_x_mean - (small_x_std*2.78)
    if (mu<t_conf_95_pos and mu>t_conf_95_neg):
        t_conf_95_succ+=1
    
    t_conf_99_pos = small_x_mean + (small_x_std*4.6)
    t_conf_99_neg = small_x_mean - (small_x_std*4.6)    
    if (mu<t_conf_99_pos and mu>t_conf_99_neg):
        t_conf_99_succ+=1

conf_95_perc = conf_95_succ/M
conf_99_perc = conf_99_succ/M

t_conf_95_perc = t_conf_95_succ/M
t_conf_99_perc = t_conf_99_succ/M

print("n= ", n)
print("conf_95_perc: ",conf_95_perc)
print("conf_99_perc: ",conf_99_perc)

print("t_conf_95_perc: ",t_conf_95_perc)
print("t_conf_99_perc: ",t_conf_99_perc)



n = 40

conf_95_succ=0
conf_99_succ=0

t_conf_95_succ=0
t_conf_99_succ=0
for num in range(M):
    
    small_x = np.zeros((n,1))
    for i in range(1, n+1):
        X = np.random.normal(mu, std, i)
        small_x[i-1] = np.sum(X) / i
        
    small_x_mean = sum(small_x)/len(small_x)
    small_x_std = np.std(small_x)
    
    
    conf_95_pos = small_x_mean + (small_x_std*1.96)
    conf_95_neg = small_x_mean - (small_x_std*1.96)
    if (mu<conf_95_pos and mu>conf_95_neg):
        conf_95_succ+=1
    
    conf_99_pos = small_x_mean + (small_x_std*2.58)
    conf_99_neg = small_x_mean - (small_x_std*2.58)
    if (mu<conf_99_pos and mu>conf_99_neg):
        conf_99_succ+=1
    
    t_conf_95_pos = small_x_mean + (small_x_std*2.78)
    t_conf_95_neg = small_x_mean - (small_x_std*2.78)
    if (mu<t_conf_95_pos and mu>t_conf_95_neg):
        t_conf_95_succ+=1
    
    t_conf_99_pos = small_x_mean + (small_x_std*4.6)
    t_conf_99_neg = small_x_mean - (small_x_std*4.6)    
    if (mu<t_conf_99_pos and mu>t_conf_99_neg):
        t_conf_99_succ+=1

conf_95_perc = conf_95_succ/M
conf_99_perc = conf_99_succ/M

t_conf_95_perc = t_conf_95_succ/M
t_conf_99_perc = t_conf_99_succ/M

print("n= ", n)
print("conf_95_perc: ",conf_95_perc)
print("conf_99_perc: ",conf_99_perc)

print("t_conf_95_perc: ",t_conf_95_perc)
print("t_conf_99_perc: ",t_conf_99_perc)




