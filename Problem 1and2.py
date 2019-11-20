import numpy as np
import matplotlib.pyplot as plt
import math
import statistics

################ Problem 1

N = 1200000 #total number of bearings

mean = 45 #grams

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
    conf_95_pos[i] = mean + (sigma_small_x[i]*1.96)
    conf_95_neg[i] = mean - (sigma_small_x[i]*1.96)
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
    conf_99_pos[i] = mean + (sigma_small_x[i]*2.58)
    conf_99_neg[i] = mean - (sigma_small_x[i]*2.58)
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

def confidence_success(n, M, mean, std, crit):
    succ = 0
    for i in range(0, M):
        X = np.random.normal(mean, std, n)
        small_x = np.sum(X) / n
        
        temp = np.zeros((n,1))
        for j in range(0,n):
            temp[j] = (X[j]-small_x)**2
        S = math.sqrt(np.sum(temp)/(n-1))
        
        high = small_x + (crit * (S/math.sqrt(n)))
        low = small_x - (crit * (S/math.sqrt(n)))
        
        if (mean<high and mean>low):
            succ+=1
    return succ/M

n1 = 5

n2 = 40

n3 = 120


print("n= ", n1)
print("conf_95_perc: ",confidence_success(n1, M, mean, std, 1.96))
print("conf_99_perc: ",confidence_success(n1, M, mean, std, 2.58))

print("t_conf_95_perc: ",confidence_success(n1, M, mean, std, 2.7763))
print("t_conf_99_perc: ",confidence_success(n1, M, mean, std, 4.6022))


print("\nn= ", n2)
print("conf_95_perc: ",confidence_success(n1, M, mean, std, 1.96))
print("conf_99_perc: ",confidence_success(n1, M, mean, std, 2.58))

print("t_conf_95_perc: ",confidence_success(n1, M, mean, std, 2.0227))
print("t_conf_99_perc: ",confidence_success(n1, M, mean, std, 2.7079))


print("\nn= ", n3)
print("conf_95_perc: ",confidence_success(n1, M, mean, std, 1.96))
print("conf_99_perc: ",confidence_success(n1, M, mean, std, 2.58))

print("t_conf_95_perc: ",confidence_success(n1, M, mean, std, 1.9801))
print("t_conf_99_perc: ",confidence_success(n1, M, mean, std, 2.6178))






