import random
import numpy as np
from Main import Fit

def initialize(n, m, l, u):

    ran_data = []
    for i in range(n):
        tem = []
        for j in range(m):
            tem.append(random.uniform(0,10) * (u - l) + l)
        ran_data.append(tem)
    return ran_data

def func(soln):

    fit = []
    for i in range(len(soln)):
        fit.append(random.random()) # random fit
    return fit

def random_ran(n):

    ran = []
    for i in range(n):
        ran.append(random.uniform(0,1))
    return ran

def random_ran1(n,c):

    ran = []
    for i in range(len(n)):
        if n[i]<c: a = n[i]
        else: a = random.uniform(0,c)
        ran.append(a)
    return ran

def adaptive_parameter(R2,R3,idx):

    z = []
    for i in range(len(idx)):
        z.append((int(R2) ^ idx[i]) + int(R3[i]) ^ (~idx[i]))
    return z
def generate_trajectory(n_d):
    return [(random.uniform(0, 20), random.uniform(0, 20)) for _ in range(n_d)]

def mew_mean(n,x):
    mean = []
    for i in range(n):
        y = (1 /n) * (sum(x[i]))
        mean.append(y)
    return mean

def euclidean_distance(x,y):
    d = []
    for i in range(len(x)):
        for j in range(len(x[i])):
            D = (np.square(x[i][j] * y[j]))**(1/2)
            d.append(D)
    return d

def prey_position(dis,kbest,pop):
    d = np.argsort(dis)
    pos = []
    for i in range(int(kbest)):
        pos.append(d[i])
    return pos

def optimum_soln(x,C,Z,Ppos,mean):
    soln = []
    for i in range(len(x[0])):
        Soln = x[i]
        soln.append(Soln)
    return soln

def updated_soln(C,Z,mean,pop,Ppos):
    x = []
    for jj in range(len(pop)):
        for k in range(1):
            X = pop[jj][k] + 0.5 * (
                        (2 * C * Z[k] * Ppos[k] - pop[jj][k]) + (2 * (1 - C) * Z[k] * mean[k] - pop[jj][k]))  # eq12a
            x.append(X)
    return x

def updated_soln1(C,Z,Tpos,pop,Ppos):
    x = []
    for i in range(len(Tpos)):
        for j in range(len(Tpos)):
            X = Tpos[i][j] + C * Z[i] * np.cos(2 * np.pi * random.uniform(-1, 1)) * (Tpos[i][j] - pop[i][j])  # eq12b
            x.append(X)
    return x

def algm(robots,n_r,n_d,obs):

    N, M, l, u = n_d, 20, 1, 5
    beta = 0.1
    t, Max_itr = 0, n_r

    pop = initialize(N,M,l,u)                                                                                                       #eq(1)
    n = len(pop)


    while(t<Max_itr):
        fit, PL, S = Fit.func(pop, robots[t],obs)

        R1 = random_ran(len(pop))
        R2 = random.uniform(0,1)
        R3 = random_ran(len(pop))
        C = 1 - t * (0.98/Max_itr)                                                                                                  #eq(5)
        P = random_ran1(R1,C)


        idx=[]
        for j in range(len(R1)):
            p = int(P[j])
            if p==0: idx.append(j)

        Z = adaptive_parameter(R2,R3,idx)                                                                                           #eq(4)
        R5 = random.uniform(0,1)

        mean = mew_mean(n,pop)                                                                                                      #eq(6)
        dis = euclidean_distance(pop,mean)                                                                                           #eq(7)
        kbest = np.round(C * n)
        Ppos = prey_position(dis,kbest,pop)
        Tpos = optimum_soln(pop,C,Z,Ppos,mean)
        if R5 < beta:
            soln = updated_soln(C,Z,mean,pop,Ppos)
        else:
            soln = updated_soln1(C, Z, Tpos, pop, Ppos)
        t=t+1
    return soln, PL, S, np.mean(fit)
