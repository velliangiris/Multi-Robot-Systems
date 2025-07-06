import numpy as np
import random
from Main import Fit

def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = lb[j] + random.random() * (ub[j] - lb[j])
    return X


def solution(N, M):  # generating solution
    data = []
    for i in range(len(N)):  # N rows, M columns
        tem = []
        for j in range(M): tem.append(random.random())  # random values
        data.append(tem)
    return data


def Fitn(soln):  # creating function for fitness
    S = []
    for i in range(len(soln)):
        s = 0
        for j in range(len(soln[i])):
            s += soln[i][j] * random.random()
            f = 1 / s
            S.append(f)
        return S
def generate_trajectory(n_d):
    return [(random.uniform(0, 20), random.uniform(0, 20)) for _ in range(n_d)]

def obj_fun(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    F = np.zeros([pop])

    for i in range(pop):
        for j in range(dim):
            X[i, j] = lb[j] + random.random() * (ub[j] - lb[j])
            F[i] = X[i, j] ** 2
    return F


def opt(pop, dim, lb, ub, t, Tmax,robots,n_d,obs):
    best_soln = []

    while (t < Tmax):
        best_cand = np.zeros(Tmax)

        R = 0.2

        p1 = np.zeros([pop, dim])
        p2 = np.zeros([pop, dim])

        c = np.zeros([pop])

        X = initial(pop, dim, lb, ub)
        Fp,PL, S = Fit.func(X, robots[t],obs)
        Positions = np.asarray(solution(X, pop))
        soln = np.array(Fitn(Positions))
        I = random.randint(0, 1)
        rand = random.randint(0, 1)

        for i in range(0, pop):

            Fi = soln[i]
            Fp = np.mean(Fp)
            for j in range(0, dim):
                c[i] = (X[i, j] ** 2) / (sum(X[i, j] ** 2 for i in range(1, pop)))

                if Fp < Fi:
                    K = X[i, j] + random.random() * (Positions[i] - I * X[i, j])
                    p1[i, j] = np.mean(K)

                else:
                    K = X[i, j] + random.random() * (X[i, j] - Positions[i])
                    p1[i, j] = np.mean(K)
            if Fp < Fi:
                X[i] = p1[i]
            else:
                X[i] = X[i]
            for j in range(0, dim):
                K = X[i, j] + R * (1 - (t / Tmax)) * (2 * random.random() - 1) * X[i, j]
                p2[i, j] = np.mean(K)
            if Fp < Fi:
                X[i] = p2[i]
            else:
                X[i] = X[i]
        if p1.all() > p2.all():
            best_cand[t] = p1[0][t]
        else:
            best_cand[t] = p2[0][t]

        best_soln.append(best_cand)
        return best_soln,  PL, S, Fp


def algm(robots,n_robots,n_d,obs):
    pop = n_d
    t, Tmax = 0, n_robots
    dim = 20  # The dimension.
    fl = -10
    ul = 10
    lb = fl * np.ones([dim, 1])
    ub = ul * np.ones([dim, 1])
    path, PL, S,fit = opt(pop, dim, lb, ub, t, Tmax,robots,n_d,obs)
    return path,PL, S,np.mean(fit)










