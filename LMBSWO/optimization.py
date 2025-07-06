import random
import time
import numpy as np

rng = np.random.default_rng()
from Main import Fit

# Sort ObjFun.
def SortObjFun(Fit):
    ObjFun = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return ObjFun, index


# Sort the position  according to ObjFun.
def select_prey(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


def fun(X):
    output = sum(np.square(X))
    return output

def generate_trajectory(n_d):
    return [(random.uniform(0, 20), random.uniform(0, 20)) for _ in range(n_d)]
def fit(N, X):
    ObjFun = np.zeros([N, 1])
    for i in range(N):
        ObjFun[i] = CaculateObjFun1(X[i, :], fun)  ###### eqn 3
    return ObjFun


# This function is to initialize
def initial(N, dim, ub, lb):
    X = np.zeros([N, dim])
    for i in range(N):
        for j in range(dim):
            X[i, j] = lb[j] + random.random() * (ub[j] - lb[j])  ###### eqn 2
    return X


# Calculate ObjFun values
def CaculateObjFun1(X, fun):
    ObjFun = fun(X)
    return ObjFun


def algm(robots,n_robots,n_d,obs):
    N = n_d
    t, T = 1, n_robots
    M = 10  # The dimension.
    fl = -10  # The lower bound of the search interval.
    ul = 10  # The upper bound of the search interval.
    F,C = np.random.uniform(-5,5),np.random.uniform(-5,5)
    lb = fl * np.ones([M, 1])
    ub = ul * np.ones([M, 1])
    X = initial(N, M, lb, ub)  ### eqn 1
    Cand_Prey = np.zeros((N, M))

    Xnew = np.zeros([N, M])


    rand = np.random.rand(N, M)
    best_Pos = np.argmax(X, axis=1)
    # best = np.argmax(ObjFun)
    # --------- Genenrate Trajectory
    for uav in robots:
        uav.path = generate_trajectory(n_d)

    while t < T:

        ObjFun, PL, S = Fit.func(X, robots[t],  obs)

        ObjFun, sortIndex = SortObjFun(ObjFun)  # Sort the ObjFun values
        SCP = select_prey(X, sortIndex)  ###### strongest walrusus
        r1 =r3 = random.uniform(0, 2)

        r2 =K= random.uniform(0, 1)
        I = random.uniform(1, 2)
        r = random.uniform(0, 1)
        mu = 100
        sigma = 50

        for i in range(N-1):
            ######### phase 1 :::::: Position Identification and Hunting of Insects
            if r <0.5:
                W = r1 *(X[i,:]+2*K)   ########eqn 4
                Xnew[i+1,:] = X[i,:] + W *(2*K +r2)  ##### eqn 5

                if (Xnew[i + 1, :] >X[i + 1, :]).any() ==True:
                    Xnew[i + 1, :] = random.gauss(mu, sigma) + r1 *(X[i,:] +2*K /W)  ###### eqn 6
            else:
                ####### phase 2:::::::::::: Carrying the Insect in the Suitable Tube
                W = r3 * (K* best_Pos[i] + r3* X[i,:])  ########eqn 7
                Xnew[i + 1, :] = X[i, :] + W * K ##### eqn 8

                if (Xnew[i + 1, :] > X[i + 1, :]).any() == True:
                    Xnew[i + 1, :] = (r1 +K) * np.sin(F/C)  ###### eqn 9


            K = (1+ (2*(t**2)/T) +F)  ########eqn 10



            ObjFun_p1 = fit(N, X)
            bst = np.argmax(ObjFun_p1)

            # best_Pos[t] =X[t][bst]
        t += 1
    return bst, PL, S,np.mean(ObjFun)



