import random
import numpy as np
import math

Nf = 10  # Normalized factor


def calculate_path_length(path):
    """
    Calculate the total path length based on Equation 14.
    path: List of coordinates [(x1, y1), (x2, y2), ...].
    """
    length = 0
    for i in range(len(path) - 1):
        length += np.sqrt((path[i + 1][0] - path[i][0]) ** 2 + (path[i + 1][1] - path[i][1]) ** 2)
    return length


def calculate_obstacle_avoidance(path, obstacles, R):
    """
    Calculate obstacle avoidance cost based on Equations 16–18.
    path: List of coordinates [(x1, y1), (x2, y2), ...].
    obstacles: List of obstacle coordinates [(x_o1, y_o1), (x_o2, y_o2), ...].
    R: Radius of influence of the obstacle.
    """
    P_obs_total = 0
    for point in path:
        for obs in obstacles:
            d = np.sqrt((point[0] - obs[0]) ** 2 + (point[1] - obs[1]) ** 2)
            if d < R:
                P_obs_total += (1 / d - 1 / R) ** 2
    return P_obs_total


def calculate_path_smoothness(path):
    """
    Calculate the path smoothness (J2) by summing the absolute turning angles.

    Args:
        path (list of tuples): Coordinates of the path [(x1, y1), (x2, y2), ..., (xd, yd)].

    Returns:
        float: Smoothness cost (J2), lower values indicate smoother paths.
    """
    smoothness_cost = 0



    # Iterate through the path to calculate angles
    for i in range(1, len(path) - 1):
        # Calculate angles between vectors Pi-1->Pi and Pi->Pi+1
        angle1 = np.arctan2(path[i][1] - path[i - 1][1], path[i][0] - path[i - 1][0])
        angle2 = np.arctan2(path[i + 1][1] - path[i][1], path[i + 1][0] - path[i][0])

        # Compute the absolute difference of angles
        smoothness_cost += abs(angle2 - angle1)

    return smoothness_cost


def func(soln, uav, obs):
    N =  3 #---- normalization factor
    Fit, PL, PS = [], [], []
    for i in range(len(soln)):
        traj = uav.path

        # ------ calculate path length
        pl = calculate_path_length(traj)
        PL.append(pl)

        # ------ calculate path smoothness
        ps = calculate_path_smoothness(traj)
        PS.append(ps)

        # ----- calculate obstacle avoidance
        od = calculate_obstacle_avoidance(traj, obs, 2)

        # ----- calculate fitness
        f = ((1- pl) + ps + od)/N
        Fit.append(f)

    return Fit, np.mean(PL), np.mean(PS)


