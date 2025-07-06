import matplotlib.pyplot as plt
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import heapq
from Proposed import run_main

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
def route_finder(grid, start, goal):
    def heuristic(a, b):
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)

            if 0 <= neighbor[0] < grid.shape[0]:
                if 0 <= neighbor[1] < grid.shape[1]:
                    if grid[neighbor[0]][neighbor[1]] == 1:  # obstacle detected
                        continue
                else:
                    continue
            else:
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return False

def plot_multiple_trajectories(uavs, grid, targets):
    # Plot the environment grid
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set the background color to 'thistle' (light purple)
    ax.set_facecolor('thistle')  # Set the background color to thistle

    # Plot the grid with obstacles
    ax.imshow(grid, cmap=plt.cm.Dark2, alpha=0.5)  # Grid remains the same with obstacles

    # Define colors for the UAV paths
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'yellow']

    # Plot each UAV's path
    for i, uav in enumerate(uavs):
        x_coords, y_coords = zip(*uav.path)
        ax.plot(y_coords, x_coords, color=colors[i % len(colors)], label=f"Robot {i + 1}")

    # Plot all targets as stars, ensuring they're labeled only once
    for i, target in enumerate(targets):
        ax.scatter(target[1], target[0], marker="*", color="red", s=200)

    # Display the legend
    ax.legend()
    plt.show()

def sim_run(Robots,grid,targets,n_r, n_t):
    # For each UAV, find the closest target and calculate the path to it
    for uav in Robots:
        # Find the closest target
        closest_target = min(targets, key=lambda target: euclidean_distance(uav.start_point, target))

        # Use A* to find a path to the closest target
        def get_route(grid):

            route = route_finder(grid, uav.start_point, closest_target)
            if route:
                uav.path = route
            else:
                _,_,_,grid = run_main.IoT_env(n_r, n_t)
                get_route(grid)



        get_route(grid)

    # Plot the UAVs' paths towards the shared goal
    plot_multiple_trajectories(Robots, grid, targets)