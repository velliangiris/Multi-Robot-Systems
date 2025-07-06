import math
import random
import matplotlib.pyplot as plt
import numpy as np
from . import optimization


def IoT_env(n_r, n_targets):
    Robots = []
    # Robot start positions (different random starting points)
    start = (0, 0)
    for i in range(n_r):
        Robots.append(Robot(start))
        start = list(start)
        start[0] = start[0] + random.randint(2, 5)
        start = tuple(start)

    # Define the grid size and create a grid with obstacles
    shape = (20, 20)
    grid_array = np.ones(shape, dtype=int)
    total_elements = shape[0] * shape[1]
    num_zeroes = int(0.8 * total_elements)
    zero_indices = np.random.choice(total_elements, num_zeroes, replace=False)
    zero_indices_2d = np.unravel_index(zero_indices, shape)
    grid_array[zero_indices_2d] = 0
    grid = grid_array

    # Generate random target locations (ensure they are different from Robot start positions)
    targets = []
    while len(targets) < n_targets:
        target = (random.randint(0, 19), random.randint(9, 19))
        # Ensure target is not in a Robot's initial position
        if target not in [rob.start_point for rob in Robots]:
            targets.append(target)

    return Robots, grid, targets,grid


class Robot_single:
    def __init__(self, start_point, end_point):
        self.start_point = start_point
        self.end_point = end_point
        self.current_point = start_point
        self.path = [start_point]

    def update_position(self, new_point):
        self.current_point = new_point
        self.path.append(new_point)

    def segment_length(self, index):
        if index < len(self.path) - 1:
            return euclidean_distance(self.path[index], self.path[index + 1])
        else:
            return 0

# UAV class definition and other helper functions (as before)
class Robot:
    def __init__(self, start_point):
        self.start_point = start_point
        self.current_point = start_point
        self.path = [start_point]

    def update_position(self, new_point):
        self.current_point = new_point
        self.path.append(new_point)

    def segment_length(self, index):
        if index < len(self.path) - 1:
            return euclidean_distance(self.path[index], self.path[index + 1])
        else:
            return 0
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_path_length(path):
    length = 0
    for i in range(len(path) - 1):
        length += euclidean_distance(path[i], path[i + 1])
    return length


def generate_obstacles_grid(grid_size, num_obstacles, min_x, max_x, min_y, max_y):
    obstacles = set()
    while len(obstacles) < num_obstacles:
        x = random.randint(min_x, max_x)
        y = random.randint(min_y, max_y)
        obstacles.add((x, y))
    return obstacles


def plot_trajectory(rob, obstacles):
    x = [0]
    y = [20]
    for point in rob.path:
        x.append(point[0])
        y.append(point[1])
    x.append(20)
    y.append(0)
    plt.plot(x, y, marker='.')

    for obstacle in obstacles:
        plt.gca().add_patch(plt.Rectangle(obstacle, 1, 1, fill=None, edgecolor='red'))

    plt.title('Best Trajectory')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()


def path_segment_length_constraint(rob, l_min):
    for i in range(len(rob.path) - 1):
        if rob.segment_length(i) < l_min:
            return False
    return True


def distance_constraint(uavs, L_max):
    total_length = sum(calculate_path_length(rob.path) for rob in uavs)
    return total_length <= L_max


def collision_avoidance_between_uavs(uavs, d_min):
    for i in range(len(uavs)):
        for j in range(i + 1, len(uavs)):
            if euclidean_distance(uavs[i].current_point, uavs[j].current_point) < d_min:
                return False
    return True



def generate_random_trajectory(num_points=50):
    return [(random.uniform(0, 20), random.uniform(0, 20)) for _ in range(num_points)]





def segment_intersects_obstacle(segment, obstacle):
    (x1, y1), (x2, y2) = segment
    (ox, oy) = obstacle

    # Check if line intersects the rectangle edges of the grid cell (obstacle)
    if (line_intersects_line((x1, y1), (x2, y2), (ox, oy), (ox + 1, oy)) or
            line_intersects_line((x1, y1), (x2, y2), (ox, oy), (ox, oy + 1)) or
            line_intersects_line((x1, y1), (x2, y2), (ox + 1, oy), (ox + 1, oy + 1)) or
            line_intersects_line((x1, y1), (x2, y2), (ox, oy + 1), (ox + 1, oy + 1))):
        return True
    return False


def line_intersects_line(p1, p2, p3, p4):
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    def on_segment(p, q, r):
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    if (o1 != o2 and o3 != o4) or (o1 == 0 and on_segment(p1, p3, p2)) or (
            o2 == 0 and on_segment(p1, p4, p2)) or (o3 == 0 and on_segment(p3, p1, p4)) or (
            o4 == 0 and on_segment(p3, p2, p4)):
        return True
    return False







class GridMap:
    def __init__(self, grid_size, num_rows):
        self.grid_size = grid_size
        self.num_rows = num_rows
        self.grid_map = [[0 for _ in range(num_rows)] for _ in range(num_rows)]  # Initialize grid map

    def is_obstacle(self, point):
        row, col = self.convert_to_grid_coordinates(point)
        return self.grid_map[row][col] == 1

    def convert_to_grid_coordinates(self, point):
        x, y = point
        col = int((x / self.grid_size) % self.num_rows)
        row = int((y / self.grid_size) % self.num_rows)
        return row, col





def main(n_r, n_t,Path_Length,Path_Smoothness,fitness):
    n_d = 100
    #   ------- IoT enviornment

    Robots,obstacles, targets,grid = IoT_env(n_r, n_t)

    # ------- multiobjective path planning
    path,PL, S, F = optimization.algm(Robots, n_r, n_d, obstacles)




    Path_Smoothness.append(S)
    Path_Length.append(PL)
    fitness.append(F)
