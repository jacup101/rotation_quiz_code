import matplotlib.pyplot as plt
import numpy as np


# Add aspect ratio, to prevent distortion
figure, axes = plt.subplots()
axes.set_aspect(1)
# Set up axis
plt.axis([-10, 10, -10, 10])


def plot_shape(shape, local_coords, global_coords):
    for face in shape:
        plot_coords(face, local_coords, global_coords)


def plot_coords(face, local_coords, global_coords):
    print(global_coords[1])
    for i in range(len(face)):
        current = local_coords[face[i] - 1]
        next = local_coords[face[(i+1) % len(face)] - 1]
        plt.plot([current['x'] + global_coords[0], next['x'] + global_coords[0]], [current['y'] + global_coords[1], next['y'] + global_coords[1]], color='k')

def rotate(local_coords, global_coords, angle, matrix):
    final_coords = []

    for i in range(len(matrix)):
        final_coords.append(np.inner(matrix[i], local_coords))
    return final_coords

def rotate_list(all_coords, shape, global_coords, angle, axis):
    coord_list = []
    for i in range(len(all_coords)):
        coord_list.append(convert_list_to_coord(rotate(convert_coord_to_list(all_coords[i]), global_coords, angle, generate_matrix(axis, angle))))
    print(coord_list)
    plot_shape(shape, coord_list, global_coords)

def convert_coord_to_list(coord):
    return [coord['x'], coord['y'], coord['z']]

def convert_list_to_coord(list_coord):
    return {'x': list_coord[0], 'y': list_coord[1], 'z': list_coord[2]}

def generate_matrix(axis, angle):
    matrices = {
        'x': [[1, 0, 0], [0, np.cos(np.radians(angle)), -1 * np.sin(np.radians(angle))], [0, np.sin(np.radians(angle)), np.cos(np.radians(angle))]],
        'y': [[np.cos(np.radians(angle)), 0, np.sin(np.radians(angle))], [0, 1, 0], [- np.sin(np.radians(angle)), 0, np.cos(np.radians(angle))]],
        'z': [[np.cos(np.radians(angle)), -1 * np.sin(np.radians(angle)), 0], [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0], [0, 0, 1]]
    }
    return matrices[axis]


# Triangle
tri_coords = [
    {'x': 0, 'y' : 1, 'z': 0},
    {'x': -.471404, 'y' : -.33333, 'z': 0.816497},
    {'x': 0.942809, 'y' : -0.333333, 'z': 0},
    {'x': -0.471404, 'y' : -0.333333, 'z': -0.81649}
]

triangle = [
    [1, 3, 2],
    [1, 2, 4],
    [1, 4, 3],
    [3, 4, 2]
]

# Octohedron

oct_coords = [
    {'x': 0, 'y': 0, 'z': 1},
    {'x': 0, 'y': 1, 'z': 0},
    {'x': 1, 'y': 0, 'z': 0},
    {'x': 0, 'y': -1, 'z': 0},
    {'x': -1, 'y': 0, 'z': 0},
    {'x': 0, 'y': 0, 'z': -1}
]
oct = [
    [2, 3, 1],
    [2, 1, 5],
    [2, 5, 6],
    [2, 6, 3],
    [3, 6, 4],
    [3, 4, 1],
    [1, 4, 5],
    [5, 4, 6]
]

cube_coords = [
    {'x': 0, 'y': 1, 'z': 0},
    {'x': 0.942809, 'y': 0.333333, 'z': 0},
    {'x': 0.471404, 'y': -0.333333, 'z': 0.816496},
    {'x': -0.471404, 'y': 0.333333, 'z': 0.816496},
    {'x': -0.942809, 'y': -0.333333, 'z': 0},
    {'x': -0.471404, 'y': 0.333333, 'z': -0.816496},
    {'x': 0.471404, 'y': -0.333333, 'z': -0.816496},
    {'x': 0, 'y': -1, 'z': 0}
]

cube = [
    [8, 7, 6, 5],
    [8, 5, 4, 3],
    [5, 6, 1, 4],
    [3, 4, 1, 2],
    [8, 3, 2, 7],
    [7, 2, 1, 6]
]


rotate_list(oct_coords, oct, [0, 5, 0], 30, 'x')
rotate_list(oct_coords, oct, [0, 0, 0], 30, 'y')
rotate_list(oct_coords, oct, [0, -5, 0], 30, 'z')

rotate_list(cube_coords, cube, [5, 5, 0], 30, 'x')
rotate_list(cube_coords, cube, [5, 0, 0], 30, 'y')
rotate_list(cube_coords, cube, [5, -5, 0], 30, 'z')

rotate_list(tri_coords, triangle, [-5, 5, 0], 30, 'x')
rotate_list(tri_coords, triangle, [-5, 0, 0], 30, 'y')
rotate_list(tri_coords, triangle, [-5, -5, 0], 30, 'z')

#rotate_list_y(coords, triangle, [0, 0, 0], 240)

plt.show()