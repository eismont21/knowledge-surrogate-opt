"""
Script for generating and saving shape geometries, stiffness distributions, and strain fields.

This script creates shape geometries based on given node coordinates, calculates stiffness
distributions around various gripper positions, and simulates strain fields due to material
properties and external forces.
"""

import numpy as np
import os
import json
import matplotlib.image as mpli
import scipy.interpolate as spi
import pandas as pd
from tqdm import tqdm

path = os.path.dirname("__file__")
data_path = os.path.join(path, "data")

data_dics = {
    "doubledome": {
        "folder_path": os.path.join(path, "Austausch_IPD_Optimierung"),
        "shape_data_path": os.path.join(path, "Austausch_IPD_Optimierung", "gen_greyscale_image_tool"),
        "spring_coords_path": os.path.join(path, "Austausch_IPD_Optimierung", "spring_coords"),
        "u_spring_data_path": os.path.join(path, "Austausch_IPD_Optimierung", "u_spring"),
        "el_coords_path": os.path.join(path, "Austausch_IPD_Optimierung", "gen_greyscale_shear"),
        "train_data": {
            "x_data_path": os.path.join(path, "Austausch_IPD_Optimierung", "XY-Simulationsdaten_train", "x_vals.npy"),
            "y_data_path": os.path.join(path, "Austausch_IPD_Optimierung", "XY-Simulationsdaten_train", "y_vals.npy"),
        },
        "val_data": {
            "x_data_path": os.path.join(path, "Austausch_IPD_Optimierung", "XY-Simulationsdaten_val", "x_vals.npy"),
            "y_data_path": os.path.join(path, "Austausch_IPD_Optimierung", "XY-Simulationsdaten_val", "y_vals.npy"),
        }
    },
    "L_shape": {
        "folder_path": os.path.join(path, "2023_06"),
        "shape_data_path": os.path.join(path, "2023_06", "01_Generate_Tool_Greyscale_Image"),
        "u_spring_data_path": os.path.join(path, "2023_06", "04_Spring_Elongation"),
        "el_coords_path": os.path.join(path, "2023_06", "02_Generate_Shear_Greyscale_Image"),
        "train_data": {
            "x_data_path": os.path.join(path, "2023_06", "03_Spring_Stiffnesses", "X-Vals_-_Spring_stiffnesses.npy"),
            "y_data_path": os.path.join(path, "2023_06", "02_Generate_Shear_Greyscale_Image",
                                        "Y-Vals_-_ShearAngles_L-Winkel.npy"),
        }
    }
}


def create_shape(shape_name, dic_paths):
    """
    Create a shape image using greyscale parameters.
    """
    shape_data_path = os.path.join(data_path, shape_name, "shape")
    os.makedirs(shape_data_path, exist_ok=True)

    image_tool_path = dic_paths["shape_data_path"]
    abs_path_gs_json = os.path.join(image_tool_path, "tool_greyscale.json")
    abs_path_nodes = os.path.join(image_tool_path, f"nodes_{shape_name}.npy")
    abs_path_gs_img = os.path.join(shape_data_path, "shape.png")
    abs_path_gs_matrix = os.path.join(shape_data_path, "shape.npy")

    # load greyscale parameters:
    with open(abs_path_gs_json, "r") as gs_json:
        gs_param = json.load(gs_json)

    # load nodes of geometry:
    nodes_xyz = np.load(abs_path_nodes)

    x_min = gs_param["bb_x_min"]
    x_max = gs_param["bb_x_max"]
    y_min = gs_param["bb_y_min"]
    y_max = gs_param["bb_y_max"]
    num_pxl_x = gs_param["num_pxl_x"]
    num_pxl_y = gs_param["num_pxl_y"]
    h_min = gs_param["h_min"]
    h_max = gs_param["h_max"]

    x_grid = np.linspace(x_min, x_max, num_pxl_x)
    y_grid = np.linspace(y_min, y_max, num_pxl_y)
    pix_x, pix_y = np.meshgrid(x_grid, y_grid)

    if np.any(nodes_xyz):  # Check if any non-zero values in array (i.e. valid mesh):
        nodes_xyz = np.unique(nodes_xyz, axis=0)
        nodes_xy = nodes_xyz[:, 0:2]
        nodes_z = nodes_xyz[:, 2]

        pix_gs = spi.griddata(nodes_xy, nodes_z, (pix_x, pix_y), method="linear")
    else:
        pix_gs = np.zeros((32, 32))

    mask = pix_gs < 0
    pix_gs[mask] = 0

    mpli.imsave(abs_path_gs_img, pix_gs, cmap="gray", vmin=h_min, vmax=h_max)
    np.save(abs_path_gs_matrix, pix_gs)


for shape_name, dic_paths in data_dics.items():
    create_shape(shape_name, dic_paths)

grippers_path = data_dics["doubledome"]["spring_coords_path"]
grippers_directory_coordinates_xy = np.load(grippers_path + '/spring_dir_of_attack_coords.npy')[:2]
grippers_point_coordinates_xy = np.load(grippers_path + '/spring_pnt_of_attack_coords.npy')[:2]

encoding_data_path = os.path.join(data_path, "encoding")
os.makedirs(encoding_data_path, exist_ok=True)


def stiffness_distribution(x_0, y_0, x_dir, y_dir):
    """
    Calculate the stiffness distribution for given spring parameters.
    """
    x_max = 300
    y_max = 460

    E_1 = 140.0
    E_2 = 10.0
    nu_12 = 0.3
    G_12 = 0.1

    Q_11 = E_1 / (1 - nu_12 ** 2 * E_2 / E_1)
    Q_12 = nu_12 * E_2 / (1 - nu_12 ** 2 * E_2 / E_1)
    Q_22 = E_2 / (1 - nu_12 ** 2 * E_2 / E_1)
    Q_33 = G_12

    def calculate_q11(x, y, _x_0, _y_0):
        eps = 0.1  # prevent division by zero
        if x_dir == -10 and y_dir == 0 and _x_0 == 0:  # spring on left
            alpha_rad = np.arctan((y - _y_0) / (x + eps))
        elif x_dir == 10 and y_dir == 0 and _x_0 == x_max:  # spring on right
            alpha_rad = np.arctan((y - _y_0) / (x - _x_0 + eps)) + np.pi
        elif x_dir == 0 and y_dir == -10 and _y_0 == 0:  # spring on bottom
            x, y = y, x
            _x_0, _y_0 = _y_0, _x_0
            alpha_rad = np.arctan((y - _y_0) / (x + eps))
        elif x_dir == 0 and y_dir == 10 and _y_0 == y_max:  # spring on top
            x, y = y, x
            _x_0, _y_0 = _y_0, _x_0
            alpha_rad = np.arctan((y - _y_0) / (x - _x_0 + eps)) + np.pi
        else:
            raise ValueError('Invalid spring configuration!')
        q_vec = np.array([Q_11, Q_12, Q_22, Q_33])
        cs_vec = np.array([
            np.cos(alpha_rad) ** 4,
            2 * np.cos(alpha_rad) ** 2 * np.sin(alpha_rad) ** 2,
            np.sin(alpha_rad) ** 4,
            4 * np.cos(alpha_rad) ** 2 * np.sin(alpha_rad) ** 2
        ])
        q_11s = np.inner(cs_vec, q_vec)
        r = np.sqrt((x - _x_0) ** 2 + (y - _y_0) ** 2)
        len_aff_inf = 30
        len_aff_0 = 10
        r_ref = 200
        len_aff = (1 - np.exp(-(r / r_ref))) * (len_aff_inf - len_aff_0) + len_aff_0
        kd = np.exp(-((y - _y_0) / len_aff) ** 2)

        q_11s_kd = kd * q_11s / Q_11
        return q_11s_kd

    _xx = np.arange(0, x_max)
    _yy = np.arange(0, y_max)
    xg, yg = np.meshgrid(_xx, _yy)

    q_rel = np.array([calculate_q11(x, y, x_0, y_0) for x, y in zip(xg.ravel(), yg.ravel())]).reshape(
        xg.shape)

    return q_rel


for i in tqdm(range(len(grippers_point_coordinates_xy[0]))):
    distrib = stiffness_distribution(grippers_point_coordinates_xy[0][i] + 150,
                                     grippers_point_coordinates_xy[1][i] + 230,
                                     grippers_directory_coordinates_xy[0][i],
                                     grippers_directory_coordinates_xy[1][i])
    abs_path_gs_img = os.path.join(encoding_data_path, f"encoding_{i}.png")
    abs_path_gs_matrix = os.path.join(encoding_data_path, f"encoding_{i}.npy")
    mpli.imsave(abs_path_gs_img, distrib, cmap="gray", vmin=0, vmax=1)
    np.save(abs_path_gs_matrix, distrib)


def create_strain_field(shape_name, dic_paths, data_type, gamma, image_name):
    """
    Create a strain field image using the provided data.
    """
    strain_field_data_path = os.path.join(data_path, shape_name, "strain_field", data_type)
    os.makedirs(strain_field_data_path, exist_ok=True)

    shear_path = dic_paths["el_coords_path"]
    el_coords_xyz = np.load(os.path.join(shear_path, 'el_coords_xyz.npy'))

    abs_path_out_image = os.path.join(strain_field_data_path, image_name + '.png')
    abs_path_out_matrix = os.path.join(strain_field_data_path, image_name + '.npy')

    x_min = -150.0
    x_max = 150.0
    num_pxl_x = 300

    y_min = -230.0
    y_max = 230.0
    num_pxl_y = 460

    h_min = 0
    h_max = 90.0

    x_grid = np.linspace(x_min, x_max, num_pxl_x)
    y_grid = np.linspace(y_min, y_max, num_pxl_y)
    pix_x, pix_y = np.meshgrid(x_grid, y_grid)

    nodes_xy = el_coords_xyz[:2, :].T
    pix_gs = spi.griddata(nodes_xy, gamma, (pix_x, pix_y), method='linear')

    # fill in NaN values
    x_ind, y_ind = np.indices(pix_gs.shape)
    missing = np.isnan(pix_gs)
    points = np.column_stack([x_ind[~missing], y_ind[~missing]])
    values = pix_gs[~missing]
    pix_gs[missing] = spi.griddata(points, values, (x_ind[missing], y_ind[missing]), method='nearest')

    mpli.imsave(abs_path_out_image, pix_gs, cmap='gray', vmin=h_min, vmax=h_max)
    np.save(abs_path_out_matrix, pix_gs)


# Material properties
E_1 = 140.0  # in N/mm² --> in direction of the springs --> x-direction here
E_2 = 10.0  # in N/mm² --> transverse direction of the springs
nu_12 = 0.3  # Poissons ratio
G_12 = 0.1  # Shear modulus

# Calculate stiffnesses before rotation
Q_11 = E_1 / (1 - nu_12 ** 2 * E_2 / E_1)
Q_12 = nu_12 * E_2 / (1 - nu_12 ** 2 * E_2 / E_1)
Q_22 = E_2 / (1 - nu_12 ** 2 * E_2 / E_1)
Q_33 = G_12

for shape_name, dic_paths in data_dics.items():
    for data_type in ['train', 'val']:
        if f'{data_type}_data' not in dic_paths.keys():
            continue

        simulated_forces = np.load(dic_paths[f"{data_type}_data"]["x_data_path"])
        simulated_angels = np.load(dic_paths[f"{data_type}_data"]["y_data_path"])
        u_springs_path = os.path.join(dic_paths["u_spring_data_path"], f"U_spring_{shape_name}.csv")
        u_spring_data = pd.read_csv(u_springs_path)

        n_grippers = len(grippers_point_coordinates_xy[0])
        n_angels = len(simulated_angels[0])

        columns = ['stamp_shape_image_path', 'stamp_shape_matrix_path'] + \
                  [f'gripper_x_{i}' for i in range(n_grippers)] + \
                  [f'gripper_y_{i}' for i in range(n_grippers)] + \
                  [f'gripper_dir_x_{i}' for i in range(n_grippers)] + \
                  [f'gripper_dir_y_{i}' for i in range(n_grippers)] + \
                  [f'gripper_force_{i}' for i in range(n_grippers)] + \
                  [f'gripper_length_{i}' for i in range(n_grippers)] + \
                  [f'gripper_encoding_image_path_{i}' for i in range(n_grippers)] + \
                  [f'gripper_encoding_matrix_path_{i}' for i in range(n_grippers)] + \
                  ['characteristic_e_1', 'characteristic_e_2', 'characteristic_nu_12', 'characteristic_g_12'] + \
                  ['stiffness_q_11', 'stiffness_q_12', 'stiffness_q_22', 'stiffness_q_33'] + \
                  [f'angle_{i}' for i in range(n_angels)] + \
                  ['strain_field_image_path', 'strain_field_matrix_path']

        data = pd.DataFrame(columns=columns)

        for idx, forces in tqdm(enumerate(simulated_forces), total=len(simulated_forces)):
            angles = simulated_angels[idx]
            image_name = f'strain_field_{idx}'
            create_strain_field(shape_name, dic_paths, data_type, angles, image_name)
            data.loc[idx] = [f'data/{shape_name}/shape/shape.png', f'data/{shape_name}/shape/shape.npy'] + \
                            list(grippers_point_coordinates_xy[0] + 150) + \
                            list(grippers_point_coordinates_xy[1] + 230) + \
                            list(grippers_directory_coordinates_xy[0]) + \
                            list(grippers_directory_coordinates_xy[1]) + \
                            list(forces) + \
                            u_spring_data['u'].tolist() + \
                            [f'data/encoding/encoding_{i}.png' for i in range(n_grippers)] + \
                            [f'data/encoding/encoding_{i}.npy' for i in range(n_grippers)] + \
                            [E_1, E_2, nu_12, G_12] + \
                            [Q_11, Q_12, Q_22, Q_33] + \
                            list(angles) + \
                            [f'data/{shape_name}/strain_field/{data_type}/{image_name}.png',
                             f'data/{shape_name}/strain_field/{data_type}/{image_name}.npy']

        data.to_csv(os.path.join(data_path, shape_name, f'{data_type}.csv'), index=False)
        data = data.filter(regex="stamp_shape_matrix_path|gripper_force|gripper_length|strain_field_matrix_path")
        data.to_csv(os.path.join(data_path, shape_name, f'{data_type}_short.csv'), index=False)
