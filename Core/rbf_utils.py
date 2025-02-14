"""
rbf_utils.py

This module implements a full 3D Radial Basis Function (RBF) interpolation method
with support for multiple drift options (including no drift, second order polynomial,
spherical, dome shaped, first order polynomial, 2D custom, and 3D custom).
It also provides helper functions for systematic sampling, KMeans-based sampling,
and evaluation of parameters via K-fold cross-validation.

Functions:
    - RBF_3D_kernel: Main interpolation function.
    - systematic_sampling: Systematically sample rows from a DataFrame.
    - kmeans_sampling_indices: Obtain sampling indices using KMeans.
    - evaluate_params_kfold: Evaluate a set of parameters using K-fold cross-validation.
"""

import numpy as npy
from autograd import numpy as np
from autograd import elementwise_grad as egrad
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
import itertools
import time
import traceback

def RBF_3D_kernel(G_position, G_orientation, layer_position, test_data,
                  eps, range_val, drift=0, cv=True, a=1, b=1, c=1, 
                  kernel_name="gaussian", show_drift=0,drift_scalar=None,
                  scaling_factor=1):
    """
    Perform 3D RBF interpolation using the provided positions, orientations, and surface points.
    
    Parameters:
      G_position (ndarray): (N,3) array of positions.
      G_orientation (ndarray): (N,3) array of orientation vectors.
      layer_position (ndarray): (M,3) array of surface points.
      test_data (ndarray): Test data points (used when cv is True).
      eps (float): Kernel parameter epsilon.
      range_val (float): Kernel parameter controlling the range.
      drift (int): Drift type indicator:
         0: No drift,
         1: Second Order Polynomial,
         2: Spherical,
         3: Dome Shaped,
         4: First Order Polynomial,
         5: 2D Custom,
         6: 3D Custom.
      cv (bool): If True, returns a 1D array (for cross-validation); otherwise returns a tuple (intp, mesh)
                 where intp is the interpolated field reshaped as (50,50,50) and mesh is a PyVista mesh.
      a, b, c (float): Additional drift parameters.
      kernel_name (str): Name of the kernel function.
    
    Returns:
      If cv is True:
         ndarray: Interpolated result as a 1D array.
      Else:
         tuple: (intp, mesh) where intp is the scalar field (50×50×50) and mesh is a PyVista mesh.
    """
    # Local copies of input arrays
    G_1 = G_position
    G_1_o = G_orientation
    layer1 = layer_position

    # Define a squared Euclidean distance function.
    def squared_euclidean_distance(x_1, x_2):
        sqd = np.sqrt(np.reshape(np.sum(x_1**2, axis=1), (x_1.shape[0], 1)) +
                      np.reshape(np.sum(x_2**2, axis=1), (1, x_2.shape[0])) -
                      2 * (x_1 @ x_2.T))
        return np.nan_to_num(sqd)

    # Define the kernel function.
    def kernel(radius, function, a=1, b=1, n=1):
        if function == 'cubic':
            return radius**3
        elif function == 'gaussian':
            return np.exp(- (radius**2) / (2*eps**2))
        elif function == 'quintic':
            return radius**5
        elif function == 'cubic_covariance':
            return (1 - 7*(radius/range_val)**2 +
                    35/4*(radius/range_val)**3 - 7/2*(radius/range_val)**5 +
                    3/4*(radius/range_val)**7)
        elif function == 'multiquadric':
            return np.sqrt(radius**2+eps**2)
        elif function == 'linear':
            return -radius
        elif function == 'thin_plate':
            return np.where(radius==0, 0, radius**2 * np.log(radius))
        elif function == 'inverse_quadratic':
            return 1 / (eps**2 + radius**2)
        elif function == 'inverse_multiquadric':
            return 1 / np.sqrt(radius**2+eps**2)
        elif function == 'gaussian_covariance':
            return np.exp(- (radius**2)/(range_val**2))
        elif function == 'ga_cub_cov':
            return a * (1 - np.exp(-(radius/range_val)**2)) + b * (1 - 7*(radius/range_val)**2 +
                                                                  35/4*(radius/range_val)**3 - 7/2*(radius/range_val)**5 +
                                                                  3/4*(radius/range_val)**7)
        elif function == 'ga_cub':
            return np.exp(- (radius**2)/(2*eps**2)) + n*(radius**3)
        else:
            raise ValueError("Unknown kernel function: " + function)

    # Compute gradients of the kernel using autograd.
    df = egrad(kernel)
    ddf = egrad(df)

    # Define a Cartesian distance function that repeats differences for 3 components.
    def cartesian_dist(x_1, x_2):
        return np.concatenate([
            np.tile(x_1[:,0] - np.reshape(x_2[:,0], (x_2[:,0].shape[0],1)), (1,3)),
            np.tile(x_1[:,1] - np.reshape(x_2[:,1], (x_2[:,1].shape[0],1)), (1,3)),
            np.tile(x_1[:,2] - np.reshape(x_2[:,2], (x_2[:,2].shape[0],1)), (1,3))
        ], axis=0)

    def cartesian_dist_no_tile(x_1, x_2):
        return np.concatenate([
            np.transpose(x_1[:,0] - np.reshape(x_2[:,0], (x_2.shape[0],1))),
            np.transpose(x_1[:,1] - np.reshape(x_2[:,1], (x_2.shape[0],1))),
            np.transpose(x_1[:,2] - np.reshape(x_2[:,2], (x_2.shape[0],1)))
        ], axis=0)

    def perpendicularity(G_1):
        a_mat = np.concatenate([np.ones((G_1.shape[0], G_1.shape[0])),
                                np.zeros((G_1.shape[0], G_1.shape[0])),
                                np.zeros((G_1.shape[0], G_1.shape[0]))], axis=1)
        b_mat = np.concatenate([np.zeros((G_1.shape[0], G_1.shape[0])),
                                np.ones((G_1.shape[0], G_1.shape[0])),
                                np.zeros((G_1.shape[0], G_1.shape[0]))], axis=1)
        c_mat = np.concatenate([np.zeros((G_1.shape[0], G_1.shape[0])),
                                np.zeros((G_1.shape[0], G_1.shape[0])),
                                np.ones((G_1.shape[0], G_1.shape[0]))], axis=1)
        return np.concatenate([a_mat, b_mat, c_mat], axis=0)

    def cov_gradients(dist_tiled):
        t1 = np.divide(h_u * h_v, dist_tiled**2, out=np.zeros_like(h_u*h_v), where=dist_tiled!=0)
        t2 = np.where(dist_tiled < range_val, np.nan_to_num(ddf(dist_tiled, function=kernel_name)) - np.nan_to_num(df(dist_tiled, function=kernel_name)/dist_tiled), 0)
        t3 = np.where(dist_tiled < range_val, np.nan_to_num(df(dist_tiled, function=kernel_name)/dist_tiled), 0)
        t3 = perpendicularity_matrix * t3
        t4 = 1/3 * np.eye(dist_tiled.shape[0])
        return t1 * t2 - t3 + t4

    def set_rest_ref_matrix(num_pts):
        ref_layer_points = np.repeat(np.stack([layer1[-1]], axis=0), repeats=num_pts-1, axis=0)
        rest_layer_points = np.concatenate([layer1[0:-1]], axis=0)
        return ref_layer_points, rest_layer_points

    def cov_interface(ref_layer_points, rest_layer_points):
        sed_rest_rest = squared_euclidean_distance(rest_layer_points, rest_layer_points)
        sed_ref_rest = squared_euclidean_distance(ref_layer_points, rest_layer_points)
        sed_rest_ref = squared_euclidean_distance(rest_layer_points, ref_layer_points)
        sed_ref_ref = squared_euclidean_distance(ref_layer_points, ref_layer_points)
        return kernel(sed_rest_rest, function=kernel_name) - kernel(sed_ref_rest, function=kernel_name) - \
               kernel(sed_rest_ref, function=kernel_name) + kernel(sed_ref_ref, function=kernel_name)

    def cov_interface_gradients(hu_rest, hu_ref):
        return hu_rest * np.where(sed_dips_rest < range_val, np.nan_to_num(df(sed_dips_rest, function=kernel_name)/sed_dips_rest), 0) - \
               hu_ref * np.where(sed_dips_ref < range_val, np.nan_to_num(df(sed_dips_ref, function=kernel_name)/sed_dips_ref), 0)

    def plot_3D(grid, value, surfaces_nr=10):
        """
        Generate and display a 3D mesh using a StructuredGrid.
        
        This function uses the global variables XX, YY, ZZ, layer1, layer_position, intp, 
        model_color, G_1, and G_1_o. It sets the jupyter backend to static, creates a 
        StructuredGrid from the meshgrid arrays, plots the mesh, adds points, contours, 
        bounding box, axes, arrows, and then shows the plot.
        
        Parameters:
        grid: (ignored; overwritten inside the function)
        value: the scalar field values (expected to be a 1D array that can be reshaped)
        surfaces_nr: number of surfaces (not used in this branch)
        
        Returns:
        The contours (mesh) extracted from the grid.
        """
        import pyvista as pv
        pv.set_jupyter_backend('static')
        grid = pv.StructuredGrid(XX, YY, ZZ)
        p = pv.Plotter(notebook=True, window_size=[1500, 1500])
        p.set_background('white')
        
        p.add_points(layer1[:, [0, 1, 2]], render_points_as_spheres=True, point_size=20.0, color='blue')
        
        grid.point_data['scalar'] = value.ravel(order='F')
        
        index = np.where(layer_position[:, 0] + layer_position[:, 1] == (layer_position[:, 0] + layer_position[:, 1]).min())
        distances = np.sqrt((XX - layer1[index, 0])**2 + (YY - layer1[index, 1])**2 + (ZZ - layer1[index, 2])**2)
        closest_point_indices = np.unravel_index(np.argmin(distances), distances.shape)
        lvl_t = np.array([intp[closest_point_indices]])
        print('surface scalar value:', lvl_t)
        contours_2 = grid.contour(lvl_t)
        p.add_mesh(contours_2, show_scalar_bar=0, label='surface', style='surface', opacity=0.8)
        
        p.add_bounding_box()
        
        p.show_bounds(
            grid='front',
            location='outer',
            xlabel='X Axis',
            ylabel='Y Axis',
            zlabel='Z Axis',
            color='black',
            font_size=30
        )
        p.set_scale(1, 1, 1.5)
        p.add_arrows(G_1[:, [0, 1, 2]], direction=G_1_o[:, [0, 1, 2]], color='black', mag=50)
        p.add_axes()
        p.screenshot()
        p.show()
        return contours_2

    def plot_3D_drift(grid, value, surfaces_nr=10):
        """
        Generate and display a 3D mesh for drift contribution using a StructuredGrid.
        
        This function uses the global variables XX, YY, ZZ, layer1, G_1, and G_1_o.
        It sets the jupyter backend to static, creates a StructuredGrid from the meshgrid 
        arrays, and then extracts a set of contours from a range of scalar field values.
        Finally, it plots the points, contours, arrows, and axes.
        
        Parameters:
        grid: (ignored; overwritten inside the function)
        value: the scalar field values (expected to be a 1D array that can be reshaped)
        surfaces_nr: number of contour surfaces to extract
        
        Returns:
        The contours (mesh) extracted from the grid.
        """
        import pyvista as pv
        pv.set_jupyter_backend('static')
        grid = pv.StructuredGrid(XX, YY, ZZ)
        
        p = pv.Plotter(notebook=True, window_size=[400, 400])
        p.set_background('white')
        
        p.add_points(layer1[:, [0, 1, 2]], render_points_as_spheres=True, point_size=14.0, color='blue')
        
        grid.point_data['scalar'] = value.ravel(order='F')
        contours_1 = grid.contour(np.linspace(value.min(), value.max(), surfaces_nr))
        p.add_mesh(contours_1, show_scalar_bar=0, label='scalar_field', style='surface', opacity=0.8, cmap='viridis')
        p.add_arrows(G_1[:, [0, 1, 2]], direction=G_1_o[:, [0, 1, 2]], color='black', mag=50)
        p.add_axes()
        p.show()
        return contours_1

    # Begin main computations.
    G_1_tiled = np.tile(G_1, (3, 1))
    h_u = cartesian_dist(G_1, G_1)
    h_v = h_u.T
    perpendicularity_matrix = perpendicularity(G_1)
    dist_tiled = squared_euclidean_distance(G_1_tiled, G_1_tiled) + np.eye(G_1_tiled.shape[0])
    C_G = cov_gradients(dist_tiled)
    num_pts = np.array([layer1.shape[0]])
    ref_layer_points, rest_layer_points = set_rest_ref_matrix(num_pts)
    C_I = cov_interface(ref_layer_points, rest_layer_points)

    # Create simulation grid based on layer1 extents.
    # xx = np.linspace(layer1[:,0].min()-100, layer1[:,0].max()+100, 50)
    # yy = np.linspace(layer1[:,1].min()-100, layer1[:,1].max()+100, 50)
    # zz = np.linspace(layer1[:,2].min()-100, layer1[:,2].max()+100, 50)
    # XX, YY, ZZ = np.meshgrid(xx, yy, zz)
    # X = np.reshape(XX, (-1)).T
    # Y = np.reshape(YY, (-1)).T
    # Z = np.reshape(ZZ, (-1)).T
    # if cv:
    #     grid = test_data
    # else:
    #     grid = np.stack([X, Y, Z], axis=1)


    sed_dips_rest = squared_euclidean_distance(G_1_tiled, rest_layer_points)
    sed_dips_ref = squared_euclidean_distance(G_1_tiled, ref_layer_points)
    hu_rest = cartesian_dist_no_tile(G_1, rest_layer_points)
    hu_ref = cartesian_dist_no_tile(G_1, ref_layer_points)
    C_GI = cov_interface_gradients(hu_rest, hu_ref)
    C_IG = C_GI.T

    x0 = (layer1[:,0].min() + layer1[:,0].max())/2
    y0 = (layer1[:,1].min() + layer1[:,1].max())/2
    z0 = (layer1[:,2].min() + layer1[:,2].max())/2

    # Recreate grid for simulation.
    xx = np.linspace(layer1[:,0].min()-100, layer1[:,0].max()+100, 50)
    yy = np.linspace(layer1[:,1].min()-100, layer1[:,1].max()+100, 50)
    zz = np.linspace(layer1[:,2].min()-100, layer1[:,2].max()+100, 50)
    XX, YY, ZZ = np.meshgrid(xx, yy, zz)
    X = np.reshape(XX, (-1)).T
    Y = np.reshape(YY, (-1)).T
    Z = np.reshape(ZZ, (-1)).T
    if cv:
        grid = test_data
    else:
        grid = np.stack([X, Y, Z], axis=1)

    hu_Simpoints = cartesian_dist_no_tile(G_1, grid)
    sed_dips_SimPoint = squared_euclidean_distance(G_1_tiled, grid)
    sed_rest_SimPoint = squared_euclidean_distance(rest_layer_points, grid)
    sed_ref_SimPoint = squared_euclidean_distance(ref_layer_points, grid)

    # Branch on drift type.
    if drift == 1:
        # Second Order Polynomial drift
        n = G_1.shape[0]
        sub_x = np.tile(np.array([[1,0,0]]), (n,1))
        sub_y = np.tile(np.array([[0,1,0]]), (n,1))
        sub_z = np.tile(np.array([[0,0,1]]), (n,1))
        sub_block1 = np.concatenate([sub_x, sub_y, sub_z], axis=0)
        sub_x_2 = np.zeros((n,3))
        sub_y_2 = np.zeros((n,3))
        sub_z_2 = np.zeros((n,3))
        sub_x_2[:,0] = 2 * G_1[:,0]
        sub_y_2[:,1] = 2 * G_1[:,1]
        sub_z_2[:,2] = 2 * G_1[:,2]
        sub_block2 = np.concatenate([sub_x_2, sub_y_2, sub_z_2], axis=0)
        sub_xy = np.reshape(np.concatenate([G_1[:,1], G_1[:,0]], axis=0), (2*n, 1))
        sub_xy = np.pad(sub_xy, ((0,n),(0,0)))
        sub_xz = np.concatenate([np.pad(np.reshape(G_1[:,2], (n,1)), ((0,n),(0,0))), np.reshape(G_1[:,0], (n,1))], axis=0)
        sub_yz = np.reshape(np.concatenate([G_1[:,2], G_1[:,1]], axis=0), (2*n,1))
        sub_yz = np.pad(sub_yz, ((n,0),(0,0)))
        sub_block3 = np.concatenate([sub_xy, sub_xz, sub_yz], axis=1)
        U_G = np.concatenate([sub_block1, sub_block2, sub_block3], axis=1)
        U_I = -np.stack([
            rest_layer_points[:,0]-ref_layer_points[:,0],
            rest_layer_points[:,1]-ref_layer_points[:,1],
            rest_layer_points[:,2]-ref_layer_points[:,2],
            rest_layer_points[:,0]**2-ref_layer_points[:,0]**2,
            rest_layer_points[:,1]**2-ref_layer_points[:,1]**2,
            rest_layer_points[:,2]**2-ref_layer_points[:,2]**2,
            rest_layer_points[:,0]*rest_layer_points[:,1]-ref_layer_points[:,0]*ref_layer_points[:,1],
            rest_layer_points[:,0]*rest_layer_points[:,2]-ref_layer_points[:,0]*ref_layer_points[:,2],
            rest_layer_points[:,1]*rest_layer_points[:,2]-ref_layer_points[:,1]*ref_layer_points[:,2]
        ], axis=1)
        length_of_CG = C_G.shape[1]
        length_of_CGI = C_GI.shape[1]
        U_G = U_G[:length_of_CG, :9]
        U_I = U_I[:length_of_CGI, :9]
        U = np.concatenate([U_G, U_I], axis=0)
        zero_matrix = np.zeros((U.shape[1], U.shape[1]))
        K_full = np.concatenate([np.concatenate([C_G, C_GI], axis=1),
                                 np.concatenate([C_IG, C_I], axis=1)], axis=0)
        K_D = np.concatenate([np.concatenate([K_full, U], axis=1),
                              np.concatenate([U.T, zero_matrix], axis=1)], axis=0)
        bk = np.concatenate([G_1_o[:,0], G_1_o[:,1], G_1_o[:,2],
                             np.zeros(K_D.shape[0]-G_1.shape[0]*3)], axis=0)
        bk = np.reshape(bk, (-1,1))
        w = np.linalg.lstsq(K_D, bk, rcond=None)[0]
        sigma_0_grad = w[:G_1.shape[0]*3] * (-hu_Simpoints * (sed_dips_SimPoint < range_val) *
                                              (np.nan_to_num(df(sed_dips_SimPoint, function=kernel_name)/sed_dips_SimPoint))
                                             )
        sigma_0_grad = np.sum(sigma_0_grad, axis=0)
        sigma_0_interf = -w[G_1.shape[0]*3:-9] * (((sed_rest_SimPoint < range_val) * (kernel(sed_rest_SimPoint, function=kernel_name)) -
                                                    (sed_ref_SimPoint < range_val) * (kernel(sed_ref_SimPoint, function=kernel_name))))
        sigma_0_interf = np.sum(sigma_0_interf, axis=0)
        sigma_0_2nd_drift_1 = (grid[:,0]*w[-9] + grid[:,1]*w[-8] + grid[:,2]*w[-7] +
                               grid[:,0]**2*w[-6] + grid[:,1]**2*w[-5] + grid[:,2]**2*w[-4] +
                               grid[:,0]*grid[:,1]*w[-3] + grid[:,0]*grid[:,2]*w[-2] + grid[:,1]*grid[:,2]*w[-1])
        sigma_0_2nd_drift = sigma_0_2nd_drift_1
        drift_contribution = sigma_0_2nd_drift
        interpolate_result = sigma_0_grad + sigma_0_interf + sigma_0_2nd_drift
    elif drift == 0:
        # No drift branch
        K_full = np.concatenate([np.concatenate([C_G, C_GI], axis=1),
                                 np.concatenate([C_IG, C_I], axis=1)], axis=0)
        K_D = np.nan_to_num(K_full)
        bk = np.concatenate([G_1_o[:,0], G_1_o[:,1], G_1_o[:,2],
                             np.zeros(K_D.shape[0]-G_1.shape[0]*3)], axis=0)
        bk = np.reshape(bk, (-1,1))
        w = np.linalg.lstsq(K_D, bk, rcond=None)[0]
        sigma_0_grad = w[:G_1.shape[0]*3] * (-hu_Simpoints * (sed_dips_SimPoint < range_val) *
                                              (np.nan_to_num(df(sed_dips_SimPoint, function=kernel_name)/sed_dips_SimPoint)))
        sigma_0_grad = np.sum(sigma_0_grad, axis=0)
        sigma_0_interf = -w[G_1.shape[0]*3:] * (((sed_rest_SimPoint < range_val) * (kernel(sed_rest_SimPoint, function=kernel_name)) -
                                                   (sed_ref_SimPoint < range_val) * (kernel(sed_ref_SimPoint, function=kernel_name))))
        sigma_0_interf = np.sum(sigma_0_interf, axis=0)
        interpolate_result = sigma_0_grad + sigma_0_interf
    elif drift == 2:
        # Spherical drift branch
        r_val = 1
        K_full = np.concatenate([np.concatenate([C_G, C_GI], axis=1),
                                 np.concatenate([C_IG, C_I], axis=1)], axis=0)
        D_Z = ((2*(G_1[:,2]-z0)/c)**2).reshape(G_1.shape[0],1)
        D_X = ((2*(G_1[:,1]-x0)/a)**2).reshape(G_1.shape[0],1)
        D_Y = ((2*(G_1[:,0]-y0)/b)**2).reshape(G_1.shape[0],1)
        D_I = ((((ref_layer_points[:,0]-x0)/a)**2 + ((ref_layer_points[:,1]-y0)/b)**2 + ((ref_layer_points[:,2]-z0)/c)**2 - r_val**2) -
               (((rest_layer_points[:,0]-x0)/a)**2 + ((rest_layer_points[:,1]-y0)/b)**2 + ((rest_layer_points[:,2]-z0)/c)**2 - r_val**2)).reshape(-1,1)
        D_E = np.concatenate([D_X, D_Y, D_Z, D_I], axis=0)
        D_T = D_E.T
        zero_matrix = np.zeros((D_E.shape[1], D_E.shape[1]))
        K_D = np.concatenate([np.concatenate([K_full, D_E], axis=1),
                              np.concatenate([D_T, zero_matrix], axis=1)], axis=0)
        bk = np.concatenate([G_1_o[:,0], G_1_o[:,1], G_1_o[:,2],
                             np.zeros(K_D.shape[0]-G_1.shape[0]*3)], axis=0)
        bk = np.reshape(bk, (-1,1))
        w = np.linalg.lstsq(K_D, bk, rcond=None)[0]
        sigma_0_grad = w[:G_1.shape[0]*3] * (-hu_Simpoints * (sed_dips_SimPoint < range_val) *
                                              (np.nan_to_num(df(sed_dips_SimPoint, function=kernel_name)/sed_dips_SimPoint)))
        sigma_0_grad = np.sum(sigma_0_grad, axis=0)
        sigma_0_interf = -w[G_1.shape[0]*3:-1] * (((sed_rest_SimPoint < range_val) * (kernel(sed_rest_SimPoint, function=kernel_name)) -
                                                   (sed_ref_SimPoint < range_val) * (kernel(sed_ref_SimPoint, function=kernel_name))))
        sigma_0_interf = np.sum(sigma_0_interf, axis=0)
        external_drift = (((grid[:,0]-x0)/a)**2 + ((grid[:,1]-y0)/b)**2 + ((grid[:,2]-z0)/c)**2 + r_val**2) * (w[-1]).T
        drift_contribution = external_drift
        interpolate_result = sigma_0_grad + sigma_0_interf + external_drift
    elif drift == 3:
        # Dome Shaped drift branch
        K_full = np.concatenate([np.concatenate([C_G, C_GI], axis=1),
                                 np.concatenate([C_IG, C_I], axis=1)], axis=0)
        def F_x(x, y, z, a, b, c):
            return z - (c - ((x-x0)**2+(y-y0)**2)/a**2)*np.exp(-((x-x0)**2+(y-y0)**2)/b**2)
        def partial_F_x(x, y, a, b, c):
            term1 = (2*(x-x0)*(c - ((x-x0)**2+(y-y0)**2)/a**2))/b**2
            term2 = (2*(x-x0))/a**2
            return (term1+term2)*np.exp(-((x-x0)**2+(y-y0)**2)/b**2)
        def partial_F_y(y, x, a, b, c):
            term1 = (2*(y-y0)*(c - ((x-x0)**2+(y-y0)**2)/a**2))/b**2
            term2 = (2*(y-y0))/a**2
            return (term1+term2)*np.exp(-((x-x0)**2+(y-y0)**2)/b**2)
        def partial_F_z():
            return np.ones_like(x)
        x = G_1[:,0]
        y = G_1[:,1]
        z = G_1[:,2]
        D_Z = partial_F_z().reshape(G_1.shape[0],1)
        D_X = partial_F_x(x, y, a, b, c).reshape(G_1.shape[0],1)
        D_Y = partial_F_y(y, x, a, b, c).reshape(G_1.shape[0],1)
        D_I = (F_x(ref_layer_points[:,0], ref_layer_points[:,1], ref_layer_points[:,2], a, b, c) -
               F_x(rest_layer_points[:,0], rest_layer_points[:,1], rest_layer_points[:,2], a, b, c)).reshape(-1,1)
        D_E = np.concatenate([D_X, D_Y, D_Z, D_I], axis=0)
        D_T = D_E.T
        zero_matrix = np.zeros((D_E.shape[1], D_E.shape[1]))
        K_D = np.concatenate([np.concatenate([K_full, D_E], axis=1),
                              np.concatenate([D_T, zero_matrix], axis=1)], axis=0)
        bk = np.concatenate([G_1_o[:,0], G_1_o[:,1], G_1_o[:,2],
                             np.zeros(K_D.shape[0]-G_1.shape[0]*3)], axis=0)
        bk = np.reshape(bk, (-1,1))
        w = np.linalg.lstsq(K_D, bk, rcond=None)[0]
        sigma_0_grad = w[:G_1.shape[0]*3] * (-hu_Simpoints * (sed_dips_SimPoint < range_val) *
                                              (np.nan_to_num(df(sed_dips_SimPoint, function=kernel_name)/sed_dips_SimPoint)))
        sigma_0_grad = np.sum(sigma_0_grad, axis=0)
        sigma_0_interf = -w[G_1.shape[0]*3:-1] * (((sed_rest_SimPoint < range_val) * (kernel(sed_rest_SimPoint, function=kernel_name)) -
                                                    (sed_ref_SimPoint < range_val) * (kernel(sed_ref_SimPoint, function=kernel_name))))
        sigma_0_interf = np.sum(sigma_0_interf, axis=0)
        external_drift = F_x(grid[:,0], grid[:,1], grid[:,2], a, b, c) * (w[-1]).T
        drift_contribution = external_drift
        interpolate_result = sigma_0_grad + sigma_0_interf + external_drift
    elif drift == 4:
        # First Order Polynomial drift branch
        K_full = np.concatenate([np.concatenate([C_G, C_GI], axis=1),
                                 np.concatenate([C_IG, C_I], axis=1)], axis=0)
        n = G_1.shape[0]
        sub_x = np.tile(np.array([[1,0,0]]), (n,1))
        sub_y = np.tile(np.array([[0,1,0]]), (n,1))
        sub_z = np.tile(np.array([[0,0,1]]), (n,1))
        sub_block1 = np.concatenate([sub_x, sub_y, sub_z], axis=0)
        sub_x_2 = np.zeros((n,3))
        sub_y_2 = np.zeros((n,3))
        sub_z_2 = np.zeros((n,3))
        sub_x_2[:,0] = 2 * G_1[:,0]
        sub_y_2[:,1] = 2 * G_1[:,1]
        sub_z_2[:,2] = 2 * G_1[:,2]
        sub_block2 = np.concatenate([sub_x_2, sub_y_2, sub_z_2], axis=0)
        sub_xy = np.reshape(np.concatenate([G_1[:,1], G_1[:,0]], axis=0), (2*n, 1))
        sub_xy = np.pad(sub_xy, ((0,n),(0,0)))
        sub_xz = np.concatenate([np.pad(np.reshape(G_1[:,2], (n,1)), ((0,n),(0,0))), np.reshape(G_1[:,0], (n,1))], axis=0)
        sub_yz = np.reshape(np.concatenate([G_1[:,2], G_1[:,1]], axis=0), (2*n,1))
        sub_yz = np.pad(sub_yz, ((n,0),(0,0)))
        sub_block3 = np.concatenate([sub_xy, sub_xz, sub_yz], axis=1)
        U_G = np.concatenate([sub_block1, sub_block2, sub_block3], axis=1)
        U_I = -np.stack([
            rest_layer_points[:,0]-ref_layer_points[:,0],
            rest_layer_points[:,1]-ref_layer_points[:,1],
            rest_layer_points[:,2]-ref_layer_points[:,2],
            rest_layer_points[:,0]**2-ref_layer_points[:,0]**2,
            rest_layer_points[:,1]**2-ref_layer_points[:,1]**2,
            rest_layer_points[:,2]**2-ref_layer_points[:,2]**2,
            rest_layer_points[:,0]*rest_layer_points[:,1]-ref_layer_points[:,0]*ref_layer_points[:,1],
            rest_layer_points[:,0]*rest_layer_points[:,2]-ref_layer_points[:,0]*ref_layer_points[:,2],
            rest_layer_points[:,1]*rest_layer_points[:,2]-ref_layer_points[:,1]*ref_layer_points[:,2]
        ], axis=1)
        length_of_CG = C_G.shape[1]
        length_of_CGI = C_GI.shape[1]
        U_G = U_G[:length_of_CG, :3]
        U_I = U_I[:length_of_CGI, :3]
        U = np.concatenate([U_G, U_I], axis=0)
        zero_matrix = np.zeros((U.shape[1], U.shape[1]))
        K_D = np.concatenate([np.concatenate([K_full, U], axis=1),
                              np.concatenate([U.T, zero_matrix], axis=1)], axis=0)
        bk = np.concatenate([G_1_o[:,0], G_1_o[:,1], G_1_o[:,2],
                             np.zeros(K_D.shape[0]-G_1.shape[0]*3)], axis=0)
        bk = np.reshape(bk, (-1,1))
        w = np.linalg.lstsq(K_D, bk, rcond=None)[0]
        sigma_0_grad = w[:G_1.shape[0]*3] * (-hu_Simpoints * (sed_dips_SimPoint < range_val) *
                                              (np.nan_to_num(df(sed_dips_SimPoint, function=kernel_name)/sed_dips_SimPoint)))
        sigma_0_grad = np.sum(sigma_0_grad, axis=0)
        sigma_0_interf = -w[G_1.shape[0]*3:-3] * (((sed_rest_SimPoint < range_val) * (kernel(sed_rest_SimPoint, function=kernel_name)) -
                                                   (sed_ref_SimPoint < range_val) * (kernel(sed_ref_SimPoint, function=kernel_name))))
        sigma_0_interf = np.sum(sigma_0_interf, axis=0)
        sigma_0_2nd_drift_1 = np.sum(grid*(w[-3:]).T, axis=1)
        sigma_0_2nd_drift = sigma_0_2nd_drift_1
        drift_contribution = sigma_0_2nd_drift
        interpolate_result = sigma_0_grad + sigma_0_interf + sigma_0_2nd_drift
    elif drift == 5:
        # 2D Custom drift branch
        from scipy.optimize import minimize
        # Call a gravity forward 2D function (assumed to be available via self)
        gravity, aa, bb, cc, dd, ee, ff = self.gravity_fwd_2d()
        def F_x(x, y, aa, bb, cc, dd, ee, ff):
            return aa*x**2 + bb*y**2 + cc*x*y + dd*x + ee*y + ff
        def misfit(params, easting, northing, gravity_observed, alpha=1e-6):
            gravity_calculated = F_x(easting, northing, *params)
            residuals = gravity_observed - gravity_calculated
            regularization = alpha * np.sum(params**2)
            return np.sum(residuals**2) + regularization
        initial_params = np.zeros(6)
        result = minimize(misfit, initial_params, args=(gravity[1], gravity[2], gravity[0]),
                          method='L-BFGS-B', options={'maxiter':10000, 'ftol':1e-12})
        a_opt, b_opt, c_opt, d_opt, e_opt, f_opt = result.x
        gravity_calculated = F_x(gravity[1], gravity[2], *result.x)
        K_full = np.concatenate([np.concatenate([C_G, C_GI], axis=1),
                                 np.concatenate([C_IG, C_I], axis=1)], axis=0)
        w = np.linalg.lstsq(K_full, np.zeros((K_full.shape[0],1)), rcond=None)[0]
        sigma_0_grad = np.sum(w[:G_1.shape[0]*3] * (-hu_Simpoints * (sed_dips_SimPoint < range_val) *
                                                     (np.nan_to_num(df(sed_dips_SimPoint, function=kernel_name)/sed_dips_SimPoint))), axis=0)
        sigma_0_interf = np.sum(-w[G_1.shape[0]*3:]*(((sed_rest_SimPoint < range_val) * (kernel(sed_rest_SimPoint, function=kernel_name)) -
                                                      (sed_ref_SimPoint < range_val) * (kernel(sed_ref_SimPoint, function=kernel_name)))), axis=0)
        sigma_0_2nd_drift = 0
        interpolate_result = sigma_0_grad + sigma_0_interf + sigma_0_2nd_drift
    elif drift == 6:

        def numerical_gradient(F, x, eps=1e-5):
            # F is a scalar function from R^n -> R
            # x is a point in R^n
            grad = np.zeros_like(x, dtype=float)
            for i in range(len(x)):
                x_plus = x.copy();   x_plus[i] += eps
                x_minus = x.copy();  x_minus[i] -= eps
                grad[i] = (F(x_plus) - F(x_minus)) / (2*eps)
            return grad
        # 3D Custom drift branch
        K_full = np.concatenate([np.concatenate([C_G, C_GI], axis=1),
                                 np.concatenate([C_IG, C_I], axis=1)], axis=0)
        scalar_field = drift_scalar 
        scalar_field *= float(scaling_factor)
        scalar_field = scalar_field.transpose(2,1,0)
        from scipy.interpolate import RegularGridInterpolator
        interpolator = RegularGridInterpolator((xx,yy,zz), scalar_field, method='linear', bounds_error=False, fill_value=None)
        def F(point):
            return interpolator(point.reshape(1, -1))[0]
        def F_x(points):
            return interpolator(points)
        gradient_F = egrad(F)
        points_custom = G_1  
        gradients = np.array([numerical_gradient(F, p) for p in points_custom])
        D_X = gradients[:,0].reshape(-1,1)
        D_Y = gradients[:,1].reshape(-1,1)
        D_Z = gradients[:,2].reshape(-1,1)
        D_I = (F_x(np.array(ref_layer_points)) - F_x(np.array(rest_layer_points))).reshape(-1,1)
        D_E = np.concatenate([D_X, D_Y, D_Z, D_I], axis=0)
        D_T = D_E.T
        zero_matrix = np.zeros((D_E.shape[1], D_E.shape[1]))
        K_D = np.concatenate([np.concatenate([K_full, D_E], axis=1), np.concatenate([D_T, zero_matrix], axis=1)], axis=0)
        bk = np.concatenate([G_1_o[:,0], G_1_o[:,1], G_1_o[:,2],
                             np.zeros(K_D.shape[0]-G_1.shape[0]*3)], axis=0)
        bk = np.reshape(bk, (-1,1))
        w = np.linalg.lstsq(K_D, bk, rcond=None)[0]
        sigma_0_grad = np.sum(w[:G_1.shape[0]*3] * (-hu_Simpoints * (sed_dips_SimPoint < range_val) *
                                                     (np.nan_to_num(df(sed_dips_SimPoint, function=kernel_name)/sed_dips_SimPoint))), axis=0)
        sigma_0_interf = np.sum(-w[G_1.shape[0]*3:-1]*(((sed_rest_SimPoint < range_val) * (kernel(sed_rest_SimPoint, function=kernel_name)) -
                                                        (sed_ref_SimPoint < range_val) * (kernel(sed_ref_SimPoint, function=kernel_name)))), axis=0)
        external_drift = F_x(np.array(grid))*(w[-1]).T
        drift_contribution = external_drift
        interpolate_result = sigma_0_grad + sigma_0_interf + external_drift

    if cv:
        return interpolate_result
    else:
        if show_drift == 0:
            intp = np.reshape(interpolate_result, [50, 50, 50])
            mesh = plot_3D(grid, intp, surfaces_nr=2)
            return intp, mesh
        elif show_drift == 1:
            drift_contribution = np.reshape(drift_contribution, [50, 50, 50])                    
            mesh_drift = plot_3D_drift(grid, drift_contribution, surfaces_nr=10)                    
            return drift_contribution, mesh_drift

def systematic_sampling(df, n):
    """
    Sample rows from a DataFrame in a systematic (evenly spaced) fashion.
    
    Parameters:
        df (pandas.DataFrame): The input data frame.
        n (int): The desired number of samples.
    
    Returns:
        pandas.DataFrame: The sampled data.
    """
    count = len(df)
    step = max(1, count // n)
    indices = list(range(0, count, step))[:n]
    return df.iloc[indices]

def kmeans_sampling_indices(df, n_samples, random_state=None):
    """
    Obtain sample indices using KMeans clustering.
    
    Parameters:
        df (pandas.DataFrame): Data frame with data points.
        n_samples (int): Number of samples/clusters.
        random_state (int, optional): Random state for reproducibility.
    
    Returns:
        list: Indices corresponding to the closest point in each cluster.
    """
    kmeans = KMeans(n_clusters=n_samples, random_state=random_state)
    kmeans.fit(df)
    sampled_indices = []
    for i in range(n_samples):
        cluster_indices = (kmeans.labels_ == i)
        cluster_points = df[cluster_indices]
        centroid = kmeans.cluster_centers_[i]
        closest_index = cluster_points.apply(lambda row: ((row - centroid)**2).sum(), axis=1).idxmin()
        sampled_indices.append(closest_index)
    return sampled_indices

def evaluate_params_kfold(a, b, c, eps, range_val, G_position, G_orientation, layer_position, drift_type, n_splits=5, kernel_name="gaussian",drift_scalar=None,scaling_factor=1.0):
    """
    Evaluate RBF interpolation parameters using K-fold cross-validation.
    
    Parameters:
        a, b, c (float): Drift parameters.
        eps (float): Kernel epsilon parameter.
        range_val (float): Kernel range parameter.
        G_position (ndarray): RBF positions.
        G_orientation (ndarray): RBF orientations.
        layer_position (ndarray): Surface points.
        drift_type (int): Drift type indicator.
        n_splits (int): Number of K-fold splits.
        kernel_name (str): Name of the kernel function.
    
    Returns:
        tuple: (average error, (a, b, c, eps, range_val))
    """
    try:
        kf = KFold(n_splits=n_splits, shuffle=False)
        errors = []
        for train_index, val_index in kf.split(layer_position):
            train_layer = layer_position[train_index]
            val_layer = layer_position[val_index]
            real_data = train_layer[[0]]
            intp = RBF_3D_kernel(G_position, G_orientation, train_layer, test_data=val_layer,
                                 eps=eps, range_val=range_val, drift=drift_type, cv=True, a=a, b=b, c=c, 
                                 kernel_name=kernel_name,drift_scalar=drift_scalar,scaling_factor=scaling_factor)
            actual = RBF_3D_kernel(G_position, G_orientation, train_layer, test_data=real_data,
                                   eps=eps, range_val=range_val, drift=drift_type, cv=True, a=a, b=b, c=c, 
                                   kernel_name=kernel_name,drift_scalar=drift_scalar,scaling_factor=scaling_factor)
            error = np.mean(np.abs(intp - actual) / np.abs(actual))
            errors.append(error)
        avg_error = np.mean(errors)
        return (avg_error, (a, b, c, eps, range_val))
    except Exception as e:
        print(f"Error with parameters a={a}, b={b}, c={c}: {e}")
        traceback.print_exc()
        return (np.inf, (a, b, c, eps, range_val))

