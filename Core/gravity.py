"""
gravity.py

This module contains functions to compute gravity anomalies due to point masses,
generate coordinate grids, and perform forward gravity modeling.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator

def my_point_gravity(coordinates, points, masses, field="g_z"):
    """
    Compute gravitational acceleration at given observation points due to point masses.
    
    Parameters:
        coordinates (tuple): Tuple of arrays (easting, northing, upward), each of shape (N,).
        points (tuple): Tuple of arrays (x_mass, y_mass, z_mass), each of shape (M,).
        masses (array): Masses for each point (M,).
        field (str): Which component to compute ('g_z', 'g_x', or 'g_y').
    
    Returns:
        numpy.ndarray: Gravitational acceleration at each observation point.
    """
    G = 6.67430e-11  # Gravitational constant
    x_obs, y_obs, z_obs = map(np.asarray, coordinates)
    x_mass, y_mass, z_mass = map(np.asarray, points)
    masses = np.asarray(masses)
    dx = x_mass[np.newaxis, :] - x_obs[:, np.newaxis]
    dy = y_mass[np.newaxis, :] - y_obs[:, np.newaxis]
    dz = z_mass[np.newaxis, :] - z_obs[:, np.newaxis]
    r_squared = dx**2 + dy**2 + dz**2
    r = np.sqrt(r_squared)
    r[r < 1e-10] = 1e-10  # Prevent division by zero
    if field == 'g_z':
        g_contributions = G * masses / r**3 * dz
    elif field == 'g_x':
        g_contributions = G * masses / r**3 * dx
    elif field == 'g_y':
        g_contributions = G * masses / r**3 * dy
    else:
        raise ValueError(f"Unknown field component '{field}'")
    gravity = np.sum(g_contributions, axis=1)
    return gravity

def my_grid_coordinates(region, spacing):
    """
    Generate coordinate grids over a specified region with a given spacing.
    
    Parameters:
        region (tuple): (xmin, xmax, ymin, ymax)
        spacing (float): Spacing between grid points.
    
    Returns:
        tuple: Meshgrid arrays (easting, northing).
    """
    xmin, xmax, ymin, ymax = region
    easting = np.arange(xmin, xmax + spacing, spacing)
    northing = np.arange(ymin, ymax + spacing, spacing)
    easting, northing = np.meshgrid(easting, northing)
    return easting, northing

def gravity_sim(mesh, spacing=100, density=800):
    """
    Simulate the gravity disturbance from a VTK mesh by treating its points as point masses.
    
    Parameters:
        mesh: The input VTK mesh (with a .points attribute).
        spacing (float): Spacing for the observation grid.
        density (float): Density used for each point (kg/m^3).
    
    Returns:
        tuple: (gravity, easting, northing) where gravity is a 1D array of computed values.
    """
    points = mesh.points
    xmin, xmax = np.min(points[:, 0]), np.max(points[:, 0])
    ymin, ymax = np.min(points[:, 1]), np.max(points[:, 1])
    masses = np.full(points.shape[0], density)
    easting, northing = my_grid_coordinates((xmin, xmax, ymin, ymax), spacing)
    upward = np.zeros_like(easting)
    easting, northing, upward = easting.ravel(), northing.ravel(), upward.ravel()
    gravity = my_point_gravity(
        coordinates=(easting, northing, upward),
        points=(points[:, 0], points[:, 1], points[:, 2]),
        masses=masses,
        field="g_z"
    )
    return gravity, easting, northing

def gravity_fwd_2d(mesh, canvas_frame):
    """
    Compute a 2D gravity forward model based on the input mesh and display it.
    
    Parameters:
        mesh: The VTK mesh.
        canvas_frame (tk.Frame): The Tkinter frame in which to embed the plot.
    
    Returns:
        tuple: (gravity_calculated, optimized parameters a, b, c, d, e, f)
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    gravity, easting, northing = gravity_sim(mesh)
    # Normalize coordinates and gravity
    easting_mean, easting_std = easting.mean(), easting.std()
    northing_mean, northing_std = northing.mean(), northing.std()
    gravity_mean, gravity_std = gravity.mean(), gravity.std()
    easting_norm = (easting - easting_mean) / easting_std
    northing_norm = (northing - northing_mean) / northing_std
    gravity_norm = (gravity - gravity_mean) / gravity_std

    def polynomial_gravity_anomaly(easting, northing, params):
        a, b, c, d, e, f = params
        return a * easting**2 + b * northing**2 + c * easting * northing + d * easting + e * northing + f

    def misfit(params, easting, northing, gravity_observed, alpha=1e-6):
        gravity_calculated = polynomial_gravity_anomaly(easting, northing, params)
        residuals = gravity_observed - gravity_calculated
        regularization = alpha * np.sum(params**2)
        return np.sum(residuals**2) + regularization

    initial_params = np.zeros(6)
    result = minimize(
        misfit,
        initial_params,
        args=(easting_norm, northing_norm, gravity_norm),
        method='L-BFGS-B',
        options={'maxiter': 10000, 'ftol': 1e-12}
    )
    a_opt, b_opt, c_opt, d_opt, e_opt, f_opt = result.x
    gravity_calculated_norm = polynomial_gravity_anomaly(easting_norm, northing_norm, result.x)
    gravity_calculated = gravity_calculated_norm * gravity_std + gravity_mean

    fig = plt.figure(figsize=(14,6))
    plt.subplot(1, 2, 1)
    plt.tricontourf(easting, northing, gravity, levels=50, cmap='viridis')
    plt.colorbar(label='Gravity disturbance (mGal)')
    plt.xlabel('Easting [m]')
    plt.ylabel('Northing [m]')
    plt.title('Observed Gravity Disturbance')
    plt.subplot(1, 2, 2)
    plt.tricontourf(easting, northing, gravity_calculated, levels=50, cmap='viridis')
    plt.colorbar(label='Gravity disturbance (mGal)')
    plt.xlabel('Easting [m]')
    plt.ylabel('Northing [m]')
    plt.title('Fitted Gravity Disturbance (2nd Order Polynomial)')
    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=4, column=0, padx=5, pady=5)
    return gravity_calculated, a_opt, b_opt, c_opt, d_opt, e_opt, f_opt
