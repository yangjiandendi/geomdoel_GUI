"""
gempy_utils.py

This module provides utility functions to create a GemPy geological model
from drawing data (cross‑section points and user‐defined layer positions).

Functions include:
    - cal_orientation: Compute normalized orientation between two points.
    - surface_all: Combine X and Y cross‑section points into a single DataFrame.
    - create_gempy_model: Compute orientations from drawn cross‑sections, save
      the required CSV files, and create a GemPy GeoModel.
      
Usage:
    Import this module and call create_gempy_model(), passing in the drawing data
    (lists of numpy arrays for cross‑sections, saved layer positions, and formation names),
    plus a flag for auto correction.
"""

import numpy as np
import pandas as pd
import gempy as gp
import gempy_viewer as gpv
import os

def cal_orientation(pt1, pt2):
    """
    Calculate the orientation between two points in a cross‑section.
    
    For a pair of points, this function computes the normalized differences in the 
    vertical and horizontal directions. (Note: In your application the returned values 
    are used to form a pole vector in a 3D coordinate system.)
    
    Parameters:
        pt1 (array-like): First point with at least two elements [x, y].
        pt2 (array-like): Second point with at least two elements [x, y].
        
    Returns:
        tuple: (orient_component, other_component) where:
            - For an X cross‑section, the returned tuple will be used as (G_x, G_z)
              with an implied zero in the Y direction.
            - For a Y cross‑section, the returned tuple will be used as (G_y, G_z)
              with an implied zero in the X direction.
    """
    diff_y = pt2[1] - pt1[1]
    diff_x = pt2[0] - pt1[0]
    norm = np.sqrt(diff_y**2 + diff_x**2)
    if norm == 0:
        return (0, 0)
    return (diff_y / norm, diff_x / norm)

def surface_all(x_points, y_points, x_pos_saved, y_pos_saved, name_saved, auto_corr):
    """
    Combine the drawn X and Y cross‑section points into a single DataFrame of surface points.
    
    For each layer:
      - The X cross‑section points are used to create a set of points with X coordinates
        taken from the drawing, a constant Y given by x_pos_saved, and Z as the negative
        of the drawn Y value.
      - The Y cross‑section points are used to create a set of points with Y coordinates
        taken from the drawing, a constant X given by y_pos_saved, and Z as the negative
        of the drawn Y value.
      - If auto_corr is enabled, the Y cross‑section points are shifted by the difference
        between the minimum values of the two cross‑sections.
    
    Parameters:
        x_points (list of np.ndarray): List of arrays representing X cross‑section points.
        y_points (list of np.ndarray): List of arrays representing Y cross‑section points.
        x_pos_saved (list): List of X section positions (for each layer).
        y_pos_saved (list): List of Y section positions (for each layer).
        name_saved (list): List of formation names (one per layer).
        auto_corr (bool or int): Flag indicating whether auto correction is enabled.
        
    Returns:
        pandas.DataFrame: DataFrame with columns ('X', 'Y', 'Z', 'formation') containing
                          the surface points for all layers.
    """
    surface_all_df = pd.DataFrame(columns=['X', 'Y', 'Z', 'formation'])
    
    # Process each layer
    for i in range(len(x_points)):
        slice_x = np.array(x_points[i])
        slice_y = np.array(y_points[i])
        
        # Find the minimum y values in each slice (assumed to represent the lowest point)
        slice_x_min_y = np.min(slice_x[:, 1])
        slice_y_min_y = np.min(slice_y[:, 1])
        # Get corresponding x coordinate (first occurrence) for each minimum y
        slice_x_max = np.array([slice_x[slice_x[:, 1] == slice_x_min_y][0, 0], slice_x_min_y])
        slice_y_max = np.array([slice_y[slice_y[:, 1] == slice_y_min_y][0, 0], slice_y_min_y])
        
        # If auto correction is enabled, shift the Y cross‑section points.
        if auto_corr:
            diff = slice_x_max - slice_y_max
            slice_y[:, 1] = slice_y[:, 1] + diff[1]
        
        # Build DataFrame for X cross‑section:
        x_slice = pd.DataFrame({
            'X': slice_x[:, 0],
            'Y': np.full(len(slice_x), int(x_pos_saved[i])),
            'Z': -slice_x[:, 1]
        })
        # Build DataFrame for Y cross‑section:
        y_slice = pd.DataFrame({
            'X': np.full(len(slice_y), int(y_pos_saved[i])),
            'Y': slice_y[:, 0],
            'Z': -slice_y[:, 1]
        })
        formation = pd.DataFrame({'formation': [name_saved[i]] * (len(x_slice) + len(y_slice))})
        
        # Concatenate the two sections
        surface = pd.concat([pd.concat([x_slice, y_slice], ignore_index=True), formation], axis=1)
        surface_all_df = pd.concat([surface_all_df, surface], ignore_index=True)
    
    return surface_all_df

def create_gempy_model(auto_corr, x_points, y_points, x_pos_saved, y_pos_saved, name_saved):
    """
    Create a GemPy geological model based on drawn cross‑sections and user‑defined parameters.
    
    This function performs the following steps:
      1. For each layer, it computes orientations from the drawn X and Y cross‑sections.
         For the X cross‑section, the orientation vector is formed as [G_x, 0, G_z],
         and for the Y cross‑section, it is [0, G_x, G_z] (with G_x and G_z computed via cal_orientation).
      2. It aggregates the orientation data into a list of dictionaries.
      3. The orientation data is converted to a DataFrame and saved as "Orientations.csv".
      4. The surface points for each layer are computed by combining the X and Y sections
         (using surface_all) and saved as "Surface_points.csv".
      5. Finally, a GemPy model is created using gp.create_geomodel with an ImporterHelper
         that reads the two CSV files.
    
    Parameters:
        auto_corr (bool or int): Flag indicating whether to auto-correct the Y cross‑section.
        x_points (list of np.ndarray): List of drawn X cross‑section points (one array per layer).
        y_points (list of np.ndarray): List of drawn Y cross‑section points (one array per layer).
        x_pos_saved (list): List of X section positions (for each layer).
        y_pos_saved (list): List of Y section positions (for each layer).
        name_saved (list): List of formation names (one per layer).
        
    Returns:
        gempy.data.GeoModel: The constructed GemPy model.
    """
    orientation_data = []
    
    # Ensure the lists are numpy arrays (of object dtype to preserve subarrays)
    x_points_arr = np.array(x_points, dtype=object)
    y_points_arr = np.array(y_points, dtype=object)
    
    # Loop over each layer
    for n in range(x_points_arr.shape[0]):
        slice_x = np.array(x_points_arr[n])
        slice_y = np.array(y_points_arr[n])
        
        # Find the minimum y value (assumed to be the lowest point) in each slice
        slice_x_min_y = np.min(slice_x[:, 1])
        slice_y_min_y = np.min(slice_y[:, 1])
        slice_x_max = np.array([slice_x[slice_x[:, 1] == slice_x_min_y][0, 0], slice_x_min_y])
        slice_y_max = np.array([slice_y[slice_y[:, 1] == slice_y_min_y][0, 0], slice_y_min_y])
        
        if auto_corr:
            diff = slice_x_max - slice_y_max
            slice_y[:, 1] = slice_y[:, 1] + diff[1]
        
        # Generate index pairs for consecutive points along each cross‑section
        indices_x = list(range(slice_x.shape[0] - 1))
        indices_y = list(range(slice_y.shape[0] - 1))
        
        # For the X cross‑section:
        for i in indices_x:
            orient = cal_orientation(slice_x[i], slice_x[i + 1])
            # For X cross‑section, the orientation vector is [G_x, 0, G_z]
            pole_x = [orient[0], 0, orient[1]]
            orientation_data.append({
                'x': slice_x[i][0],
                'y': int(x_pos_saved[n]),
                'z': -slice_x[i][1],
                'surface': name_saved[n],
                'G_x': pole_x[0],
                'G_y': pole_x[1],
                'G_z': pole_x[2]
            })
        # For the Y cross‑section:
        for i in indices_y:
            orient = cal_orientation(slice_y[i], slice_y[i + 1])
            # For Y cross‑section, the orientation vector is [0, G_x, G_z]
            pole_y = [0, orient[0], orient[1]]
            orientation_data.append({
                'x': int(y_pos_saved[n]),
                'y': slice_y[i][0],
                'z': -slice_y[i][1],
                'surface': name_saved[n],
                'G_x': pole_y[0],
                'G_y': pole_y[1],
                'G_z': pole_y[2]
            })
    
    # Create DataFrame from orientation_data and save to CSV.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    orientations_df = pd.DataFrame(orientation_data)
    orientations_df.to_csv(os.path.join(script_dir,"Orientations.csv"), index=False)
    
    # Compute the surface points from the drawn cross‑sections.
    surface_points = surface_all(x_points, y_points, x_pos_saved, y_pos_saved, name_saved, auto_corr)
    surface_points.to_csv(os.path.join(script_dir,"Surface_points.csv"), index=False)
    
    # Create the GemPy geological model using the CSV files.
    geo_model = gp.create_geomodel(
        project_name='model',
        extent=[0, 800, 0, 800, -600, 0],
        resolution=[50, 50, 50],
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=os.path.join(script_dir,"Orientations.csv"),
            path_to_surface_points=os.path.join(script_dir,"Surface_points.csv")
        )
    )
    
    return geo_model

def plot_model_2d(geo_model, plot_frame):
    """
    Plot the 2D view of the GemPy model and embed it into a Tkinter frame.
    
    Parameters:
        geo_model: A GemPy geological model.
        plot_frame: The Tkinter frame in which the 2D plot will be embedded.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    # Create subplots for X and Y directions
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot for the x-direction
    viewer_x = gpv.plot_2d(geo_model, direction='x', show_data=True)
    for ax in viewer_x.axes:
        fig_x = ax.get_figure()
        fig_x.canvas.draw()
        axes[0].imshow(np.array(fig_x.canvas.renderer.buffer_rgba()))
        axes[0].axis('off')
    
    # Plot for the y-direction
    viewer_y = gpv.plot_2d(geo_model, direction='y', show_data=True)
    for ax in viewer_y.axes:
        fig_y = ax.get_figure()
        fig_y.canvas.draw()
        axes[1].imshow(np.array(fig_y.canvas.renderer.buffer_rgba()))
        axes[1].axis('off')
    
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

def plot_model_3d(geo_model, plot_frame=None):
    """
    Plot the 3D view of the GemPy model.
    
    Parameters:
        geo_model: A GemPy geological model.
        plot_frame: (Optional) A Tkinter frame if you wish to embed the 3D view.
                   Otherwise, this function will open a new window.
    
    Note:
        Currently, GemPy’s 3D viewer opens its own window.
    """
    # The gempy_viewer 3D plotting function opens a separate window.
    gpv.plot_3d(model=geo_model, show_surfaces=False, show_data=True, show_lith=False, image=False)
