"""
vtk_utils.py

This module provides utility functions for operations on VTK meshes, including
loading files, generating normals, extracting vertices/normals, saving data to Excel,
and computing scalar fields.
"""

import pyvista as pv
import vtk
import numpy as np
import pandas as pd

def load_vtk_mesh(file_path):
    """
    Load a VTK file (.vtp or .vtk) and return the mesh.
    
    Parameters:
        file_path (str): The path to the VTK file.
    
    Returns:
        mesh (pyvista.PolyData): The loaded VTK mesh.
    """
    return pv.read(file_path)

def generate_normals(polydata):
    """
    Generate normals for the given vtkPolyData if not already present.
    
    Parameters:
        polydata (vtk.vtkPolyData): The input polydata.
    
    Returns:
        polydata_with_normals (vtk.vtkPolyData): Polydata with computed point normals.
    """
    normal_generator = vtk.vtkPolyDataNormals()
    normal_generator.SetInputData(polydata)
    normal_generator.ComputePointNormalsOn()
    normal_generator.ComputeCellNormalsOff()
    normal_generator.Update()
    return normal_generator.GetOutput()

def get_vertices_and_normals(mesh):
    """
    Extract vertices and normals from a PyVista mesh.
    
    Parameters:
        mesh (pyvista.PolyData): The input mesh.
    
    Returns:
        vertices (list): List of (x, y, z) vertex coordinates.
        normals (list): List of (nx, ny, nz) normals.
    """
    surface_mesh = mesh.extract_surface()
    polydata_with_normals = generate_normals(surface_mesh)
    points = polydata_with_normals.GetPoints()
    vertices = [points.GetPoint(i) for i in range(points.GetNumberOfPoints())]
    normals_array = polydata_with_normals.GetPointData().GetNormals()
    normals = [normals_array.GetTuple(i) for i in range(normals_array.GetNumberOfTuples())]
    return vertices, normals

def save_mesh_data_to_excel(vertices, normals, vertices_file, normals_file):
    """
    Save vertices and normals to Excel files.
    
    Parameters:
        vertices (list): List of vertex coordinates.
        normals (list): List of normal vectors.
        vertices_file (str): Output path for vertices Excel file.
        normals_file (str): Output path for normals Excel file.
    """
    vertices_df = pd.DataFrame(vertices, columns=['X', 'Y', 'Z'])
    normals_df = pd.DataFrame(normals, columns=['x', 'y', 'z'])
    vertices_df.to_excel(vertices_file, index=False)
    normals_df.to_excel(normals_file, index=False)

def vtk_to_scalar(mesh):
    """
    Convert a VTK mesh to a scalar field by computing distances on a uniform grid.
    
    Parameters:
        mesh (pyvista.PolyData): The input VTK mesh.
    
    Returns:
        scalar_field (numpy.ndarray): A 3D numpy array representing the scalar field.
    """
    surface_mesh = mesh.extract_surface()
    x_min, x_max, y_min, y_max, z_min, z_max = surface_mesh.bounds
    padding = 10
    grid = pv.ImageData()
    grid.dimensions = [50, 50, 50]
    grid.origin = [x_min - padding, y_min - padding, z_min - padding]
    grid.spacing = [(x_max - x_min + 2 * padding) / (grid.dimensions[0] - 1),
                    (y_max - y_min + 2 * padding) / (grid.dimensions[1] - 1),
                    (z_max - z_min + 2 * padding) / (grid.dimensions[2] - 1)]
    distances = grid.compute_implicit_distance(surface_mesh, inplace=False)
    grid.point_data['Distance'] = distances.point_data['implicit_distance']
    scalar = np.array(grid.point_data['Distance'])
    scalar_field = scalar.reshape(50, 50, 50)
    return scalar_field
