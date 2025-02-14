"""
gui.py

This module defines the GeologicalModelApp class, which is responsible for the entire
Tkinter user interface. It creates menus, tabs, and delegates computational tasks
to functions defined in gempy_utils.py, vtk_utils.py, rbf_utils.py, and gravity.py.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk, font
from PIL import Image, ImageTk
import json
import os
import sys
import threading
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import gempy as gp
import gempy_viewer as gpv
import vtk
import warnings
import itertools
import traceback
import subprocess

# Ignore some warnings (as in your original code)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Import backend functionality from separate modules
from gempy_utils import create_gempy_model, plot_model_2d, plot_model_3d
from vtk_utils import load_vtk_mesh, generate_normals, get_vertices_and_normals, save_mesh_data_to_excel, vtk_to_scalar
from rbf_utils import (
    RBF_3D_kernel,
    systematic_sampling,
    kmeans_sampling_indices,
    evaluate_params_kfold
)
from gravity import gravity_sim, gravity_fwd_2d

class GeologicalModelApp:
    """
    The main GUI application class.
    
    This class builds the main window, menus, and tabs for all functionality:
      - Saving/loading data
      - Creating GemPy models
      - Drawing cross‑sections
      - VTK operations (loading, computing scalar fields)
      - RBF interpolation and cross‑validation
      - Scalar field comparison
      - Gravity simulation
    """
    def __init__(self, root):
        """
        Initialize the application window and create all menus and tabs.
        
        Parameters:
            root (tk.Tk): The root Tkinter window.
        """
        self.root = root
        self.root.geometry("1920x1080")
        self.menu_font = font.Font(family='Arial', size=16)
        self.root.option_add('*Menu.font', self.menu_font)
        self.menubar = tk.Menu(root)
        root.config(menu=self.menubar)
        
        # Create menus for different function groups
        self.function_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Model generator", menu=self.function_menu)
        self.process_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Model Process", menu=self.process_menu)
        self.interpolation_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Interpolation and Comparison", menu=self.interpolation_menu)
        self.menubar.entryconfig("Model generator", font=self.menu_font)
        
        # Add commands to menus that call show_tab() with the appropriate tab and title
        self.function_menu.add_command(label="save/load model", command=lambda: self.show_tab(self.tab1, 'Save/Load Data'))
        self.function_menu.add_command(label="Create GemPy Model", command=lambda: self.show_tab(self.tab2, 'Create GemPy Model'))
        self.function_menu.add_command(label="Drawing", command=lambda: self.show_tab(self.tab3, 'Drawing'))
        self.function_menu.add_command(label="2D and 3D Plot", command=lambda: self.show_tab(self.tab4, '2D and 3D Plot'))
        self.process_menu.add_command(label="VTK to Scalar Field", command=lambda: self.show_tab(self.tab5, 'VTK to Scalar Field'))
        self.interpolation_menu.add_command(label="Implicit Interpolation", command=lambda: self.show_tab(self.tab6, 'Implicit Interpolation'))
        self.process_menu.add_command(label="Cross-Validation", command=lambda: self.show_tab(self.tab7, 'Cross-Validation'))
        self.interpolation_menu.add_command(label="Scalar Field Comparison", command=lambda: self.show_tab(self.tab8, 'Scalar Field Comparison'))
        self.process_menu.add_command(label="Gravity Simulation", command=lambda: self.show_tab(self.tab11, 'Gravity Simulation'))
        
        # Create a notebook for tabbed interface
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)

        self.style = ttk.Style()
        self.style.configure('TNotebook.Tab', font=('Arial', '16'))
        self.style.configure('TButton', font=('Arial', 14), padding=10, borderwidth=10, relief="flat", background="#C0E6A7")
 
        # Create tabs (some tabs are defined here; others may be built later)
        self.tab1 = ttk.Frame(self.notebook)  # Save/Load Data
        self.tab2 = ttk.Frame(self.notebook)  # GemPy Model creation
        self.tab3 = ttk.Frame(self.notebook)  # Drawing cross‑sections
        self.tab4 = ttk.Frame(self.notebook)  # 2D and 3D Plot
        self.tab5 = ttk.Frame(self.notebook)  # VTK Operations
        self.tab6 = ttk.Frame(self.notebook)  # RBF Interpolation
        self.tab7 = ttk.Frame(self.notebook)  # Cross‑Validation
        self.tab8 = ttk.Frame(self.notebook)  # Scalar Field Comparison
        self.tab9 = ttk.Frame(self.notebook)  # Help/About
        self.tab10 = ttk.Frame(self.notebook) # Workflow Guide
        self.tab11 = ttk.Frame(self.notebook) # Gravity Simulation

        # Add tabs to the notebook (the first visible tab will be inserted via show_tab())
        self.notebook.add(self.tab9, text='Help')
        # self.notebook.add(self.tab10, text='Workflow')

        # Initialize some shared variables used across methods
        self.auto_corr = tk.IntVar(value=0)
        self.show_drift_var = tk.IntVar(value=0)
        self.only_show_data = tk.IntVar(value=0)
        self.name = {}
        self.name_saved = []
        self.x_pos = {}
        self.x_pos_saved = []
        self.y_pos = {}
        self.y_pos_saved = []
        self.x_points = []
        self.y_points = []
        self.vertices_df = None
        self.normals_df = None

        # Define font styles for the UI
        self.title_font = ("Helvetica", 16, "bold")
        self.label_font = ("Helvetica", 14)
        self.label_font_s = ("Helvetica", 10)
        self.entry_font = ("Helvetica", 14)
        self.button_font = ("Helvetica", 14, "bold")

        # -------------- Build Tab 1: Save/Load Data --------------
        self.save_button = ttk.Button(self.tab1, text="Save Data",  command=self.save_data, style='TButton')
        self.save_button.grid(row=0, column=0, padx=10, pady=10, sticky='w')
        self.load_button = ttk.Button(self.tab1, text="Load Data",  command=self.load_data, style='TButton')
        self.load_button.grid(row=0, column=1, padx=10, pady=10, sticky='w')
        self.save_model_button = ttk.Button(self.tab1, text="Save Gempy Model",  command=self.save_gempy_model, style='TButton')
        self.save_model_button.grid(row=0, column=2, padx=10, pady=10, sticky='w')

        # -------------- Build Tab 2: Create GemPy Model --------------
        self.Layer_number_label = tk.Label(self.tab2, text="Please enter the layer number:", font=self.label_font)
        self.layer_number_entry = tk.Entry(self.tab2, font=self.entry_font)
        self.layer_number_confirm = ttk.Button(self.tab2, text="Confirm", style='TButton', command=self.set_layer_name)
        self.Layer_number_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')
        self.layer_number_entry.grid(row=0, column=1, padx=10, pady=10, sticky='w')
        self.layer_number_confirm.grid(row=0, column=2, padx=10, pady=10, sticky='w')
        self.en = tk.Checkbutton(self.tab2, text="Enable auto correction", variable=self.auto_corr, onvalue=1, offvalue=0, font=self.label_font)
        self.en.grid(row=1, column=0, padx=10, pady=10, sticky='w')

        # -------------- Build Tab 3: Drawing --------------
        self.draw_frame = tk.Frame(self.tab3)
        self.draw_frame.grid(row=0, column=0, padx=10, pady=10, sticky='w')

        # -------------- Build Tab 4: 2D and 3D Plot --------------
        self.plot_2d_frame = tk.Frame(self.tab4)
        self.plot_2d_frame.grid(row=0, column=0, padx=10, pady=10, sticky='w')
        self.plot_3d_button = ttk.Button(self.plot_2d_frame, text="Show 3D Plot", style='TButton', command=self.plot_3d)
        self.plot_3d_button.pack(pady=10)

        # -------------- Build Tab 5: VTK Operations --------------
        self.vtk_operations_frame = tk.Frame(self.tab5)
        self.vtk_operations_frame.grid(row=0, column=0, padx=10, pady=10, sticky='w')
        self.load_vtk_button = ttk.Button(self.vtk_operations_frame, text="Load VTK File", style='TButton', command=self.load_vtk_file)
        self.load_vtk_button.grid(row=0, column=0, padx=10, pady=10, sticky='w')
        self.compute_scalar_button = ttk.Button(self.vtk_operations_frame, text="Compute Scalar Field", style='TButton', command=self.compute_scalar_field)
        self.compute_scalar_button.grid(row=0, column=1, padx=10, pady=10, sticky='w')
        self.save_scalar_button = ttk.Button(self.vtk_operations_frame, text="Save Scalar Field", style='TButton', command=self.save_scalar_field)
        self.save_scalar_button.place(relx=0.68, rely=0.19, anchor='center')

        self.vertices_file_path = tk.StringVar()
        self.normals_file_path = tk.StringVar()
        self.scalar_file_path = tk.StringVar()
        tk.Label(self.vtk_operations_frame, text="Vertices File Path:", font=self.label_font).grid(row=1, column=0, padx=10, pady=5, sticky='w')
        self.vertices_entry = tk.Entry(self.vtk_operations_frame, textvariable=self.vertices_file_path, font=self.entry_font, width=50)
        self.vertices_entry.grid(row=1, column=1, padx=10, pady=5, sticky='w')
        tk.Label(self.vtk_operations_frame, text="Normals File Path:", font=self.label_font).grid(row=2, column=0, padx=10, pady=5, sticky='w')
        self.normals_entry = tk.Entry(self.vtk_operations_frame, textvariable=self.normals_file_path, font=self.entry_font, width=50)
        self.normals_entry.grid(row=2, column=1, padx=10, pady=5, sticky='w')
        tk.Label(self.vtk_operations_frame, text="Scalar Field File Path:", font=self.label_font).grid(row=3, column=0, padx=10, pady=5, sticky='w')
        self.scalar_entry = tk.Entry(self.vtk_operations_frame, textvariable=self.scalar_file_path, font=self.entry_font, width=50)
        self.scalar_entry.grid(row=3, column=1, padx=10, pady=5, sticky='w')
        self.save_excel_button = ttk.Button(self.vtk_operations_frame, text="Save to Excel", style='TButton', command=self.save_vtk_to_excel)
        self.save_excel_button.grid(row=4, column=0, padx=10, pady=10, sticky='w')
        self.save_excel_button.place(relx=0.90, rely=0.19, anchor='center')
        self.pv_canvas_frame = tk.Frame(self.tab5, width=800, height=600)
        self.pv_canvas_frame.grid(row=5, column=0, columnspan=3, padx=10, pady=10)
        self.plotter = None
        # The function to initialize the PyVista plotter has been moved into vtk_utils.py (if needed)
        # For now, we simply reserve the frame.

        # -------------- Build Tab 6: RBF Interpolation --------------
        # self.rbf_frame = tk.Frame(self.tab6)
        # self.rbf_frame.grid(row=0, column=0, padx=10, pady=10, sticky='w')
        # self.load_surface_button = ttk.Button(self.rbf_frame, text="Load Surface Points", style='TButton', command=self.load_surface_points)
        # self.load_surface_button.grid(row=0, column=0, padx=10, pady=10, sticky='w')
        # self.load_orientation_button = ttk.Button(self.rbf_frame, text="Load Orientation Points", style='TButton', command=self.load_orientation_points)
        # self.load_orientation_button.grid(row=0, column=1, padx=10, pady=10, sticky='w')
        # self.load_orientations_button = ttk.Button(self.rbf_frame, text="Load Orientations", style='TButton', command=self.load_orientations)
        # self.load_orientations_button.grid(row=0, column=2, padx=10, pady=10, sticky='w')
        # (Additional widgets for RBF interpolation, sampling method, kernel selection, etc. are built here.)
        # For brevity, these have been omitted; in your refactoring move the long widget‐initialization code as needed.
        # self.tab6 = ttk.Frame(self.tab6)
        # self.notebook.add(self.tab6, text="Implicit Interpolation")
        self.create_implicit_interpolation_tab()



        # -------------- Build Tab 7: Cross‑Validation --------------
        self.create_cross_validation_tab()

        # -------------- Build Tab 8: Scalar Field Comparison --------------
        self.scalar1 = None
        self.scalar2 = None
        self.use_flip = tk.BooleanVar()
        self.use_reverse = tk.BooleanVar()
        self.create_scalar_comparison_tab()

        # -------------- Build Tab 9: Help/About --------------
        self.build_help_tab()

        # -------------- Build Tab 10: Workflow Guide --------------
        self.build_workflow_tab()

        # -------------- Build Tab 11: Gravity Simulation --------------
        self.build_gravity_tab()


    def show_tab(self, tab, tab_text):
        """
        Remove all other tabs and insert the specified tab into the notebook.
        
        Parameters:
            tab (ttk.Frame): The tab/frame to show.
            tab_text (str): The text label for the tab.
        """
        # Remove any tabs whose text matches the known names
        for t, text in [(self.tab1, 'Save/Load Data'), (self.tab2, 'Create GemPy Model'),
                        (self.tab3, 'Drawing'), (self.tab4, '2D and 3D Plot'), (self.tab5, 'VTK to Scalar Field'),
                        (self.tab6, 'Implicit Interpolation'), (self.tab7, 'Cross-Validation'), (self.tab8, 'Scalar Field Comparison'),
                        (self.tab11, 'Gravity Simulation')]:
            if text in [self.notebook.tab(nt, "text") for nt in self.notebook.tabs()]:
                self.notebook.forget(t)
        self.notebook.insert(0, tab, text=tab_text)
        self.notebook.select(tab)

    def combine_funcs(self, *funcs):
        """
        Return a function that calls multiple functions sequentially.
        
        Parameters:
            funcs: A sequence of functions to call.
            
        Returns:
            A function that, when called, calls each function in sequence.
        """
        def inner_combined_func(*args, **kwargs):
            for f in funcs:
                f(*args, **kwargs)
        return inner_combined_func

    # ------------------ Methods related to GemPy Model Creation ------------------

    def set_layer_name(self):
        """
        Create input fields for each layer’s name and section positions.
        Called when the user confirms the number of layers.
        """
        try:
            num_layers = int(self.layer_number_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer for the layer number.")
            return

        for i in range(num_layers):
            # Create widgets for layer name and section positions
            tk.Label(self.tab2, text="Layer " + str(i + 1) + " name:", font=self.label_font).grid(row=i + 2, column=0, padx=10, pady=5, sticky='w')
            layer_name = f'Layer {str(i + 1)} name'
            self.name[layer_name] = tk.Entry(self.tab2, font=self.entry_font)
            self.name[layer_name].grid(row=i + 2, column=1, padx=10, pady=5, sticky='w')
            tk.Label(self.tab2, text="Layer " + str(i + 1) + " X section Position:", font=self.label_font).grid(row=i + 2, column=2, padx=10, pady=5, sticky='w')
            self.x_pos[layer_name] = tk.Entry(self.tab2, font=self.entry_font)
            self.x_pos[layer_name].grid(row=i + 2, column=3, padx=10, pady=5, sticky='w')
            tk.Label(self.tab2, text="Layer " + str(i + 1) + " Y section Position:", font=self.label_font).grid(row=i + 2, column=4, padx=10, pady=5, sticky='w')
            self.y_pos[layer_name] = tk.Entry(self.tab2, font=self.entry_font)
            self.y_pos[layer_name].grid(row=i + 2, column=5, padx=10, pady=5, sticky='w')
        tk.Button(self.tab2, text="Draw!", font=self.button_font,
                  command=self.combine_funcs(self.save_name, self.set_draw, lambda: self.show_tab(self.tab3, 'Drawing'))
                 ).grid(row=i + 3, column=0, padx=10, pady=10, sticky='w')

    def save_name(self):
        """
        Save the names and positions entered for each layer.
        """
        num_layers = int(self.layer_number_entry.get())
        for i in range(num_layers):
            self.name_saved.append(self.name[f'Layer {str(i + 1)} name'].get())
            self.x_pos_saved.append(self.x_pos[f'Layer {str(i + 1)} name'].get())
            self.y_pos_saved.append(self.y_pos[f'Layer {str(i + 1)} name'].get())

    def set_draw(self):
        """
        Create buttons for drawing the X and Y cross‑sections of each layer.
        """
        num_layers = int(self.layer_number_entry.get())
        for i in range(num_layers):
            ttk.Button(self.draw_frame, text="Please draw Layer " + str(i + 1) + " X cross-section",
                       style='TButton', command=self.draw_X).grid(row=i, column=0, padx=10, pady=5, sticky='w')
            ttk.Button(self.draw_frame, text="Please draw Layer " + str(i + 1) + " Y cross-section",
                       style='TButton', command=self.draw_Y).grid(row=i, column=1, padx=10, pady=5, sticky='w')
        ttk.Button(self.tab3, text="Create model!", style='TButton', command=self.create_model
                  ).place(relx=0.6, rely=0.015)

    def draw_X(self):
        """
        Execute an external drawing script to capture an X cross‑section,
        read the resulting JSON file, and plot the new slice on the drawing frame.
        """
        # Run an external script (assuming you use a magic command or similar)
        # For example, you might call a function from a module instead of %run.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # draw_script = os.path.join(script_dir, "draw.py")
        # subprocess.run([sys.executable, draw_script])
        import draw
        draw.run_pygame_window()
        with open(os.path.join(script_dir,"curve_points.json"), "r") as file:
            points = json.load(file)
        slice_x = np.array(points)[-1]
        self.x_points.append(slice_x)
        fig = plt.Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot()
        for pts in self.x_points:
            ax.plot(pts[:, 0], pts[:, 1], "-")
        ax.set_xlim(0, 800)
        ax.set_ylim(0, 600)
        ax.grid()
        ax.invert_yaxis()
        canvas = FigureCanvasTkAgg(fig, master=self.draw_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=int(self.layer_number_entry.get()) + 10, column=0, rowspan=1, columnspan=5, padx=10, pady=10)

    def draw_Y(self):
        """
        Execute an external drawing script to capture a Y cross‑section,
        read the resulting JSON file, and plot the new slice on the drawing frame.
        """
        import json
        import numpy as np
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # draw_script = os.path.join(script_dir, "draw.py")
        # subprocess.run([sys.executable, draw_script])
        import draw
        draw.run_pygame_window()
        with open(os.path.join(script_dir,"curve_points.json"), "r") as file:
            points = json.load(file)
        slice_y = np.array(points)[-1]
        self.y_points.append(slice_y)
        fig = plt.Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot()
        for pts in self.y_points:
            ax.plot(pts[:, 0], pts[:, 1], "-")
        ax.set_xlim(0, 800)
        ax.set_ylim(0, 600)
        ax.grid()
        ax.invert_yaxis()
        canvas = FigureCanvasTkAgg(fig, master=self.draw_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=int(self.layer_number_entry.get()) + 10, column=5, rowspan=1, columnspan=5, padx=10, pady=10)

  
        # """
        # Load drawing data from a JSON file and automatically create the GemPy model.
        # """
        # import json, numpy as np
        # file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        # if file_path:
        #     with open(file_path, "r") as file:
        #         data = json.load(file)
        #     self.name_saved = data["name_saved"]
        #     self.x_pos_saved = data["x_pos_saved"]
        #     self.y_pos_saved = data["y_pos_saved"]
        #     self.x_points = [np.array(pts) for pts in data["x_points"]]
        #     self.y_points = [np.array(pts) for pts in data["y_points"]]
        #     messagebox.showinfo("Load Data", "Data loaded successfully!")
        #     self.create_model()

    def create_model(self):
        """
        Create the GemPy geological model using the saved layer data.
        This method calls the backend function (from gempy_utils) to create and compute the model.
        """
        # Call the gempy_utils function; pass in necessary variables.
        geo_model = create_gempy_model(
            auto_corr=self.auto_corr.get(),
            x_points=self.x_points,
            y_points=self.y_points,
            x_pos_saved=self.x_pos_saved,
            y_pos_saved=self.y_pos_saved,
            name_saved=self.name_saved
        )
        gp.compute_model(geo_model)  # Compute the GemPy model
        # Plot the 2D result in the GUI
        plot_model_2d(geo_model, self.plot_2d_frame)

    def plot_3d(self):
        """
        Create and compute the GemPy model and then plot it in 3D.
        """
        geo_model = create_gempy_model(
            auto_corr=self.auto_corr.get(),
            x_points=self.x_points,
            y_points=self.y_points,
            x_pos_saved=self.x_pos_saved,
            y_pos_saved=self.y_pos_saved,
            name_saved=self.name_saved
        )
        gp.compute_model(geo_model)
        plot_model_3d(geo_model, self.plot_2d_frame)

    def save_gempy_model(self):
        """
        Save the computed GemPy model to a specified directory.
        This method uses gempy_viewer to generate VTK meshes for each surface.
        """
        directory = filedialog.askdirectory()
        if directory:
            # Create model and compute solution
            geo_model = create_gempy_model(
                auto_corr=self.auto_corr.get(),
                x_points=self.x_points,
                y_points=self.y_points,
                x_pos_saved=self.x_pos_saved,
                y_pos_saved=self.y_pos_saved,
                name_saved=self.name_saved
            )
            sol = gp.compute_model(geo_model, engine_config=gp.data.GemPyEngineConfig(
                use_gpu=False, dtype='float32', backend=gp.data.AvailableBackends.PYTORCH))
            # Save surface points and orientations to CSV
            surface_points_path = os.path.join(directory, 'surface_points.csv')
            orientations_path = os.path.join(directory, 'orientations.csv')
            geo_model.surface_points_copy.df.to_csv(surface_points_path)
            geo_model.orientations_copy.df.to_csv(orientations_path)
            # Save each surface as a .vtp file
            for name in self.name_saved:
                mesh_path = os.path.join(directory, f'{name}.vtp')
                try:
                    gpv.plot_3d(geo_model, image=False, plotter_type='basic', show_data=0).surface_poly[name].save(mesh_path)
                except KeyError as e:
                    print(f"Error: {e}")
                    messagebox.showerror("Save Error", f"Could not save {name}.vtp. Please check the surface name.")
                    continue
            messagebox.showinfo("Save Gempy Model", "Gempy model saved successfully!")

    # ------------------ Methods related to VTK Operations ------------------

    def load_vtk_file(self):
        """
        Load a VTK file (.vtp or .vtk) and update the file path variables.
        Uses the vtk_utils function load_vtk_mesh().
        """
        file_path = filedialog.askopenfilename(filetypes=[("VTK files", "*.vtp"), ("VTK files", "*.vtk")])
        if file_path:
            self.mesh = load_vtk_mesh(file_path)
            # Update file paths based on the selected file
            if file_path.endswith(".vtp"):
                self.vertices_file_path.set(file_path.replace(".vtp", "_points.xlsx"))
                self.normals_file_path.set(file_path.replace(".vtp", "_normals.xlsx"))
                self.scalar_file_path.set(file_path.replace(".vtp", "_scalar.npy"))
            else:
                self.vertices_file_path.set(file_path.replace(".vtk", "_points.xlsx"))
                self.normals_file_path.set(file_path.replace(".vtk", "_normals.xlsx"))
                self.scalar_file_path.set(file_path.replace(".vtk", "_scalar.npy"))
            messagebox.showinfo("VTK Operations", "VTK file loaded successfully!")

    def save_vtk_to_excel(self):
        """
        Extract vertices and normals from the current VTK mesh and save them to Excel files.
        Uses functions from vtk_utils.
        """
        vertices, normals = get_vertices_and_normals(self.mesh)
        save_mesh_data_to_excel(vertices, normals, self.vertices_file_path.get(), self.normals_file_path.get())
        messagebox.showinfo("VTK Operations", "Vertices and normals saved to Excel successfully!")

    def compute_scalar_field(self):
        """
        Compute a scalar field from the current VTK mesh, store it, and plot it in the GUI.
        """
        scalar_field = vtk_to_scalar(self.mesh)
        self.scalar_field = scalar_field  # Store for later saving
        self.plot_scalar_field_in_gui(scalar_field)

    def plot_scalar_field_in_gui(self, scalar_field):
        """
        Plot the scalar field in a new window (using PyVista) on a separate thread.
        
        Parameters:
            scalar_field (numpy.ndarray): The computed scalar field.
        """
        def plot():
            # This function creates a PyVista plotter to display the scalar field.
            import pyvista as pv
            plotter = pv.Plotter(notebook=False)
            surface_mesh = self.mesh.extract_surface()
            # Create a uniform grid surrounding the mesh (same logic as in vtk_to_scalar)
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
            plotter.add_mesh(surface_mesh, color='white', label='Mesh Surface')
            plotter.add_mesh(grid, scalars='Distance', cmap='viridis', opacity=0.7, label='Scalar Field')
            plotter.add_legend()
            plotter.show()
        # Start plotting in a new thread
        plot_thread = threading.Thread(target=plot)
        plot_thread.start()

    def save_scalar_field(self):
        """
        Save the computed scalar field to a .npy file.
        """
        file_path = self.scalar_file_path.get()
        if file_path:
            np.save(file_path, vtk_to_scalar(self.mesh))
            messagebox.showinfo("Save Scalar Field", "Scalar field saved successfully!")

    # ------------------ Methods related to RBF Interpolation and Cross‑Validation ------------------

    def load_surface_points(self):
        """
        Load surface points from an Excel file.
        """
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            self.vertices_df = pd.read_excel(file_path)
            messagebox.showinfo("RBF Interpolation", "Surface points loaded successfully!")
            self.model_name = os.path.basename(file_path).split('_')[0]

    def load_orientation_points(self):
        """
        Load orientation points from an Excel file.
        """
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            self.normals_df = pd.read_excel(file_path)
            messagebox.showinfo("RBF Interpolation", "Orientation points loaded successfully!")

    def load_orientations(self):
        """
        Load orientations from an Excel file.
        """
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            self.normals_o_df = pd.read_excel(file_path)
            messagebox.showinfo("RBF Interpolation", "Orientations loaded successfully!")

    def load_cv_surface_points(self):
        """
        Load surface points for cross-validation from an Excel file.
        """
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            self.cv_vertices_df = pd.read_excel(file_path)
            messagebox.showinfo("RBF Interpolation", "Surface points loaded successfully!")

    def load_cv_orientation_points(self):
        """
        Load orientation points for cross-validation from an Excel file.
        """
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            self.cv_normals_df = pd.read_excel(file_path)
            messagebox.showinfo("RBF Interpolation", "Orientation points loaded successfully!")

    def create_cross_validation_tab(self):
        """
        Create and arrange the widgets for the cross-validation tab.
        
        This tab allows the user to load cross-validation surface and orientation points,
        choose a sampling method, specify the number of sample points, select a kernel function,
        choose a drift type, and set parameter ranges for eps, range, and drift parameters.
        It also displays a progress bar, elapsed/remaining time, and the final result.
        """
        self.cv_frame = tk.Frame(self.tab7)
        self.cv_frame.grid(row=0, column=0, padx=10, pady=10, sticky='w')

        self.load_surface_cv_button = ttk.Button(
            self.cv_frame, text="Load Surface Points", style='TButton', command=self.load_cv_surface_points
        )
        self.load_surface_cv_button.grid(row=0, column=0, padx=10, pady=10, sticky='w')

        self.load_orientation_cv_button = ttk.Button(
            self.cv_frame, text="Load Orientation Points", style='TButton', command=self.load_cv_orientation_points
        )
        self.load_orientation_cv_button.grid(row=0, column=1, padx=10, pady=10, sticky='w')

        tk.Label(self.cv_frame, text="Sampling method:", font=self.label_font).grid(
            row=0, column=2, padx=10, pady=5, sticky='w'
        )
        self.Sampling_method_cv = ttk.Combobox(
            self.cv_frame, font=self.entry_font,
            values=['Random', 'Systematic', 'Statistic(K-means)']
        )
        self.Sampling_method_cv.grid(row=0, column=3, padx=10, pady=5, sticky='w')

        tk.Label(self.cv_frame, text="Number of Sample Surface Points:", font=self.label_font).grid(
            row=1, column=0, padx=10, pady=5, sticky='w'
        )
        self.num_surface_points_cv = tk.Entry(self.cv_frame, font=self.entry_font)
        self.num_surface_points_cv.grid(row=1, column=1, padx=10, pady=5, sticky='w')

        tk.Label(self.cv_frame, text="Number of Sample Orientation Points:", font=self.label_font).grid(
            row=2, column=0, padx=10, pady=5, sticky='w'
        )
        self.num_orientation_points_cv = tk.Entry(self.cv_frame, font=self.entry_font)
        self.num_orientation_points_cv.grid(row=2, column=1, padx=10, pady=5, sticky='w')

        tk.Label(self.cv_frame, text="Kernel Function:", font=self.label_font).grid(
            row=3, column=0, padx=10, pady=5, sticky='w'
        )
        self.kernel_function_cv = ttk.Combobox(
            self.cv_frame, font=self.entry_font,
            values=['cubic', 'gaussian', 'quintic', 'cubic_covariance', 'multiquadric',
                    'linear', 'thin_plate', 'inverse_quadratic', 'inverse_multiquadric',
                    'gaussian_covariance', 'ga_cub_cov', 'ga_cub']
        )
        self.kernel_function_cv.grid(row=3, column=1, padx=10, pady=5, sticky='w')

        tk.Label(self.cv_frame, text="Drift Type:", font=self.label_font).grid(
            row=4, column=0, padx=10, pady=5, sticky='w'
        )
        self.drift_type_cv = ttk.Combobox(
            self.cv_frame, font=self.entry_font,
            values=['None', 'First Order Polynomial', 'Second Order Polynomial', 'Ellipsoid', 'Dome Shaped', '2D Custom','3D Custom']
        )
        self.drift_type_cv.grid(row=4, column=1, padx=10, pady=5, sticky='w')

        self.load_drift_scalar_button_cv = ttk.Button(self.cv_frame, text="Load Drift Scalar Field", style='TButton', command=self.load_drift_scalar)
        self.load_drift_scalar_button_cv.grid(row=4, column=2, padx=10, pady=10, sticky='w')

        tk.Label(self.cv_frame, text="Drift scaling factor:", font=self.label_font).grid(row=4, column=3, padx=10, pady=5, sticky='w')
        self.scaling_factor_cv = tk.Entry(self.cv_frame, font=self.entry_font)
        self.scaling_factor_cv.grid(row=4, column=4, padx=10, pady=5, sticky='w')

        # Entries for epsilon (eps) range
        tk.Label(self.cv_frame, text="eps range (start, stop, step):", font=self.label_font).grid(
            row=5, column=0, padx=10, pady=5, sticky='w'
        )
        self.epsilon_start_cv = tk.Entry(self.cv_frame, font=self.entry_font, width=5)
        self.epsilon_start_cv.place(relx=0.63, rely=0.36, anchor='center')
        self.epsilon_stop_cv = tk.Entry(self.cv_frame, font=self.entry_font, width=5)
        self.epsilon_stop_cv.place(relx=0.73, rely=0.36, anchor='center')
        self.epsilon_step_cv = tk.Entry(self.cv_frame, font=self.entry_font, width=5)
        self.epsilon_step_cv.place(relx=0.83, rely=0.36, anchor='center')

        # Entries for range value range
        tk.Label(self.cv_frame, text="Range value range (start, stop, step):", font=self.label_font).grid(
            row=6, column=0, padx=10, pady=5, sticky='w'
        )
        self.range_start_cv = tk.Entry(self.cv_frame, font=self.entry_font, width=5)
        self.range_start_cv.place(relx=0.63, rely=0.42, anchor='center')
        self.range_stop_cv = tk.Entry(self.cv_frame, font=self.entry_font, width=5)
        self.range_stop_cv.place(relx=0.73, rely=0.42, anchor='center')
        self.range_step_cv = tk.Entry(self.cv_frame, font=self.entry_font, width=5)
        self.range_step_cv.place(relx=0.83, rely=0.42, anchor='center')

        # Entries for drift parameter 'a' range
        tk.Label(self.cv_frame, text="Parameter a range (start, stop, step):", font=self.label_font).grid(
            row=7, column=0, padx=10, pady=5, sticky='w'
        )
        self.param_a_start_cv = tk.Entry(self.cv_frame, font=self.entry_font, width=5)
        self.param_a_start_cv.place(relx=0.63, rely=0.48, anchor='center')
        self.param_a_stop_cv = tk.Entry(self.cv_frame, font=self.entry_font, width=5)
        self.param_a_stop_cv.place(relx=0.73, rely=0.48, anchor='center')
        self.param_a_step_cv = tk.Entry(self.cv_frame, font=self.entry_font, width=5)
        self.param_a_step_cv.place(relx=0.83, rely=0.48, anchor='center')

        # Entries for drift parameter 'b' range
        tk.Label(self.cv_frame, text="Parameter b range (start, stop, step):", font=self.label_font).grid(
            row=8, column=0, padx=10, pady=5, sticky='w'
        )
        self.param_b_start_cv = tk.Entry(self.cv_frame, font=self.entry_font, width=5)
        self.param_b_start_cv.place(relx=0.63, rely=0.54, anchor='center')
        self.param_b_stop_cv = tk.Entry(self.cv_frame, font=self.entry_font, width=5)
        self.param_b_stop_cv.place(relx=0.73, rely=0.54, anchor='center')
        self.param_b_step_cv = tk.Entry(self.cv_frame, font=self.entry_font, width=5)
        self.param_b_step_cv.place(relx=0.83, rely=0.54, anchor='center')

        # Entries for drift parameter 'c' range
        tk.Label(self.cv_frame, text="Parameter c range (start, stop, step):", font=self.label_font).grid(
            row=9, column=0, padx=10, pady=5, sticky='w'
        )
        self.param_c_start_cv = tk.Entry(self.cv_frame, font=self.entry_font, width=5)
        self.param_c_start_cv.place(relx=0.63, rely=0.60, anchor='center')
        self.param_c_stop_cv = tk.Entry(self.cv_frame, font=self.entry_font, width=5)
        self.param_c_stop_cv.place(relx=0.73, rely=0.60, anchor='center')
        self.param_c_step_cv = tk.Entry(self.cv_frame, font=self.entry_font, width=5)
        self.param_c_step_cv.place(relx=0.83, rely=0.60, anchor='center')

        self.compute_cv_button = ttk.Button(
            self.cv_frame, text="Compute Cross-Validation", style='TButton', command=self.run_cross_validation
        )
        self.compute_cv_button.grid(row=10, column=0, padx=10, pady=10, sticky='w')

        self.progress = ttk.Progressbar(self.cv_frame, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress.grid(row=12, column=0, columnspan=4, padx=10, pady=10, sticky='w')

        self.elapsed_time_label = tk.Label(self.cv_frame, text="", font=self.label_font)
        self.elapsed_time_label.grid(row=13, column=0, columnspan=4, padx=10, pady=10, sticky='w')

        self.remaining_time_label = tk.Label(self.cv_frame, text="", font=self.label_font)
        self.remaining_time_label.grid(row=14, column=0, columnspan=4, padx=10, pady=10, sticky='w')

        self.result_label = tk.Label(self.cv_frame, text="", font=self.label_font)
        self.result_label.grid(row=11, column=0, columnspan=4, padx=10, pady=10, sticky='w')

    def create_scalar_comparison_tab(self):
        """
        Create and arrange the widgets for the scalar field comparison tab.
        
        This tab allows the user to load up to five scalar fields, select a comparison plane,
        choose whether to flip or reverse Scalar Field 1, and then compare the fields.
        """
        self.scalar_comparison_frame = tk.Frame(self.tab8)
        self.scalar_comparison_frame.grid(row=0, column=0, padx=10, pady=10, sticky='w')

        self.load_scalar1_button = ttk.Button(
            self.scalar_comparison_frame, text="Load Scalar Field 1", style='TButton', command=self.load_scalar1
        )
        self.load_scalar1_button.grid(row=0, column=0, padx=10, pady=10, sticky='w')

        self.load_scalar2_button = ttk.Button(
            self.scalar_comparison_frame, text="Load Scalar Field 2", style='TButton', command=self.load_scalar2
        )
        self.load_scalar2_button.grid(row=0, column=1, padx=10, pady=10, sticky='w')

        self.load_scalar3_button = ttk.Button(
            self.scalar_comparison_frame, text="Load Scalar Field 3", style='TButton', command=self.load_scalar3
        )
        self.load_scalar3_button.grid(row=0, column=2, padx=10, pady=10, sticky='w')

        self.load_scalar4_button = ttk.Button(
            self.scalar_comparison_frame, text="Load Scalar Field 4", style='TButton', command=self.load_scalar4
        )
        self.load_scalar4_button.grid(row=0, column=3, padx=10, pady=10, sticky='w')

        self.load_scalar5_button = ttk.Button(
            self.scalar_comparison_frame, text="Load Scalar Field 5", style='TButton', command=self.load_scalar5
        )
        self.load_scalar5_button.grid(row=0, column=4, padx=10, pady=10, sticky='w')

        tk.Label(self.scalar_comparison_frame, text="Compare Plane:", font=self.label_font).grid(
            row=1, column=0, padx=10, pady=5, sticky='w'
        )
        self.compare_plane = ttk.Combobox(
            self.scalar_comparison_frame, font=self.entry_font, values=['x', 'y', 'z']
        )
        self.compare_plane.grid(row=1, column=1, padx=10, pady=5, sticky='w')
        self.compare_plane.current(1)

        self.flip_checkbox = tk.Checkbutton(
            self.scalar_comparison_frame, text="Flip Scalar Field 1", variable=self.use_flip, font=self.label_font
        )
        self.flip_checkbox.grid(row=2, column=0, padx=10, pady=5, sticky='w')

        self.reverse_checkbox = tk.Checkbutton(
            self.scalar_comparison_frame, text="Reverse Scalar Field 1", variable=self.use_reverse, font=self.label_font
        )
        self.reverse_checkbox.grid(row=2, column=1, padx=10, pady=5, sticky='w')

        self.compare_button = ttk.Button(
            self.scalar_comparison_frame, text="Compare", style='TButton', command=self.compare_scalars
        )
        self.compare_button.grid(row=3, column=0, padx=10, pady=10, sticky='w')

        self.scalar_canvas_frame = tk.Frame(self.tab8)
        self.scalar_canvas_frame.grid(row=4, column=0, padx=10, pady=10, columnspan=4)

    def create_implicit_interpolation_tab(self):
        """
        Create and arrange all widgets for the Implicit Interpolation (RBF) tab.
        
        This tab provides controls for:
        - Loading surface points, orientation points, and orientations (using separate buttons)
        - Choosing a sampling method (Random, Systematic, or Statistic(K-means))
        - Entering the number of sample surface points and orientation points
        - Selecting a kernel function and a drift type (with options for None, First/Second Order Polynomial, Ellipsoid, Dome Shaped, 2D Custom, or 3D Custom)
        - Optionally loading a drift scalar field and setting a drift scaling factor
        - Optionally entering values for epsilon and range (if you wish to override the calculated defaults)
        - Optionally setting drift parameters (a, b, c)
        - Checkboxes for showing the drift contribution and/or only showing data
        - Selecting a model color
        - A button to compute RBF interpolation that launches the process and displays the result in a PyVista window.
        """
        # Create a frame inside tab6
        self.rbf_frame = tk.Frame(self.tab6)
        self.rbf_frame.grid(row=0, column=0, padx=10, pady=10, sticky='w')
        
        # Row 0: Load data buttons
        self.load_surface_button = ttk.Button(
            self.rbf_frame, text="Load Surface Points", style='TButton', command=self.load_surface_points
        )
        self.load_surface_button.grid(row=0, column=0, padx=10, pady=10, sticky='w')
        
        self.load_orientation_button = ttk.Button(
            self.rbf_frame, text="Load Orientation Points", style='TButton', command=self.load_orientation_points
        )
        self.load_orientation_button.grid(row=0, column=1, padx=10, pady=10, sticky='w')
        
        self.load_orientations_button = ttk.Button(
            self.rbf_frame, text="Load Orientations", style='TButton', command=self.load_orientations
        )
        self.load_orientations_button.grid(row=0, column=2, padx=10, pady=10, sticky='w')
        
        # Row 1: Sampling method and number of surface points
        tk.Label(self.rbf_frame, text="Sampling method:", font=self.label_font).grid(
            row=1, column=2, padx=10, pady=5, sticky='w'
        )
        self.Sampling_method = ttk.Combobox(
            self.rbf_frame, font=self.entry_font,
            values=['Random', 'Systematic', 'Statistic(K-means)']
        )
        self.Sampling_method.grid(row=1, column=3, padx=10, pady=5, sticky='w')
        
        tk.Label(self.rbf_frame, text="Number of Sample Surface Points:", font=self.label_font).grid(
            row=1, column=0, padx=10, pady=5, sticky='w'
        )
        self.num_surface_points = tk.Entry(self.rbf_frame, font=self.entry_font)
        self.num_surface_points.grid(row=1, column=1, padx=10, pady=5, sticky='w')
        
        # Row 2: Number of sample orientation points
        tk.Label(self.rbf_frame, text="Number of Sample Orientation Points:", font=self.label_font).grid(
            row=2, column=0, padx=10, pady=5, sticky='w'
        )
        self.num_orientation_points = tk.Entry(self.rbf_frame, font=self.entry_font)
        self.num_orientation_points.grid(row=2, column=1, padx=10, pady=5, sticky='w')
        
        # Row 3: Kernel function selection
        tk.Label(self.rbf_frame, text="Kernel Function:", font=self.label_font).grid(
            row=3, column=0, padx=10, pady=5, sticky='w'
        )
        self.kernel_function = ttk.Combobox(
            self.rbf_frame, font=self.entry_font,
            values=['cubic', 'gaussian', 'quintic', 'cubic_covariance', 'multiquadric', 
                    'linear', 'thin_plate', 'inverse_quadratic', 'inverse_multiquadric', 
                    'gaussian_covariance', 'ga_cub_cov', 'ga_cub']
        )
        self.kernel_function.grid(row=3, column=1, padx=10, pady=5, sticky='w')
        
        # Row 4: Drift type, load drift scalar, and scaling factor
        tk.Label(self.rbf_frame, text="Drift Type:", font=self.label_font).grid(
            row=4, column=0, padx=10, pady=5, sticky='w'
        )
        self.drift_type = ttk.Combobox(
            self.rbf_frame, font=self.entry_font,
            values=['None', 'First Order Polynomial', 'Second Order Polynomial', 'Ellipsoid', 'Dome Shaped', '2D Custom', '3D Custom']
        )
        self.drift_type.grid(row=4, column=1, padx=10, pady=5, sticky='w')
        
        self.load_drift_scalar_button = ttk.Button(
            self.rbf_frame, text="Load Drift Scalar Field", style='TButton', command=self.load_drift_scalar
        )
        self.load_drift_scalar_button.grid(row=4, column=2, padx=10, pady=10, sticky='w')
        
        tk.Label(self.rbf_frame, text="Drift scaling factor:", font=self.label_font).grid(
            row=4, column=3, padx=10, pady=5, sticky='w'
        )
        self.scaling_factor = tk.Entry(self.rbf_frame, font=self.entry_font)
        self.scaling_factor.grid(row=4, column=4, padx=10, pady=5, sticky='w')
        
        # Row 5: Epsilon for gaussian kernel
        tk.Label(self.rbf_frame, text="Epsilon for gaussian kernel (default is calculated):", font=self.label_font).grid(
            row=5, column=0, padx=10, pady=5, sticky='w'
        )
        self.epsilon_entry = tk.Entry(self.rbf_frame, font=self.entry_font)
        self.epsilon_entry.grid(row=5, column=1, padx=10, pady=5, sticky='w')
        
        # Row 6: Range for covariance kernel
        tk.Label(self.rbf_frame, text="Range for covariance kernel (default is calculated):", font=self.label_font).grid(
            row=6, column=0, padx=10, pady=5, sticky='w'
        )
        self.range_entry = tk.Entry(self.rbf_frame, font=self.entry_font)
        self.range_entry.grid(row=6, column=1, padx=10, pady=5, sticky='w')
        
        # Row 7, 8, 9: Drift parameters a, b, and c
        tk.Label(self.rbf_frame, text="Parameter a (default: 1):", font=self.label_font).grid(
            row=7, column=0, padx=10, pady=5, sticky='w'
        )
        self.param_a_entry = tk.Entry(self.rbf_frame, font=self.entry_font)
        self.param_a_entry.grid(row=7, column=1, padx=10, pady=5, sticky='w')
        
        tk.Label(self.rbf_frame, text="Parameter b (default: 1):", font=self.label_font).grid(
            row=8, column=0, padx=10, pady=5, sticky='w'
        )
        self.param_b_entry = tk.Entry(self.rbf_frame, font=self.entry_font)
        self.param_b_entry.grid(row=8, column=1, padx=10, pady=5, sticky='w')
        
        tk.Label(self.rbf_frame, text="Parameter c (default: 1):", font=self.label_font).grid(
            row=9, column=0, padx=10, pady=5, sticky='w'
        )
        self.param_c_entry = tk.Entry(self.rbf_frame, font=self.entry_font)
        self.param_c_entry.grid(row=9, column=1, padx=10, pady=5, sticky='w')
        
        # Row 10: Compute button
        self.compute_rbf_button = ttk.Button(
            self.rbf_frame, text="Compute RBF Interpolation", style='TButton', command=self.compute_rbf_interpolation
        )
        self.compute_rbf_button.grid(row=10, column=0, padx=10, pady=10, sticky='w')
        
        # Row 11: Checkbox for showing drift contribution
        self.show_drift = tk.Checkbutton(
            self.rbf_frame, text="Show drift contribution", variable=self.show_drift_var, onvalue=1, offvalue=0, font=self.label_font
        )
        self.show_drift.grid(row=11, column=0, padx=10, pady=10, sticky='w')
        
        # Row 12: Checkbox for showing only data (no additional drift)
        self.show_data = tk.Checkbutton(
            self.rbf_frame, text="Only show data", variable=self.only_show_data, onvalue=1, offvalue=0, font=self.label_font
        )
        self.show_data.grid(row=12, column=0, padx=10, pady=10, sticky='w')
        
        # Row 13: Model color selection
        tk.Label(self.rbf_frame, text="Model color:", font=self.label_font).grid(
            row=13, column=0, padx=10, pady=5, sticky='w'
        )
        self.model_color = ttk.Combobox(
            self.rbf_frame, font=self.entry_font,
            values=['red', 'green', 'blue', 'orange', 'purple', 'black', 'darkgrey']
        )
        self.model_color.grid(row=13, column=1, padx=10, pady=5, sticky='w')

    def compare_scalars(self):
        """
        Compare two or more scalar fields by computing gradient differences and then
        displaying statistical metrics and contour plots.
        """
        if self.scalar1 is None or self.scalar2 is None:
            messagebox.showerror("Error", "Please load both scalar fields before comparison.")
            return

        scalar1 = self.scalar1
        scalar2 = self.scalar2
        scalar3 = getattr(self, 'scalar3', None)
        scalar4 = getattr(self, 'scalar4', None)
        scalar5 = getattr(self, 'scalar5', None)

        if self.use_flip.get():
            flipped_field1 = np.zeros_like(scalar1)
            for i in range(scalar1.shape[1]):
                flipped_field1[:, i, :] = np.fliplr(scalar1[:, i, :])
            scalar1 = flipped_field1
        if self.use_reverse.get():
            scalar1 = -scalar1

        axis = self.compare_plane.get()
        mid_index = {'x': scalar1.shape[0] // 2,
                    'y': scalar1.shape[1] // 2,
                    'z': scalar1.shape[2] // 2}[axis]

        self.scalar_comparison(scalar1, scalar2, scalar3, scalar4, scalar5, axis, mid_index)

    def load_scalar1(self):
        """Load Scalar Field 1 from a .npy file."""
        file_path = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")])
        if file_path:
            self.scalar1 = np.load(file_path, allow_pickle=True)
            messagebox.showinfo("Load Scalar Field 1", "Scalar Field 1 loaded successfully!")

    def load_scalar2(self):
        """Load Scalar Field 2 from a .npy file."""
        file_path = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")])
        if file_path:
            self.scalar2 = np.load(file_path, allow_pickle=True)
            messagebox.showinfo("Load Scalar Field 2", "Scalar Field 2 loaded successfully!")

    def load_scalar3(self):
        """Load Scalar Field 3 from a .npy file."""
        file_path = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")])
        if file_path:
            self.scalar3 = np.load(file_path, allow_pickle=True)
            messagebox.showinfo("Load Scalar Field 3", "Scalar Field 3 loaded successfully!")
    
    def load_scalar4(self):
        """Load Scalar Field 4 from a .npy file."""
        file_path = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")])
        if file_path:
            self.scalar4 = np.load(file_path, allow_pickle=True)
            messagebox.showinfo("Load Scalar Field 4", "Scalar Field 4 loaded successfully!")

    def load_scalar5(self):
        """Load Scalar Field 5 from a .npy file."""
        file_path = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")])
        if file_path:
            self.scalar5 = np.load(file_path, allow_pickle=True)
            messagebox.showinfo("Load Scalar Field 5", "Scalar Field 5 loaded successfully!")

    def load_drift_scalar(self):
        """
        Load a drift scalar field from a .npy file.
        """
        file_path = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")])
        if file_path:
            import numpy as np
            self.drift_scalar = np.load(file_path)
            messagebox.showinfo("Load Drift Scalar Field", "Drift Scalar Field loaded successfully!")

    def run_cross_validation(self):
        """
        Run cross-validation to determine the best RBF interpolation parameters.
        """
        import numpy as np
        from rbf_utils import evaluate_params_kfold, systematic_sampling, kmeans_sampling_indices
        def squared_euclidean_distance(x_1, x_2):
            sqd = np.sqrt(np.reshape(np.sum(x_1**2, 1), newshape=(x_1.shape[0], 1)) +
                        np.reshape(np.sum(x_2**2, 1), newshape=(1, x_2.shape[0])) -
                        2 * (x_1 @ x_2.T))
            return np.nan_to_num(sqd)
        num_surface_points = int(self.num_surface_points_cv.get())
        dis = squared_euclidean_distance(
                systematic_sampling(self.cv_vertices_df, num_surface_points)[['X','Y','Z']].values,
                systematic_sampling(self.cv_vertices_df, num_surface_points)[['X','Y','Z']].values)
        eps = dis.mean()
        range_val = dis.max() * 2
        try:
            # Retrieve parameter ranges from entry widgets
            a_start = float(self.param_a_start_cv.get()) if self.param_a_start_cv.get() else 1.0
            a_stop = float(self.param_a_stop_cv.get()) if self.param_a_stop_cv.get() else 2.0
            a_step = float(self.param_a_step_cv.get()) if self.param_a_step_cv.get() else 1.0

            b_start = float(self.param_b_start_cv.get()) if self.param_b_start_cv.get() else 1.0
            b_stop = float(self.param_b_stop_cv.get()) if self.param_b_stop_cv.get() else 2.0
            b_step = float(self.param_b_step_cv.get()) if self.param_b_step_cv.get() else 1.0

            c_start = float(self.param_c_start_cv.get()) if self.param_c_start_cv.get() else 1.0
            c_stop = float(self.param_c_stop_cv.get()) if self.param_c_stop_cv.get() else 2.0
            c_step = float(self.param_c_step_cv.get()) if self.param_c_step_cv.get() else 1.0

            eps_start = float(self.epsilon_start_cv.get()) if self.epsilon_start_cv.get() else eps
            eps_stop = float(self.epsilon_stop_cv.get()) if self.epsilon_stop_cv.get() else eps*2
            eps_step = float(self.epsilon_step_cv.get()) if self.epsilon_step_cv.get() else eps

            range_start = float(self.range_start_cv.get()) if self.range_start_cv.get() else range_val
            range_stop = float(self.range_stop_cv.get()) if self.range_stop_cv.get() else range_val*2
            range_step = float(self.range_step_cv.get()) if self.range_step_cv.get() else range_val

            import numpy as np, itertools, time
            a_values = np.arange(a_start, a_stop, a_step)
            b_values = np.arange(b_start, b_stop, b_step)
            c_values = np.arange(c_start, c_stop, c_step)
            eps_values = np.arange(eps_start, eps_stop, eps_step)
            range_values = np.arange(range_start, range_stop, range_step)

            param_combinations = list(itertools.product(a_values, b_values, c_values, eps_values, range_values))
            total_combinations = len(param_combinations)

            best_error = np.inf
            best_params = None

            self.progress['maximum'] = total_combinations
            self.progress['value'] = 0
            start_time = time.time()

            num_surface_points = int(self.num_surface_points_cv.get())
            num_orientation_points = int(self.num_orientation_points_cv.get())
            random_state = 42
            surface_sampled_indices = kmeans_sampling_indices(self.cv_vertices_df, num_surface_points, random_state)
            if self.Sampling_method_cv.get() == 'Statistic(K-means)':
                layer_position = self.cv_vertices_df.iloc[surface_sampled_indices][['X', 'Y', 'Z']].values
                G_position = layer_position[:num_orientation_points]
                G_orientation = self.cv_normals_df.iloc[surface_sampled_indices][['x', 'y', 'z']].values[:num_orientation_points]
            elif self.Sampling_method_cv.get() == 'Random':
                layer_position = self.cv_vertices_df.sample(n=num_surface_points, random_state=random_state)[['X', 'Y', 'Z']].values
                G_position = self.cv_vertices_df.sample(n=num_orientation_points, random_state=random_state)[['X', 'Y', 'Z']].values
                G_orientation = self.cv_normals_df.sample(n=num_orientation_points, random_state=random_state)[['x', 'y', 'z']].values
            elif self.Sampling_method_cv.get() == 'Systematic':
                layer_position = systematic_sampling(self.cv_vertices_df, num_surface_points)[['X', 'Y', 'Z']].values
                G_position = systematic_sampling(self.cv_vertices_df, num_orientation_points)[['X', 'Y', 'Z']].values
                G_orientation = systematic_sampling(self.cv_normals_df, num_orientation_points)[['x', 'y', 'z']].values

            drift_type_map = {'None': 0, 'Second Order Polynomial': 1, 'Ellipsoid': 2, 'Dome Shaped': 3, 'First Order Polynomial': 4, '2D Custom': 5, '3D Custom': 6}
            drift_type = drift_type_map.get(self.drift_type_cv.get(), 0)
            kernel_function_cv = self.kernel_function_cv.get()
            drift_scalar_cv = self.drift_scalar if hasattr(self, 'drift_scalar') else None
            scaling_factor_cv_ = self.scaling_factor_cv.get() 
            for idx, (a, b, c, eps_val, range_val_param) in enumerate(param_combinations):
                error, params = evaluate_params_kfold(a, b, c, eps_val, range_val_param, G_position, G_orientation, layer_position, drift_type, n_splits=5, kernel_name=kernel_function_cv,drift_scalar=drift_scalar_cv,scaling_factor = scaling_factor_cv_)
                if error < best_error:
                    best_error = error
                    best_params = params

                self.progress['value'] = idx + 1
                elapsed_time = time.time() - start_time
                estimated_total_time = (elapsed_time / (idx + 1)) * total_combinations
                remaining_time = estimated_total_time - elapsed_time
                self.elapsed_time_label.config(text=f"time used: {int(elapsed_time)}s")
                self.remaining_time_label.config(text=f"estimated remaining time: {int(remaining_time)}s")
                self.root.update_idletasks()

            if best_params is not None:
                self.result_label.config(
                    text=f'Best parameters: eps={best_params[3]:.2f}, range={best_params[4]:.2f}, a={best_params[0]:.2f}, b={best_params[1]:.2f}, c={best_params[2]:.2f}\nLeast error: {best_error:.2f} %'
                )
            else:
                self.result_label.config(text="Can't find best parameters.")
        except Exception as e:
            import traceback
            error_message = traceback.format_exc()
            messagebox.showerror("Error", f"Error: {e}\n\n{error_message}")

    def compute_rbf_interpolation(self):
        """
        Compute RBF interpolation using loaded surface and orientation points,
        plot the result, and optionally save the resulting mesh and scalar field.
        """
        import numpy as np
        from rbf_utils import RBF_3D_kernel, systematic_sampling, kmeans_sampling_indices
        try:
            num_surface_points = int(self.num_surface_points.get())
            num_orientation_points = int(self.num_orientation_points.get())
            kernel_function = self.kernel_function.get()
            model_color = self.model_color.get()
            drift_type_str = self.drift_type.get()

            def squared_euclidean_distance(x_1, x_2):
                sqd = np.sqrt(np.reshape(np.sum(x_1**2, 1), newshape=(x_1.shape[0], 1)) +
                            np.reshape(np.sum(x_2**2, 1), newshape=(1, x_2.shape[0])) -
                            2 * (x_1 @ x_2.T))
                return np.nan_to_num(sqd)
            dis = squared_euclidean_distance(systematic_sampling(self.vertices_df, num_surface_points)[['X','Y','Z']].values,
                                            systematic_sampling(self.vertices_df, num_surface_points)[['X','Y','Z']].values)
            default_eps = dis.mean()
            default_range = dis.max() * 2
            default_a = 1
            default_b = 1
            default_c = 1

            eps = float(self.epsilon_entry.get()) if self.epsilon_entry.get() else default_eps
            range_val = float(self.range_entry.get()) if self.range_entry.get() else default_range
            a_val = float(self.param_a_entry.get()) if self.param_a_entry.get() else default_a
            b_val = float(self.param_b_entry.get()) if self.param_b_entry.get() else default_b
            c_val = float(self.param_c_entry.get()) if self.param_c_entry.get() else default_c

            drift_type_map = {'None': 0, 'Second Order Polynomial': 1, 'Ellipsoid': 2, 'Dome Shaped': 3, 'First Order Polynomial': 4, '2D Custom': 5, '3D Custom': 6}
            drift_type = drift_type_map.get(drift_type_str, 0)

            random_state = 24
            if self.Sampling_method.get() == 'Statistic(K-means)':
                layer_position = self.vertices_df.iloc[kmeans_sampling_indices(self.vertices_df, num_surface_points, random_state)][['X', 'Y', 'Z']].values
                G_position = layer_position[:num_orientation_points]
                G_orientation = self.normals_o_df.iloc[kmeans_sampling_indices(self.vertices_df, num_surface_points, random_state)][['x', 'y', 'z']].values[:num_orientation_points]
            elif self.Sampling_method.get() == 'Random':
                layer_position = self.vertices_df.sample(n=num_surface_points, random_state=random_state)[['X', 'Y', 'Z']].values
                G_position = self.vertices_df.sample(n=num_orientation_points, random_state=random_state)[['X', 'Y', 'Z']].values
                G_orientation = self.normals_o_df.sample(n=num_orientation_points, random_state=random_state)[['x', 'y', 'z']].values
            elif self.Sampling_method.get() == 'Systematic':
                layer_position = systematic_sampling(self.vertices_df, num_surface_points)[['X', 'Y', 'Z']].values
                G_position = systematic_sampling(self.vertices_df, num_orientation_points)[['X', 'Y', 'Z']].values
                G_orientation = systematic_sampling(self.normals_o_df, num_orientation_points)[['x', 'y', 'z']].values

            import time
            start_time = time.time()
            show_drift_value = self.show_drift_var.get()
            
            scaling_factor = self.scaling_factor.get()
            drift_scalar = self.drift_scalar if hasattr(self, 'drift_scalar') else None
            intp, mesh = RBF_3D_kernel(G_position=G_position,
                                    G_orientation=G_orientation,
                                    layer_position=layer_position,
                                    test_data=None,
                                    eps=eps,
                                    range_val=range_val,
                                    drift=drift_type,
                                    cv=False,
                                    a=a_val,
                                    b=b_val,
                                    c=c_val,
                                    kernel_name=kernel_function,
                                    show_drift=show_drift_value,
                                    drift_scalar=drift_scalar,
                                    scaling_factor=scaling_factor)
            cost_time = time.time() - start_time
            print('cost time:', cost_time)

            def plot():
                import pyvista as pv
                plotter = pv.Plotter(notebook=False)
                if self.only_show_data.get() == 0:
                    plotter.add_mesh(mesh, show_scalar_bar=False, color=model_color, opacity=0.8)
                plotter.add_points(layer_position[:, [0, 1, 2]], render_points_as_spheres=True, point_size=14.0, color='blue')
                plotter.add_arrows(G_position[:, [0, 1, 2]], direction=G_orientation[:, [0, 1, 2]], color='black', mag=50)
                plotter.set_background('white')
                plotter.show()
            import threading
            threading.Thread(target=plot).start()

            file_path = filedialog.asksaveasfilename(
                initialfile=(self.model_name + "_" +
                            self.Sampling_method.get() + "_" +
                            self.num_surface_points.get() + "_" +
                            self.num_orientation_points.get() + "_" +
                            self.kernel_function.get() + "_" +
                            self.drift_type.get() + "_" +
                            self.epsilon_entry.get() + "_" +
                            self.range_entry.get() + "_" +
                            self.param_a_entry.get() + "_" +
                            self.param_b_entry.get() + "_" +
                            self.param_c_entry.get() + ".vtk"),
                defaultextension=".vtk",
                filetypes=[("VTK files", "*.vtk")]
            )
            if file_path:
                file_path_scalar = file_path.replace(".vtk", "_scalar.npy")
                mesh.save(file_path)
                import numpy as np
                np.save(file_path_scalar, vtk_to_scalar(mesh))
                messagebox.showinfo("Save Mesh and Compute Scalar Field", "Mesh and Scalar Field saved successfully!")
        except Exception as e:
            import traceback
            error_message = traceback.format_exc()
            messagebox.showerror("Error", f"An error occurred: {e}\n\n{error_message}")

    def scalar_comparison(self, scalar1, scalar2, scalar3=None, scalar4=None, scalar5=None, axis=None, index=None):
        """
        Compare several 3D scalar fields by computing their gradients, evaluating the cosine
        differences between the gradient fields, and then plotting contour and quiver plots
        along a specified slice.
        
        This method handles up to five fields:
        - scalar1 is taken as the synthetic field.
        - scalar2 is the primary interpolated field.
        - scalar3, scalar4, scalar5 (if provided) are additional interpolated fields.
        
        For each pair (A vs. B, A vs. C, etc.) the gradients are computed and the cosine similarity
        is determined (with difference = 1 - cosine similarity). Then, for each comparison, two helper
        functions are used:
        
            plot_scalar_field_slice(field, axis, index, title)
                -> extracts a 2D slice along the specified axis and returns a contour plot.
            
            plot_two_gradient_fields(field1, field2, diff, axis, index, title, skip)
                -> down-samples the gradient fields, normalizes them, and overlays quiver arrows
                on a contour of the difference field.
        
        Finally, summary statistics (mean, RMS, maximum, and median differences) are computed and
        displayed in a statistics panel, and all plots are embedded in the Tkinter canvas (self.scalar_canvas_frame).
        
        Parameters:
        scalar1 (ndarray): Synthetic scalar field (3D array).
        scalar2 (ndarray): Interpolated scalar field (3D array).
        scalar3, scalar4, scalar5 (ndarray, optional): Additional scalar fields.
        axis (str): Axis along which to take the slice ('x', 'y', or 'z').
        index (int): Slice index along the given axis.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import cmcrameri as cmc
        # --- Helper Functions ---
        def plot_two_gradient_fields(field1, field2, diff, axis, index, title, skip=5):
            # Compute gradients for both fields.
            grad_field1 = np.gradient(field1)
            grad_field2 = np.gradient(field2)
            if axis == 'x':
                grad_slice1 = [g[index, :, :] for g in grad_field1]
                grad_slice2 = [g[index, :, :] for g in grad_field2]
                diff_slice = diff[index, :, :]
            elif axis == 'y':
                grad_slice1 = [g[:, index, :] for g in grad_field1]
                grad_slice2 = [g[:, index, :] for g in grad_field2]
                diff_slice = diff[:, index, :]
            elif axis == 'z':
                grad_slice1 = [g[:, :, index] for g in grad_field1]
                grad_slice2 = [g[:, :, index] for g in grad_field2]
                diff_slice = diff[:, :, index]
            else:
                raise ValueError("Axis must be 'x', 'y', or 'z'.")
        
            # Downsample the gradients.
            fig, ax = plt.subplots(figsize=(3, 2.5)) 
            grad_x_1 = grad_slice1[0][::skip, ::skip]
            grad_y_1 = grad_slice1[1][::skip, ::skip]
            magnitude1 = np.sqrt(grad_x_1**2 + grad_y_1**2)
        
            grad_x_2 = grad_slice2[0][::skip, ::skip]
            grad_y_2 = grad_slice2[1][::skip, ::skip]
            magnitude2 = np.sqrt(grad_x_2**2 + grad_y_2**2)
        
            eps_val = 1e-10
            grad_x_norm_1 = grad_x_1 / (magnitude1 + eps_val)
            grad_y_norm_1 = grad_y_1 / (magnitude1 + eps_val)
            grad_x_norm_2 = grad_x_2 / (magnitude2 + eps_val)
            grad_y_norm_2 = grad_y_2 / (magnitude2 + eps_val)
        
            fixed_arrow_length = 6
            # Use the shape of the downsampled arrays for the meshgrid.
            contour = ax.contourf(diff_slice, levels=30, cmap='cmc.grayC_r')
            X, Y = np.meshgrid(np.arange(0, field1.shape[1], skip), np.arange(0, field1.shape[2], skip))
            ax.quiver(X, Y, grad_x_norm_1 * fixed_arrow_length, grad_y_norm_1 * fixed_arrow_length,
                        angles='xy', scale_units='xy', scale=1, color="orange", width=0.003, label='Synthetic Model')
            ax.quiver(X, Y, grad_x_norm_2 * fixed_arrow_length, grad_y_norm_2 * fixed_arrow_length,
                        angles='xy', scale_units='xy', scale=1, color="blue", width=0.003, label='Interpolated Model')
            legend = plt.legend(frameon=True,fontsize=8)
            frame = legend.get_frame()
            frame.set_edgecolor('black')
            ax.set_title(f'{title} (Slice {axis}={index})', fontsize=8)
            ax.set_xticklabels([])  
            ax.set_yticklabels([]) 

            return fig

        def plot_scalar_field_slice(field, axis, index, title):
            if axis == 'x':
                slice_data = field[index, :, :]
            elif axis == 'y':
                slice_data = field[:, index, :]
            elif axis == 'z':
                slice_data = field[:, :, index]
            else:
                raise ValueError("Axis must be 'x', 'y', or 'z'.")
            fig, ax = plt.subplots(figsize=(3, 2.5))
            ax.contourf(slice_data, levels=20, cmap='cmc.batlow')
            ax.contour(slice_data, levels=[0], colors='black')
            ax.set_title(f'{title} (Slice {axis}={index})', fontsize=8)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            return fig

        # --- Main Comparison Code ---
        # Define field variables.
        f_A = scalar1
        f_B = scalar2
        f_C = scalar3 if scalar3 is not None else None
        f_A = scalar1
        f_B = scalar2
        f_C = scalar3 if scalar3 is not None else None
        f_D = scalar4 if scalar4 is not None else None
        f_E = scalar5 if scalar5 is not None else None

        # Compute gradient fields.
        grad_f_A = np.gradient(f_A)
        grad_f_B = np.gradient(f_B)
        grad_f_C = np.gradient(f_C) if f_C is not None else None
        grad_f_D = np.gradient(f_D) if f_D is not None else None
        grad_f_E = np.gradient(f_E) if f_E is not None else None

        # Initialize dot products and magnitudes.
        dot_product_AB = np.zeros(f_A.shape)
        dot_product_AC = np.zeros(f_A.shape) if f_C is not None else None
        dot_product_AD = np.zeros(f_A.shape) if f_D is not None else None
        dot_product_AE = np.zeros(f_A.shape) if f_E is not None else None
        magnitude_A = np.zeros(f_A.shape)
        magnitude_B = np.zeros(f_A.shape)
        magnitude_C = np.zeros(f_A.shape) if f_C is not None else None
        magnitude_D = np.zeros(f_A.shape) if f_D is not None else None
        magnitude_E = np.zeros(f_A.shape) if f_E is not None else None

        for i in range(3):
            dot_product_AB += grad_f_A[i] * grad_f_B[i]
            magnitude_A += grad_f_A[i] ** 2
            magnitude_B += grad_f_B[i] ** 2
            if f_C is not None:
                dot_product_AC += grad_f_A[i] * grad_f_C[i]
                magnitude_C += grad_f_C[i] ** 2 
            if f_D is not None:
                dot_product_AD += grad_f_A[i] * grad_f_D[i]
                magnitude_D += grad_f_D[i] ** 2
            if f_E is not None:
                dot_product_AE += grad_f_A[i] * grad_f_E[i]
                magnitude_E += grad_f_E[i] ** 2


        magnitude_A = np.sqrt(magnitude_A)
        magnitude_B = np.sqrt(magnitude_B)
        magnitude_C = np.sqrt(magnitude_C) if f_C is not None else None
        magnitude_D = np.sqrt(magnitude_D) if f_D is not None else None
        magnitude_E = np.sqrt(magnitude_E) if f_E is not None else None

        epsilon = 1e-10
        magnitude_A = np.where(magnitude_A == 0, epsilon, magnitude_A)
        magnitude_B = np.where(magnitude_B == 0, epsilon, magnitude_B)
        cosine_similarity_AB = dot_product_AB / (magnitude_A * magnitude_B)
        d_AB_nabla = 1 - cosine_similarity_AB
        if f_C is not None:
            magnitude_C = np.where(magnitude_C == 0, epsilon, magnitude_C)
            cosine_similarity_AC = dot_product_AC / (magnitude_A * magnitude_C)
            d_AC_nabla = 1 - cosine_similarity_AC
        if f_D is not None:
            magnitude_D = np.where(magnitude_D == 0, epsilon, magnitude_D)
            cosine_similarity_AD = dot_product_AD / (magnitude_A * magnitude_D)
            d_AD_nabla = 1 - cosine_similarity_AD
        if f_E is not None:
            magnitude_E = np.where(magnitude_E == 0, epsilon, magnitude_E)
            cosine_similarity_AE = dot_product_AE / (magnitude_A * magnitude_E)
            d_AE_nabla = 1 - cosine_similarity_AE

        # Create figures for each field.
        figA = plot_scalar_field_slice(f_A, axis, index, 'Synthetic Scalar Field')
        figB = plot_scalar_field_slice(f_B, axis, index, 'Interpolated Scalar Field')
        figAB = plot_two_gradient_fields(f_A, f_B, d_AB_nabla, axis, index, 'Difference Plot (A vs B)', skip=7)
        if f_C is not None:
            figC = plot_scalar_field_slice(f_C, axis, index, 'Interpolated Scalar Field (C)')
            figAC = plot_two_gradient_fields(f_A, f_C, d_AC_nabla, axis, index, 'Difference Plot (A vs C)', skip=7)
        if f_D is not None:
            figD = plot_scalar_field_slice(f_D, axis, index, 'Interpolated Scalar Field (D)')
            figAD = plot_two_gradient_fields(f_A, f_D, d_AD_nabla, axis, index, 'Difference Plot (A vs D)', skip=7)
        if f_E is not None:
            figE = plot_scalar_field_slice(f_E, axis, index, 'Interpolated Scalar Field (E)')
            figAE = plot_two_gradient_fields(f_A, f_E, d_AE_nabla, axis, index, 'Difference Plot (A vs E)', skip=7)

        # Compute and display statistics (for comparison A vs. B only, as an example).
        mean_diff_AB = np.mean(d_AB_nabla)
        rms_diff_AB = np.sqrt(np.mean(d_AB_nabla**2))
        max_diff_AB = np.max(np.abs(d_AB_nabla))
        median_diff_AB = np.median(d_AB_nabla)

        if f_C is not None:
            mean_diff_AC = np.mean(d_AC_nabla)
            rms_diff_AC = np.sqrt(np.mean(d_AC_nabla**2))
            max_diff_AC = np.max(np.abs(d_AC_nabla))
            median_diff_AC = np.median(d_AC_nabla)
        if f_D is not None:
            mean_diff_AD = np.mean(d_AD_nabla)
            rms_diff_AD = np.sqrt(np.mean(d_AD_nabla**2))
            max_diff_AD = np.max(np.abs(d_AD_nabla))
            median_diff_AD = np.median(d_AD_nabla)
        if f_E is not None:
            mean_diff_AE = np.mean(d_AE_nabla)
            rms_diff_AE = np.sqrt(np.mean(d_AE_nabla**2))
            max_diff_AE = np.max(np.abs(d_AE_nabla))
            median_diff_AE = np.median(d_AE_nabla)

        for widget in self.scalar_canvas_frame.winfo_children():
            widget.destroy()

        stats_frame = tk.Frame(self.scalar_canvas_frame)
        stats_frame.grid(row=0, column=1, columnspan=6, pady=5)

        tk.Label(stats_frame, text=f"Mean Difference: {mean_diff_AB:.3f}", font=self.label_font_s).grid(row=0, column=2, sticky='w')
        tk.Label(stats_frame, text=f"RMS Difference: {rms_diff_AB:.3f}", font=self.label_font_s).grid(row=0, column=3, sticky='w')
        tk.Label(stats_frame, text=f"Maximum Difference: {max_diff_AB:.3f}", font=self.label_font_s).grid(row=1, column=2, sticky='w')
        tk.Label(stats_frame, text=f"Median Difference: {median_diff_AB:.3f}", font=self.label_font_s).grid(row=1, column=3, sticky='w')

        if f_C is not None:
            tk.Label(stats_frame, text=f"Mean Difference: {mean_diff_AC:.3f}", font=self.label_font_s).grid(row=0, column=4, sticky='w')
            tk.Label(stats_frame, text=f"RMS Difference: {rms_diff_AC:.3f}", font=self.label_font_s).grid(row=0, column=5, sticky='w')
            tk.Label(stats_frame, text=f"Maximum Difference: {max_diff_AC:.3f}", font=self.label_font_s).grid(row=1, column=4, sticky='w')
            tk.Label(stats_frame, text=f"Median Difference: {median_diff_AC:.3f}", font=self.label_font_s).grid(row=1, column=5, sticky='w')
        
        if f_D is not None:
            tk.Label(stats_frame, text=f"Mean Difference: {mean_diff_AD:.3f}", font=self.label_font_s).grid(row=0, column=6, sticky='w')
            tk.Label(stats_frame, text=f"RMS Difference: {rms_diff_AD:.3f}", font=self.label_font_s).grid(row=0, column=7, sticky='w')
            tk.Label(stats_frame, text=f"Maximum Difference: {max_diff_AD:.3f}", font=self.label_font_s).grid(row=1, column=6, sticky='w')
            tk.Label(stats_frame, text=f"Median Difference: {median_diff_AD:.3f}", font=self.label_font_s).grid(row=1, column=7, sticky='w')

        if f_E is not None:
            tk.Label(stats_frame, text=f"Mean Difference: {mean_diff_AE:.3f}", font=self.label_font_s).grid(row=0, column=8, sticky='w')
            tk.Label(stats_frame, text=f"RMS Difference: {rms_diff_AE:.3f}", font=self.label_font_s).grid(row=0, column=9, sticky='w')
            tk.Label(stats_frame, text=f"Maximum Difference: {max_diff_AE:.3f}", font=self.label_font_s).grid(row=1, column=8, sticky='w')
            tk.Label(stats_frame, text=f"Median Difference: {median_diff_AE:.3f}", font=self.label_font_s).grid(row=1, column=9, sticky='w')

        canvas1 = FigureCanvasTkAgg(figA, master=self.scalar_canvas_frame)
        canvas1.draw()
        canvas1.get_tk_widget().grid(row=1, column=0, padx=5, pady=5)

        canvas2 = FigureCanvasTkAgg(figB, master=self.scalar_canvas_frame)
        canvas2.draw()
        canvas2.get_tk_widget().grid(row=1, column=1, padx=5, pady=5)

        canvas6 = FigureCanvasTkAgg(figAB, master=self.scalar_canvas_frame)
        canvas6.draw()
        canvas6.get_tk_widget().grid(row=2, column=1, padx=5, pady=5)

        if f_C is not None:
            canvas3 = FigureCanvasTkAgg(figC, master=self.scalar_canvas_frame)
            canvas3.draw()
            canvas3.get_tk_widget().grid(row=1, column=2, padx=5, pady=5)
            canvas7 = FigureCanvasTkAgg(figAC, master=self.scalar_canvas_frame)
            canvas7.draw()
            canvas7.get_tk_widget().grid(row=2, column=2, padx=5, pady=5)
        
        if f_D is not None:
            canvas4 = FigureCanvasTkAgg(figD, master=self.scalar_canvas_frame)
            canvas4.draw()
            canvas4.get_tk_widget().grid(row=1, column=3, padx=5, pady=5)
            canvas8 = FigureCanvasTkAgg(figAD, master=self.scalar_canvas_frame)
            canvas8.draw()
            canvas8.get_tk_widget().grid(row=2, column=3, padx=5, pady=5)

        if f_E is not None:
            canvas5 = FigureCanvasTkAgg(figE, master=self.scalar_canvas_frame)
            canvas5.draw()
            canvas5.get_tk_widget().grid(row=1, column=4, padx=5, pady=5)
            canvas9 = FigureCanvasTkAgg(figAE, master=self.scalar_canvas_frame)
            canvas9.draw()
            canvas9.get_tk_widget().grid(row=2, column=4, padx=5, pady=5)


    # ------------------ Methods related to Gravity Simulation ------------------

    def build_gravity_tab(self):
        """
        Build the widgets for the Gravity Simulation tab.
        """
        self.gravity_simulation_frame = tk.Frame(self.tab11)
        self.gravity_simulation_frame.grid(row=0, column=0, padx=10, pady=10, sticky='w')
        self.load_vtk_button = ttk.Button(self.gravity_simulation_frame, text="Load VTK File", style='TButton', command=self.load_vtk_to_gravity)
        self.load_vtk_button.grid(row=0, column=0, padx=10, pady=10, sticky='w')
        self.gravity_fwd_button = ttk.Button(self.gravity_simulation_frame, text="Compute Gravity Forward Scalar Field", style='TButton', command=self.gravity_fwd_2d)
        self.gravity_fwd_button.grid(row=0, column=1, padx=10, pady=10, sticky='w')
        self.gravity_canvas_frame = tk.Frame(self.tab11)
        self.gravity_canvas_frame.grid(row=4, column=0, padx=10, pady=10, columnspan=4)

    def load_vtk_to_gravity(self):
        """
        Load a VTK file for gravity simulation.
        """
        file_path = filedialog.askopenfilename(filetypes=[("VTK files", "*.vtp"), ("VTK files", "*.vtk")])
        if file_path:
            from vtk_utils import load_vtk_mesh  # reimport if needed
            self.mesh = load_vtk_mesh(file_path)
            messagebox.showinfo("VTK Operations", "VTK file loaded successfully!")

    def gravity_fwd_2d(self):
        """
        Compute a 2D gravity forward model using the current mesh.
        This method calls the gravity_fwd_2d function from gravity.py and then plots the result.
        """
        gravity_calculated, a_opt, b_opt, c_opt, d_opt, e_opt, f_opt = gravity_fwd_2d(self.mesh, self.gravity_canvas_frame)
        self.result_label.config(
            text = f'Best gravity parameters: a={a_opt:.2f}, b={b_opt:.2f}, c={c_opt:.2f}, d={d_opt:.2f}, e={e_opt:.2f}, f={f_opt:.2f}'
        )
        return gravity_calculated

    # ------------------ Methods for Data Save/Load ------------------

    def save_data(self):
        """
        Save the current drawing data (layer names, positions, and cross‑section points) to a JSON file.
        """
        data = {
            "name_saved": self.name_saved,
            "x_pos_saved": self.x_pos_saved,
            "y_pos_saved": self.y_pos_saved,
            "x_points": [points.tolist() for points in self.x_points],
            "y_points": [points.tolist() for points in self.y_points]
        }
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "w") as file:
                json.dump(data, file)

    def load_data(self):
        """
        Load drawing data from a JSON file and automatically create the model.
        """
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "r") as file:
                data = json.load(file)
            self.name_saved = data["name_saved"]
            self.x_pos_saved = data["x_pos_saved"]
            self.y_pos_saved = data["y_pos_saved"]
            self.x_points = [np.array(points) for points in data["x_points"]]
            self.y_points = [np.array(points) for points in data["y_points"]]
            messagebox.showinfo("Load Data", "Data loaded successfully!")
            self.create_model()

    # ------------------ Help and Workflow Tabs ------------------

    def build_help_tab(self):
        """
        Build the Help/About tab with descriptive text and images.
        """
        # Load an image and text (adjust paths as needed)
        # Get the directory where the current script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Build the full path to workflow.png
        image_path= os.path.join(script_dir, 'workflow.png')
        image = Image.open(image_path)
        photo = ImageTk.PhotoImage(image)
        text_label = tk.Label(self.tab9, text="This software is designed for kernel based implicit geological modeling.\n"
                         "Functions include:\n"
                         "1. Create Gempy model by drawing.\n"
                         "2. Save and load drawing models, and export GemPy model as .vtp format.\n"
                         "3. Visualize GemPy model in 2D and 3D.\n"
                         "4. Create and save scalar fields from .vtp or .vtk models.\n"
                         "5. Implicit modeling using surface points and orientations with adjustable sampling.\n"
                         "6. Cross-validation for parameter optimization.\n"
                         "7. Scalar field comparison using cosine difference.\n"
                         "Additional explanations for drift parameters are provided here.",
                         font=self.label_font, anchor='w', justify='left', wraplength=1000)
        image_label = tk.Label(self.tab9, image=photo)
        text_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')
        image_label.grid(row=0, column=1, padx=10, pady=10, sticky='e')
        image_label.image = photo  # Keep a reference

    def build_workflow_tab(self):
        """
        Build the Workflow tab with a full-screen background image and several navigation canvases.
        
        This function loads "workguide_2.png", rescales it to fit within 1850x1080 pixels, and
        displays it as the background. It then loads additional images (tab1.png, tab2.png, tab3.png, 
        tab5.png, tab5_2.png, tab6.png, tab7.png, tab8_1.png, tab8_2.png, tab8_3.png) and creates canvases 
        with these images. Clicking on these canvases navigates to the corresponding tabs.
        """
        import os
        from PIL import Image, ImageTk

        # Determine the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Create the workflow tab and add it to the notebook
        self.tab10 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab10, text='Workflow')

        # Load and resize the background image (workguide_2.png)
        workflow_path = os.path.join(script_dir, "workguide_2.png")
        image = Image.open(workflow_path)
        original_width, original_height = image.size
        scale_factor = min(1850 / original_width, 1080 / original_height)
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        image = image.resize((new_width, new_height), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)

        # Create a label to display the background image
        image_label = tk.Label(self.tab10, image=photo)
        image_label.place(x=0, y=0, relwidth=1, relheight=1)
        image_label.image = photo  # Keep a reference to avoid garbage collection

        # Load navigation images from files (adjust paths if necessary)
        tab1_p = ImageTk.PhotoImage(Image.open(os.path.join(script_dir, "tab1.png")).resize((300, 55), Image.LANCZOS))
        tab2_p = ImageTk.PhotoImage(Image.open(os.path.join(script_dir, "tab2.png")).resize((300, 55), Image.LANCZOS))
        tab3_p = ImageTk.PhotoImage(Image.open(os.path.join(script_dir, "tab3.png")).resize((300, 55), Image.LANCZOS))
        tab5_p = ImageTk.PhotoImage(Image.open(os.path.join(script_dir, "tab5.png")).resize((300, 77), Image.LANCZOS))
        tab5_2_p = ImageTk.PhotoImage(Image.open(os.path.join(script_dir, "tab5_2.png")).resize((300, 55), Image.LANCZOS))
        tab6_p = ImageTk.PhotoImage(Image.open(os.path.join(script_dir, "tab6.png")).resize((300, 55), Image.LANCZOS))
        tab7_p = ImageTk.PhotoImage(Image.open(os.path.join(script_dir, "tab7.png")).resize((320, 55), Image.LANCZOS))
        tab8_1_p = ImageTk.PhotoImage(Image.open(os.path.join(script_dir, "tab8_1.png")).resize((300, 55), Image.LANCZOS))
        tab8_2_p = ImageTk.PhotoImage(Image.open(os.path.join(script_dir, "tab8_2.png")).resize((300, 55), Image.LANCZOS))
        tab8_3_p = ImageTk.PhotoImage(Image.open(os.path.join(script_dir, "tab8_3.png")).resize((300, 55), Image.LANCZOS))

        # Create canvases and bind them to navigation commands:
        canvas = tk.Canvas(self.tab10, width=300, height=55, highlightthickness=0)
        canvas.place(relx=0.12, rely=0.47, anchor='center')
        canvas_image = canvas.create_image(0, 0, anchor='nw', image=tab2_p)
        canvas.tag_bind(canvas_image, "<Button-1>", lambda event: self.show_tab(self.tab2, 'Create GemPy Model'))
        canvas.image = tab2_p

        canvas = tk.Canvas(self.tab10, width=300, height=55, highlightthickness=0)
        canvas.place(relx=0.12, rely=0.55, anchor='center')
        canvas_image = canvas.create_image(0, 0, anchor='nw', image=tab3_p)
        canvas.tag_bind(canvas_image, "<Button-1>", lambda event: self.show_tab(self.tab3, 'Drawing'))
        canvas.image = tab3_p

        canvas = tk.Canvas(self.tab10, width=300, height=55, highlightthickness=0)
        canvas.place(relx=0.12, rely=0.63, anchor='center')
        canvas_image = canvas.create_image(0, 0, anchor='nw', image=tab1_p)
        canvas.tag_bind(canvas_image, "<Button-1>", lambda event: self.show_tab(self.tab1, 'Save/Load Data'))
        canvas.image = tab1_p

        canvas = tk.Canvas(self.tab10, width=300, height=70, highlightthickness=0)
        canvas.place(relx=0.395, rely=0.31, anchor='center')
        canvas_image = canvas.create_image(0, 0, anchor='nw', image=tab5_p)
        canvas.tag_bind(canvas_image, "<Button-1>", lambda event: self.show_tab(self.tab5, 'VTK to Scalar Field'))
        canvas.image = tab5_p

        canvas = tk.Canvas(self.tab10, width=300, height=55, highlightthickness=0)
        canvas.place(relx=0.395, rely=0.39, anchor='center')
        canvas_image = canvas.create_image(0, 0, anchor='nw', image=tab5_2_p)
        canvas.tag_bind(canvas_image, "<Button-1>", lambda event: self.show_tab(self.tab5, 'VTK to Scalar Field'))
        canvas.image = tab5_2_p

        canvas = tk.Canvas(self.tab10, width=320, height=55, highlightthickness=0)
        canvas.place(relx=0.65, rely=0.195, anchor='center')
        canvas_image = canvas.create_image(0, 0, anchor='nw', image=tab7_p)
        canvas.tag_bind(canvas_image, "<Button-1>", lambda event: self.show_tab(self.tab7, 'Cross-Validation'))
        canvas.image = tab7_p

        canvas = tk.Canvas(self.tab10, width=320, height=55, highlightthickness=0)
        canvas.place(relx=0.89, rely=0.195, anchor='center')
        canvas_image = canvas.create_image(0, 0, anchor='nw', image=tab7_p)
        canvas.tag_bind(canvas_image, "<Button-1>", lambda event: self.show_tab(self.tab6, 'Implicit Interpolation'))
        canvas.image = tab7_p

        canvas = tk.Canvas(self.tab10, width=300, height=55, highlightthickness=0)
        canvas.place(relx=0.885, rely=0.86, anchor='center')
        canvas_image = canvas.create_image(0, 0, anchor='nw', image=tab5_2_p)
        canvas.tag_bind(canvas_image, "<Button-1>", lambda event: self.show_tab(self.tab5, 'VTK to Scalar Field'))
        canvas.image = tab5_2_p

        canvas = tk.Canvas(self.tab10, width=300, height=55, highlightthickness=0)
        canvas.place(relx=0.395, rely=0.67, anchor='center')
        canvas_image = canvas.create_image(0, 0, anchor='nw', image=tab8_1_p)
        canvas.tag_bind(canvas_image, "<Button-1>", lambda event: self.show_tab(self.tab8, 'Scalar Field Comparison'))
        canvas.image = tab8_1_p

        canvas = tk.Canvas(self.tab10, width=300, height=55, highlightthickness=0)
        canvas.place(relx=0.395, rely=0.75, anchor='center')
        canvas_image = canvas.create_image(0, 0, anchor='nw', image=tab8_2_p)
        canvas.tag_bind(canvas_image, "<Button-1>", lambda event: self.show_tab(self.tab8, 'Scalar Field Comparison'))
        canvas.image = tab8_2_p

        canvas = tk.Canvas(self.tab10, width=300, height=55, highlightthickness=0)
        canvas.place(relx=0.395, rely=0.83, anchor='center')
        canvas_image = canvas.create_image(0, 0, anchor='nw', image=tab8_3_p)
        canvas.tag_bind(canvas_image, "<Button-1>", lambda event: self.show_tab(self.tab8, 'Scalar Field Comparison'))
        canvas.image = tab8_3_p


    # ------------------ Gravity Simulation Related Helper Methods ------------------
    # (Any additional helper methods needed for gravity simulation can be added here.)

    def run(self):
        """Start the Tkinter main event loop."""
        self.root.mainloop()

# If this module is run directly, create a root window and start the application.
if __name__ == "__main__":
    root = tk.Tk()
    app = GeologicalModelApp(root)
    app.run()
