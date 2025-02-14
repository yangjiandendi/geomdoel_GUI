"""
main.py

This module serves as the main entry point for the Geological Modeling Application.
It creates the main Tkinter window and instantiates the GUI.
"""

from gui import GeologicalModelApp
import tkinter as tk
import draw

def main():
    """Initialize the Tkinter root window and run the application."""
    root = tk.Tk()
    root.title("Interpolation Method Comparison Tool")
    app = GeologicalModelApp(root)
    app.run()

if __name__ == "__main__":
    main()
