# import modules

import numpy as np
import os
import json
from scipy.interpolate import CubicSpline

import utils

# define Output class

class Output:
    """
    Represents the key variables to output from a given engine system.
    
    Creates a JSON representation of data and prints the file to a text file in a given directory.
    
    Parameters
    ----------
    engine : class
        Class representation of the engine system.
    """
    def __init__(self, engine):
        """Creates an instance of the Output class."""
        # store input variables
        self.blade_rows = []

        # set up array to interpolate span data over
        span_fine = np.linspace(-0.05, 1.05, 23)

        # iterate over all blade rows
        for index, blade_row in enumerate(engine.blade_rows):

            # calculate inlet and outlet dimensionless span datapoints
            span_in = [
                (inlet.r - blade_row.r_hub) / (blade_row.r_casing_inlet - blade_row.r_hub)
                for inlet in blade_row.inlet
            ]
            span_out = [
                (exit.r - blade_row.r_hub) / (blade_row.r_casing_inlet - blade_row.r_hub)
                for exit in blade_row.exit
            ]

            # set up cubic interpolation of span data
            cs_in = CubicSpline(span_in, blade_row.inlet_angle)
            cs_out = CubicSpline(span_out, blade_row.exit_angle)

            # create dictionary of row data and store in class
            row_data = {
                "axial_position": index,
                "inlet_r": [inlet.r for inlet in blade_row.inlet],
                "inlet_span": span_in,
                "inlet_angle": blade_row.inlet_angle,
                "inlet_span_fine": list(span_fine),
                "inlet_angle_fine": list(cs_in(span_fine)),

                "exit_r": [exit.r for exit in blade_row.exit],
                "exit_span": span_out,
                "exit_angle": blade_row.exit_angle,
                "exit_span_fine": list(span_fine),
                "exit_angle_fine": list(cs_out(span_fine))
            }
            self.blade_rows.append(row_data)

        self.save_to_json()

        

    def save_to_json(self):
        """Save the Output data as a JSON file in the same directory."""
        data = self.__dict__

        # Build file path
        file_path = os.path.join(os.path.dirname(__file__), "engine_output.json")

        # Save to JSON
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"✅ Engine data saved to {file_path}")