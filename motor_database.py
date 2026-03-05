# import modules
import numpy as np
import matplotlib.pyplot as plt
import json
from time import perf_counter as timer
import itertools

# import colours
from utils import Colours

# Inputs class
class Inputs:
    """"""
    folder_path = "../../Motor Database"
    file_names = [
        "NeuMotors 1200 Series BLDC Motor",
        "NeuMotors 3800 Series BLDC Motors",
        "NeuMotors 6500 Series BLDC Motors",
        "NeuMotors 8000 SERIES BLDC MOTORS",
        "NeuMotors 120xx Series BLDC Motors"
    ]

    # default matplotlib parameters
    figsize = (7, 5)
    markersize = 16

# Database class
class Database:
    """Constructs a database of electric motors given a list of input file paths."""
    def __init__(self, file_paths):
        """Creates an instance of the Database class."""
        # store input variables
        self.file_paths = file_paths

        # create cycle of colours
        self.colour_cycle = itertools.cycle(plt.cm.tab10.colors)

        # create empty list of motors and read in the input data
        self.motors = []
        self.load_files()
    
    def __str__(self):
        """Prints a representation of the class."""
        # create empty string to return
        string = ""

        # print number of motors
        string += f"Database contains {Colours.GREEN}{len(self.motors)}{Colours.END} motors!\n"

        # print keys of first motor
        string += f"Available motor keys:\n{[key for key, value in self.motors[0].items()]}\n"

        # print database reading duration
        string += f"Input data read in {Colours.GREEN}{self.time:.4g}{Colours.END} s."

        return string

    def load_files(self):
        """Loads in the data from all of the given file paths."""
        # start timer
        t1 = timer()

        # loop for each file
        for file_path in self.file_paths:

            # try-except block
            try:

                # parse file
                self.parse_file(file_path)

            # catch exceptions
            except Exception as error:

                print(
                    f"{Colours.RED}File parse error in file{Colours.END} {file_path} {Colours.RED}"
                    f"at line{Colours.END} {self.index}"
                )
                print(f"{error}")

        # stop timer and store time taken
        t2 = timer()
        self.time = t2 - t1
    
    def parse_file(self, file_path):
        """Parses a given input datafile."""
        # preallocate variables to store variables shared between all database entries
        dictionary = ""
        shared_vars = {"colour": next(self.colour_cycle)}
        count = 0
        
        # open file
        with open(file_path) as file:

            # loop line-by-line through file
            for index, line in enumerate(file):

                # store line position
                self.index = index
                
                # line is empty
                if not line:

                    # skip line
                    continue
                
                # line starts with "{"
                if "{" in line or "}" in line:

                    # update bracket count
                    count += line.count("{") - line.count("}")

                    # store contents of line
                    dictionary += line.strip()

                    # if count now totals zero
                    if count == 0:

                        # read dictionary and update shared variables
                        dict_data = json.loads(dictionary)

                        # loop over key-value pairs in dictionary
                        for key, value in dict_data.items():

                            # try-except block
                            try:

                                # convert to float
                                dict_data[key] = float(value.replace(",", ""))

                            # catch ValueError e.g. data is not a number
                            except ValueError:

                                # do nothing
                                pass

                            # catch AttributeError e.g. data is a list
                            except AttributeError:

                                # do nothing
                                pass

                        # update shared variables
                        shared_vars.update(dict_data)

                        # reset dictionary
                        dictionary = ""
                
                # Check if line is data (starts with a motor identifier or number)
                elif count == 0:

                    # split by whitespace
                    values = line.split()
                    
                    # Create motor dict from headings and values
                    motor = {}
                    for index, heading in enumerate(shared_vars["headings"]):

                        # first heading
                        if index == 0:

                            # store variable
                            motor[heading] = values[index]

                        # all other headings
                        else:

                            # convert to float and store variable
                            motor[heading] = float(values[index].replace(",", ""))
                    
                    # Add all shared variables to this motor
                    motor.update(shared_vars)
                    
                    # Add to database
                    self.motors.append(motor)

                else:

                    # store contents of line
                    dictionary += line.strip()

    def scatter_plot(self, x_key, y_key, x_label = "", y_label = ""):
        """Produces a scatter plot of the given variables."""
        # create plot
        fig, ax = plt.subplots(figsize = Inputs.figsize)

        # extract x- and y-data
        xx = [motor[x_key] for motor in self.motors]
        yy = [motor[y_key] for motor in self.motors]
        colours = [motor["colour"] for motor in self.motors]

        # configure plot
        ax.grid()
        ax.set_axisbelow(True)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # plot data
        ax.scatter(
            xx, yy, c = colours, marker = '.', s = Inputs.markersize,
            label = "Motors"
        )

        return fig, ax

# main function
def main():

    # create database
    database = Database(
        [Inputs.folder_path + "/" + file_name + ".txt" for file_name in Inputs.file_names]
    )
    print(database)

    # plot results
    database.scatter_plot(
        "rated_power_W", "max_rpm", "Rated Power (W)", "Max. rpm"
    )
    database.scatter_plot(
        "weight_g", "rated_power_W", "Weight (g)", "Rated Power (W)"
    )

# runs on script execution
if __name__ == "__main__":

    # run main
    main()

    # show all plots
    plt.show()
