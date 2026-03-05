# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from collections import defaultdict

# create dictionary of data
"""data = {}

# populate data
data["aircraft_classes"] = [
    "Commercial transport",
    "Regional jet",
    "Business jet",
    "High-performance"
]

# make up some numbers
data["take_off_thrust_to_weight"] = [0.3, 0.35, 0.45, 0.8]
data["cruise_lift_to_drag"] = [17, 17, 14, 12]"""

# create Constants class
class Constants:
    """Stores useful constants."""
    # physical constants
    g = 9.80665

    # plotting constants
    figsize = (8, 5)
    fontsize = 12
    fontsize_small = 11

# create Dataset class
class Dataset:
    """Stores the contents of the Jenkinson aircraft database."""
    def __init__(self, folder_path):
        """Reads all .xls files in the specified folder and converts to a python dictionary."""
        # create dictionary and loop over all files
        data = defaultdict(list)
        for filepath in glob.glob(f"{folder_path}/*.xls"):

            # read as pandas database and loop over all sheets in the .xls
            """df_all = pd.read_excel(
                filepath, header = None, index_col = 0, keep_default_na = False, sheet_name = None
            )
            for sheet_name, df in df_all.items():"""
            xl = pd.ExcelFile(filepath)
            for sheet_name in xl.sheet_names:
                df = pd.read_excel(
                    xl, sheet_name=sheet_name, header=None, index_col=0
                )
                df = df.map(lambda x: np.nan if isinstance(x, str) and x.strip() == '' else x)
                df = df.dropna(axis=1, how='all')
                df = df.fillna('')
                print(f"\nSheet: {sheet_name}")
                print(f"Shape: {df.shape}")
                last_col = df.iloc[:, -1]
                non_null = last_col.dropna()
                if len(non_null) > 0:
                    print(f"{sheet_name} last column non-null values: {non_null.tolist()}")

                prefix = ""

                # loop over all rows
                for index, row in df.iterrows():

                    if all("" == el for el in row.tolist()):

                        prefix = index.strip().lower()

                    else:

                        # append data to dictionary
                        data[prefix + " " + index.strip().lower()].extend(row.tolist())

        # convert to alphabetically-sorted dictionary and store
        self.data = dict(data)
        self.data = dict(sorted(self.data.items(), key=lambda item: item[0]))
        
        # print all dictionary keys
        for key, value in self.data.items():

            #print(f"key: {key}\nvalue: {value}")
            print(f"{key}")

    def plot_thrust(self):
        """Plots thrust against weight for the aircraft in the dataset."""
        # convert relevant data to numpy arrays
        weight = np.array(self.data["mass (weight) (kg): max. take-off"], dtype = object)
        thrust = np.array(self.data["in service (ordered) static thrust (kn)"], dtype = object)
        no_of_engines = np.array(self.data["in service (ordered) no. of engines"], dtype = object)

        # mask where neither value is an empty string
        mask = (weight != '') & (thrust != '') & (no_of_engines != '')

        # perform linear regression
        m, c = np.polyfit(
            weight[mask].astype(float),
            thrust[mask].astype(float) * no_of_engines[mask].astype(float), 1
        )

        # create plot
        fig, ax = plt.subplots(figsize = Constants.figsize)

        # plot static thrust against aircraft weight
        ax.plot(
            weight[mask].astype(float),
            thrust[mask].astype(float) * no_of_engines[mask].astype(float),
            linestyle = '', marker = '.', markersize = 8,
            label = "Nominal"
        )

        # plot results of linear regression
        xx = np.linspace(np.min(weight[mask].astype(float)), np.max(weight[mask].astype(float)), 100)
        ax.plot(xx, m * xx + c, label = f"Linear regression\nm = {m:.3g}, c = {c:.3g}")

        # configure plot
        plt.tight_layout(rect = (0.05, 0.05, 0.95, 0.9))
        ax.grid()
        ax.legend(fontsize = Constants.fontsize_small)
        ax.set_xlabel("Maximum take-off weight (kg)", fontsize = Constants.fontsize)
        ax.set_ylabel("Engine static thrust (kN)", fontsize = Constants.fontsize)
        ax.tick_params(axis = 'both', which = 'major', labelsize = Constants.fontsize_small)

    def plot_thrust_to_weight(self):
        """Plots thrust-to-weight against weight for the aircraft in the dataset."""
        # convert relevant data to numpy arrays
        weight = np.array(self.data["mass (weight) (kg): max. take-off"], dtype = object)
        thrust = np.array(self.data["in service (ordered) static thrust (kn)"], dtype = object)
        no_of_engines = np.array(self.data["in service (ordered) no. of engines"], dtype = object)
        thrust_to_weight = np.array(self.data["loadings: thrust/weight ratio"], dtype = object)

        # mask where neither value is an empty string
        mask = (weight != "") & (thrust != "") & (no_of_engines != "") & (thrust_to_weight != "")

        # create new plot
        fig, ax = plt.subplots(figsize = Constants.figsize)

        # plot thrust-to-weight ratio against aircraft weight
        ax.plot(
            weight[mask].astype(float) * Constants.g,
            thrust_to_weight[mask].astype(float),
            linestyle = '', marker = '.', markersize = 8,
            label = "Nominal"
        )

        # repeat for nominal values in the dataset
        ax.plot(
            weight[mask].astype(float) * Constants.g,
            1000 * thrust[mask].astype(float) * no_of_engines[mask].astype(float)
            / (weight[mask].astype(float) * Constants.g),
            linestyle = '', marker = '.', markersize = 4,
            label = "Calculated"
        )

        # configure plot
        plt.tight_layout(rect = (0.05, 0.05, 0.95, 0.9))
        ax.grid()
        ax.legend(fontsize = Constants.fontsize_small)
        ax.set_xlabel("Maximum take-off weight (kg)", fontsize = Constants.fontsize)
        ax.set_ylabel("Thrust-to-weight ratio", fontsize = Constants.fontsize)
        ax.tick_params(axis = 'both', which = 'major', labelsize = Constants.fontsize_small)

    def plot_lift_to_drag(self, e = 0.8, C_D0 = 0.02):
        """Plots estimated cruise L/D against MTOW for the aircraft in the dataset."""
        # create plot
        fig, ax = plt.subplots(figsize = Constants.figsize)

        # convert to numpy arrays
        weight = np.array(self.data["mass (weight) (kg): max. take-off"], dtype = object)
        aspect_ratio = np.array(self.data["wing: aspect ratio"], dtype = object)

        # mask where neither value is an empty string
        mask = (weight != "") & (aspect_ratio != "")

        # calculate L/D for masked entries
        lift_to_drag_cruise = 0.5 * np.sqrt(np.pi * aspect_ratio[mask].astype(float) * e / C_D0)

        # plot
        ax.plot(
            weight[mask].astype(float),
            lift_to_drag_cruise,
            linestyle='', marker='.', markersize = 8,
            label = "Calculated", color = "C1"
        )

        # configure plot
        plt.tight_layout(rect = (0.05, 0.05, 0.95, 0.9))
        ax.grid()
        ax.legend(fontsize = Constants.fontsize_small)
        ax.set_xlabel("Maximum take-off weight (kg)", fontsize = Constants.fontsize)
        ax.set_ylabel("Estimated cruise L/D", fontsize = Constants.fontsize)
        ax.tick_params(axis = 'both', which = 'major', labelsize = Constants.fontsize_small)
        ax.text(
            0.5, 1.05, rf"$e = {e},\ C_{{D0}} = {C_D0}$",
            transform = ax.transAxes, ha = 'center', va = 'center', fontsize = Constants.fontsize
        )

def main():
    """Main function."""
    #
    data = Dataset("../../Jenkinson Aircraft Database")
    data.plot_thrust()
    data.plot_thrust_to_weight()
    data.plot_lift_to_drag()

if __name__ == "__main__":

    main()
    plt.show()
