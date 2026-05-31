# aircraft_range.py
# 30 May 2026

# import modules
import numpy as np
import matplotlib.pyplot as plt

# import high speed solver
import utils

# load Latex font
import matplotlib.font_manager as fm
font_path = r"C:\Windows\Fonts\texgyretermes-regular.otf"
prop = fm.FontProperties(fname = font_path)

# update matplotlib global parameters
plt.rcParams.update({
    "font.family": "TeX Gyre Termes",
    "font.size": 12,
    "mathtext.fontset": "stix",
})

# Inputs class
class Inputs:

    # design decisions
    engine_mass = 1                     # kg
    no_of_engines = 2
    aircraft_mass = 300                 # kg
    scaling_factor = 10
    power_per_engine = 30000            # W
    flight_speed = 170                  # m/s

    # constants
    battery_energy_density = 936000     # J/kg

    # grid resolution
    N = 100

# main function
def main():

    # get list of battery masses to consider
    mm = np.linspace(0, Inputs.aircraft_mass, Inputs.N)

    # get list of flight durations
    tt = mm * Inputs.battery_energy_density / (Inputs.power_per_engine * Inputs.no_of_engines)

    # get list of flight ranges
    rr = Inputs.flight_speed * tt
    
    # create plot
    fig, ax = plt.subplots(figsize = utils.Defaults.figsize)

    # plot range against battery mass
    ax.plot(mm, rr)

    # configure plot
    ax.set_xlabel("Battery mass (kg)")
    ax.set_ylabel("Range (m)")
    ax.grid()
    ax.set_title(
        rf"Range vs. Battery Mass for {Inputs.aircraft_mass:.4g} kg aircraft with "
        rf"$P_\text{{in}}$ = {Inputs.power_per_engine * Inputs.no_of_engines:.4g}"
    )

    # save figure
    fig.savefig("exports/aircraft_range.png", dpi = utils.Defaults.dpi, bbox_inches = "tight")

# upon script execution
if __name__ == "__main__":

    # run main and show all plots
    main()
    plt.show()
