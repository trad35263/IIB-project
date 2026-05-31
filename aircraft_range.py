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
	no_of_engines = 2
	power_per_engine = 23550            # W
	flight_speed = 170                  # m/s

	# masses
	aircraft_mass = 307.2               # kg
	#engine_mass = 7.6106				# kg

	# empty-to-max-weight
	W_E_MTOW = 0.6

	# constants
	battery_energy_density = 936000     # J/kg

	# grid resolution
	N = 100

# main function
def main():

	# find weight of aircraft for payload and batteries
	mass = Inputs.aircraft_mass * (1 - 0.6)

	# get list of battery masses to consider
	mm = np.linspace(0, mass, Inputs.N)

	# get list of flight durations
	tt = mm * Inputs.battery_energy_density / (Inputs.power_per_engine * Inputs.no_of_engines)

	# get list of flight ranges
	ranges = Inputs.flight_speed * tt

	# convert to payload fractions
	payload_fractions = (mass - mm) / Inputs.aircraft_mass
	
	# create plot
	fig, ax = plt.subplots(figsize = utils.Defaults.figsize)

	# plot range against battery mass
	ax.plot(payload_fractions, ranges / 1000)

	# configure plot
	ax.set_xlabel("Payload Fraction")
	ax.set_ylabel("Range (km)")
	ax.grid()
	ax.set_title(
		rf"Range vs. Payload Fraction for {Inputs.aircraft_mass:.4g} kg Aircraft with "
		rf"$P_\text{{in}}$ = {Inputs.power_per_engine * Inputs.no_of_engines} W"
	)

	# save figure
	fig.savefig("exports/aircraft_range.png", dpi = utils.Defaults.dpi, bbox_inches = "tight")

# upon script execution
if __name__ == "__main__":

	# run main and show all plots
	main()
	plt.show()
