# import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import mat73

# import high speed solver modules
from engine import Engine
from flight_scenario import Flight_scenario
import utils

# Inputs class
class Inputs:
	"""Stores various inputs relevant to the post-processing script."""
	# default grid size over which to interpolate
	N = 100

	# default plotting parameters
	figsize = (9, 6)

# Post class
class Post:
	"""Stores variables from post-processing as CFD data and produces plots."""
	def __init__(self):
		"""Creates an instance of the Post class."""
		# read post-processing .mat object
		self.data = mat73.loadmat("CFD_data.mat")["post"]
		print(f"self.data: {self.data}")

	def calculate_thrust(self):
		"""Calculates the actual thrust produced by the CFD test cases via the Engine class."""
		# create lists of data to calculate
		self.data["thrust"] = np.zeros(len(self.data["phi"]))

		# loop for each CFD entry
		for index in range(len(self.data["phi"])):

			# create Flight_scenario object
			scenario = Flight_scenario(
				"",
				self.data["metadata"][index]["altitude"],
				self.data["metadata"][index]["flight_speed"],
				self.data["metadata"][index]["diameter"],
				self.data["metadata"][index]["hub_tip_ratio"],
				self.data["metadata"][index]["thrust"],
			)

			# create engine
			engine = Engine(
				scenario,
				self.data["metadata"][index]["no_of_stages"],
				self.data["phi"][index],
				self.data["psi"][index],
				self.data["metadata"][index]["vortex_exponent"],
				self.data["metadata"][index]["pressure_loss_coefficient"],
				self.data["metadata"][index]["blade_row_area_ratio"],
				self.data["metadata"][index]["inlet_mach_number"]
			)

			# store compressor exit properties at nozzle inlet
			engine.nozzle.inlet.M = self.data["M"][index]
			engine.nozzle.inlet.alpha = utils.deg_to_rad(self.data["alpha"][index])
			engine.nozzle.inlet.T_0 = self.data["T_0"][index] / self.data["metadata"][index]["T_0"]
			engine.nozzle.inlet.p_0 = self.data["p_0"][index] / self.data["metadata"][index]["p_0"]
			engine.nozzle.inlet.s = self.data["s"][index] / (utils.gamma * utils.R)

			# calculate variation of radial positions - this isn't actually "span"!
			engine.nozzle.inlet.rr = (
				self.data["span"][index] / scenario.radius
			)

			# calculate and store static properties
			engine.nozzle.inlet.T = (
				engine.nozzle.inlet.T_0 * utils.stagnation_temperature_ratio(engine.nozzle.inlet.M)
			)
			engine.nozzle.inlet.p = (
				engine.nozzle.inlet.p_0 * utils.stagnation_pressure_ratio(engine.nozzle.inlet.M)
			)

			"""fig, ax = plt.subplots()
			ax.plot(engine.nozzle.inlet.M, engine.nozzle.inlet.rr, label = "M")
			ax.plot(engine.nozzle.inlet.alpha, engine.nozzle.inlet.rr, label = "alpha")
			ax.plot(engine.nozzle.inlet.T_0, engine.nozzle.inlet.rr, label = "T_0")
			ax.plot(engine.nozzle.inlet.p_0, engine.nozzle.inlet.rr, label = "p_0")
			ax.grid()
			ax.legend()
			plt.show()"""

			# delete lists of blade rows and stages from engine
			engine.blade_rows = []
			engine.stages = []

			print(engine)

			# calculate nozzle exit conditions
			engine.design()

			# calculate engine thrust
			engine.evaluate()
			engine.dimensional_values()

			# print values
			print(f"Nominal: {self.data['metadata'][index]['thrust']} N")
			print(f"Actual: {engine.thrust} N")

			self.data["thrust"][index] = engine.thrust

	def plot_contours(self, attribute, label = ""):
		"""Creates a contour plot on axes of flow and stage loading coefficient."""
		# store x-, y- and z-values separately for convenience
		xx = self.data["phi"]
		yy = self.data["psi"]
		zz = self.data[attribute]
		print(f"zz: {zz}")
		print(f"[type(z) for z in zz]: {[type(z) for z in zz]}")

		# convert to 1D
		zz = zz.max(axis=1) if zz.ndim == 2 else zz

		# create fine grid of x- and y-values over which to interpolate
		xx_fine = np.linspace(np.min(xx), np.max(xx), Inputs.N)
		yy_fine = np.linspace(np.min(yy), np.max(yy), Inputs.N)

		# create meshed grid of x- and y-values
		xx_grid, yy_grid = np.meshgrid(xx_fine, yy_fine, indexing = "ij")

		# mask out NaNs
		mask = ~np.isnan(zz)

		# construct griddata interpolating object
		grid = griddata(
			points = (xx[mask], yy[mask]), values = zz[mask],
			xi = (xx_grid, yy_grid),
			method = "cubic"
		)

		# compute bounds from data
		vmin, vmax = np.nanmin(zz), np.nanmax(zz)

		if np.all(zz == zz.astype(int)):
			levels = np.arange(int(vmin), int(vmax) + 1)
		else:
			levels = 20

		# create plot
		fig, ax = plt.subplots(figsize = Inputs.figsize)

		# plot contours of interpolated data and individual datapoints
		cf = ax.contourf(
			xx_grid, yy_grid, np.ma.masked_invalid(grid), levels = levels, cmap = "viridis",
			vmin = vmin, vmax = vmax, extend = "both"
		)
		plt.colorbar(cf, ax = ax, label = label)
		ax.scatter(xx, yy, c = "black", s = 10, zorder = 5, label = "Datapoints")

		# configure plot
		ax.set_xlabel('Flow Coefficient, φ')
		ax.set_ylabel('Stage Loading Coefficient, ψ')
		ax.set_title(f"{label} | No. of Stages: {self.data['metadata'][0]['no_of_stages']}")

		# save figure
		fig.savefig(f"contour_plot_{attribute}", dpi = 300, bbox_inches = "tight")

# main function
def main():

	# create post-processing object
	post = Post()

	# calculate actual engine thrust
	post.calculate_thrust()

	# create contour plots of efficiency and number of blades
	post.plot_contours("eta_comp", "Compressor Polytropic Efficiency")
	post.plot_contours("thrust", "Engine Thrust (N)")
	post.plot_contours("no_of_blades", "Compressor No. of Blades")
	post.plot_contours("power", "Motor Power (W)")

# upon script execution
if __name__ == "__main__":

	# run main()
	main()

	# show all plots
	plt.show()
