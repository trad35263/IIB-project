# import modules
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata

# import high speed solver modules
from engine import Engine
from flight_scenario import Flight_scenario

# Inputs class
class Inputs:
	"""Stores various inputs relevant to the post-processing script."""
	# default grid size over which to interpolate
	N = 100

# Post class
class Post:
	"""Stores variables from post-processing as CFD data and produces plots."""
	def __init__(self):
		"""Creates an instance of the Post class."""
		# read post-processing .mat object
		self.data = scipy.io.loadmat("../trad3-multistage-fans/CFD_data.mat")
		print(f"self.data: {self.data}")

	def calculate_thrust(self):
		"""Calculates the actual thrust produced by the CFD test cases via the Engine class."""
		# loop for each CFD entry
		for index in range(len(self.data["phi"])):

			# create Flight_scenario object
			scenario = Flight_scenario(
				self.data["metadata"]["altitude"],
				self.data["metadata"]["flight_speed"],
				self.data["metadata"]["diameter"],
				self.data["metadata"]["hub_tip_ratio"],
				self.data["metadata"]["thrust"],
			)

			# create engine
			engine = Engine(
				scenario, self.data["metadata"][index]["no_of_stages"], self.data["phi"][index],
				self.data["psi"][index], self.data["metadata"][index]["vortex_exponent"],
				self.data["metadata"][index]["pressure_loss_coefficient"],
				self.data["metadata"][index]["area_ratio"]
			)

			# store compressor exit properties at nozzle inlet
			engine.nozzle.inlet.M = self.data["M"][index]
			engine.nozzle.inlet.alpha = self.data["alpha"][index]
			engine.nozzle.inlet.T_0 = self.data["T_0"][index] / self.data["metadata"]["T_0"]
			engine.nozzle.inlet.p_0 = self.data["p_0"][index] / self.data["metadata"]["p_0"]

			# calculate variation of radial positions from span
			engine.nozzle.inlet.span = (
				self.data["span"][index] * (1 - self.data["metadata"]["hub_tip_ratio"])
				+ self.data["metadata"]["hub_tip_ratio"]
			)

			# calculate nozzle exit conditions
			engine.nozzle.design()

			# set engine inlet conditions
			engine.blade_rows[0].set_inlet_conditions(
				self.data["metadata"][index]["M_1"],
				self.data["metadata"][index]["hub_tip_ratio"],
			)

			# calculate engine thrust
			engine.evaluate()
			engine.dimensional_values()

			# print values
			print(f"Nominal: {self.data['metadat'][index]['thrust']} N")
			print(f"Actual: {engine.thrust} N")

	def plot_contours(self, attribute, label = ""):
		"""Creates a contour plot on axes of flow and stage loading coefficient."""
		# store x-, y- and z-values separately for convenience
		xx = self.data["phi"]
		yy = self.data["psi"]
		zz = getattr(self.data, attribute)

		print(f"zz: {zz}")

		# create fine grid of x- and y-values over which to interpolate
		xx_fine = np.linspace(np.min(xx), np.max(xx), Inputs.N)
		yy_fine = np.linspace(np.min(yy), np.max(yy), Inputs.N)

		# create meshed grid of x- and y-values
		xx_grid, yy_grid = np.meshgrid(xx_fine, yy_fine, indexing = "ij")

		# construct griddata interpolating object
		grid = griddata(
			points = (xx, yy), values = zz,
			xi = (xx_grid, yy_grid),
			method = "cubic"
		)

		# create plot
		fig, ax = plt.subplots(figsize = Inputs.figsize)

		# plot contours of interpolated data and individual datapoints
		cf = ax.contourf(
			xx_grid, yy_grid, np.ma.masked_invalid(grid), levels = 100, cmap = "viridis"
		)
		plt.colorbar(cf, ax = ax, label = label)
		ax.scatter(xx, yy, c = "white", s = 10, zorder = 5, label = "Datapoints")

		# configure plot
		ax.set_xlabel('Flow Coefficient, φ')
		ax.set_ylabel('Stage Loading Coefficient, ψ')
		ax.set_title(f"{label} | No. of Stages: {self.data.no_of_stages}")

# main function
def main():

	# create post-processing object
	post = Post()

	# calculate actual engine thrust
	post.calculate_thrust()

	# create contour plots of efficiency and number of blades
	post.plot_contours("eta_comp", "Compressor Polytropic Efficiency")

# upon script execution
if __name__ == "__main__":

	# run main()
	main()

	# show all plots
	plt.show()
