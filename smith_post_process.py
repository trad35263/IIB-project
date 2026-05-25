# smith_post_process.py
# 25 May 2026

# import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import mat73

from scipy.interpolate import CloughTocher2DInterpolator

# system modules
import os
import subprocess
import ast
import copy

# latex imports
from pylatex import Document, Figure, NoEscape, Package
#from pylatex.utils import bold

# import high speed solver
import sys
from engine import Engine
from flight_scenario import Flight_scenario
import utils

# override matplotlib default behaviour to render negative contours as dashed lines
import matplotlib as mpl
mpl.rcParams["contour.negative_linestyle"] = "solid"

import matplotlib.font_manager as fm

# load Latex font
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
	"""Stores various inputs relevant to the post-processing script."""
	# file path to TURBOSTREAM folder containing post-processed packages of data
	folder_path = "C:/Users/tomra/OneDrive/Documents/Uni/2B Coursework/IIB Project/CFD Data/"
	#folder_path = "/mnt/Data/trad3/results/TURBOSTREAM/"

	# perfect gas constants
	R = 287
	gamma = 1.4
	c_v = R / (gamma - 1)
	c_p = R + c_v

	# default grid size over which to interpolate
	N = 100

	# contour plot resolution
	levels = 10
	gridlines = 10

	# default plotting parameters
	figsize = (8, 4)
	dpi = 300
	fontsize = 12
	titlesize = 14
	colours = "viridis"
	alpha = 0.3

	# lookup table of units
	units = {
		"ro": r"$kg\,m^{-3}$",
		"rovx": r"$kg\,m^{-2}s^{-1}$",
		"rovr": r"$kg\,m^{-2}s^{-1}$",
		"rorvt": r"$kg\,m^{-1}s^{-1}$",
		"roe": r"$J\,m^{-3}$",
		"vx": r"$m\,s^{-1}$",
		"vr": r"$m\,s^{-1}$",
		"vtheta": r"$m\,s^{-1}$",
		"T": "K",
		"T_0": "K",
		"p": "Pa",
		"p_0": "Pa",
	}

	# empty list of saved figures
	saved_figures = []

	# mass flow rate threshold for classification as a boundary layer
	BL_threshold = 0.05

# Post class
class Post:
	"""Stores variables from post-processing as CFD data and produces plots."""
	def __init__(self):
		"""Creates an instance of the Post class."""
		# create empty list of post-processing .mat objects
		self.data = []

		# loop for each filename in input folder
		for filename in os.listdir(Inputs.folder):

			# if filename is a CFD data package
			if filename.startswith("data_") and filename.endswith(".mat"):

				# read post-procesing .mat file and store in list
				filepath = os.path.join(Inputs.folder, filename)
				mat = self.load_mat(filepath)
				self.data.append(mat)

				# rename "eta_comp" to "eta_poly"
				data = self.data[-1]
				data["eta_poly"] = data.pop("eta_comp", None)

		# print keys to terminal
		self.print_keys(self.data[0])

		# preliminary data analysis
		self.calc_secondary()
		self.analyse()
		#self.quantify_loss()

	def load_mat(self, filepath):
		"""Loads the data from a .mat file."""
		# load .mat object using mat73 package
		mat = mat73.loadmat(filepath)
		return mat["post"]
	
	def print_keys(self, d, indent = 0):
		if isinstance(d, dict):
			for key in d:
				print("|---" * indent + str(key))
				self.print_keys(d[key], indent + 1)
		elif isinstance(d, list):
			for i, item in enumerate(d):
				print("|---" * indent + f"[{i}]")
				self.print_keys(item, indent + 1)

	def calc_secondary(self):
		"""Calculates secondary flow properties at inlet and exit to each blade row."""
		# get index of desired test case to investigate
		"""phi, psi = Inputs.contour_ij
		index = next(
			i for i, d in enumerate(self.data)
			if (np.abs(d["phi"] - phi) < 1e-6) and (np.abs(d["psi"] - psi) < 1e-6)
		)

		# get relevant data object
		data = self.data[index]"""

		for data in self.data:

			# loop for each blade row
			for i, blade_row in enumerate(data["blade_rows"]):

				# get inlet and outlet
				inlet = blade_row["inlet"]
				outlet = blade_row["outlet"]

				# loop for inlet and exit
				for plane in (inlet, outlet):

					# separate gamma for convenience
					gamma = Inputs.gamma

					# separate primary variables for convenience
					ro = plane["ro"]
					rovx = plane["rovx"]
					rovr = plane["rovr"]
					rorvt = plane["rorvt"]
					roe = plane["roe"]

					# get radius values
					r = np.sqrt(plane["y"]**2 + plane["z"]**2)

					# retrieve individual velocity components
					v_x = rovx / ro
					v_r = rovr / ro
					v_theta = rorvt / (ro * r)
					plane["v_x"] = v_x
					plane["v_r"] = v_r
					plane["v_theta"] = v_theta

					# calculate flow angles
					alpha = np.arctan2(v_theta, v_x)
					plane["alpha"] = alpha

					# get velocity magnitude
					v = np.sqrt(v_x**2 + v_r**2 + v_theta**2)
					plane["v"] = v

					# get static and stagnation temperature
					T = (roe / ro - 0.5 * v**2) / Inputs.c_v
					T_0 = T + 0.5 * v**2 / Inputs.c_p
					plane["T"] = T
					plane["T_0"] = T_0

					# get Mach number
					M = v / np.sqrt(gamma * Inputs.R * T)
					plane["M"] = M

					# get static and stagnation pressure
					p = ro * Inputs.R * T
					p_0 = p * np.power(1 + 0.5 * (gamma - 1) * M**2, gamma / (gamma - 1))
					plane["p"] = p
					plane["p_0"] = p_0

					# get (dimensional) specific entropy
					s = (
						Inputs.c_p * np.log(T_0 / np.mean(data["blade_rows"][0]["inlet"]["T_0"]))
						- Inputs.R * np.log(p_0 / np.mean(data["blade_rows"][0]["inlet"]["p_0"]))
					)
					plane["s"] = s

	def analyse(self):
		"""Carries out preliminary analysis for each loaded test case."""
		# loop for each test case
		for i, data in enumerate(self.data):

			# loop for each blade row
			for j, blade_row in enumerate(data["blade_rows"]):

				# get inlet and exit planes
				inlet = blade_row["inlet"]
				outlet = blade_row["outlet"]

				# repeat code for inlet and exit planes
				for plane in (inlet, outlet):

					# find coordinates of centre-line through blade passage section
					j = int(np.floor(plane["y"].shape[1] / 2))

					# generate rotated coordinates
					self.rotate(plane, j)

					# get rotated, node-centred mesh coordinates
					y = plane["y"]
					z = plane["z"]

					# calculate cell-centred areas via cross product of diagonal vectors of each cell
					y1 = y[1:, 1:] - y[:-1, :-1]
					z1 = z[1:, 1:] - z[:-1, :-1]
					y2 = y[:-1, 1:] - y[1:, :-1]
					z2 = z[:-1, 1:] - z[1:, :-1]
					plane["A"] = 0.5 * (y1 * z2 - y2 * z1)

					# get cell-centred values of rovx
					rovx = plane["rovx"]
					plane["rovx_cell"] = (
						0.25 * (rovx[1:, 1:] + rovx[1:, :-1] + rovx[:-1, 1:] + rovx[:-1, :-1])
					)

					# get cell centred mass flow rate values
					plane["m_dot_cell"] = plane["rovx_cell"] * plane["A"]

					# sum cell-centred values to get cumulative mass flow rate in r for each theta
					plane["m_dot_cumulative"] = np.cumsum(plane["m_dot_cell"], axis = 0)

					# calculate cumulative mass flow rate as a 1D function of radius
					plane["m_dot_avg"] = np.sum(plane["m_dot_cumulative"], axis = 1)
					plane["m_dot_avg"] = np.append(np.array([0]), plane["m_dot_avg"])

					# find hub and tip radii for pitch-averaged boundary layer thickness
					plane["r_boundary_hub"] = (
						np.interp(
							data["metadata"]["hub_tip_ratio"] * Inputs.BL_threshold
							* plane["m_dot_avg"][-1],
							plane["m_dot_avg"], plane["r"]
						)
					)
					plane["r_boundary_tip"] = (
						np.interp(
							(1 - Inputs.BL_threshold) * plane["m_dot_avg"][-1],
							plane["m_dot_avg"], plane["r"]
						)
					)

					# convert to span
					plane["span_boundary_hub"] = (
						(plane["r_boundary_hub"] - plane["r"][0])
						/ (plane["r"][-1] - plane["r"][0])
					)
					plane["span_boundary_tip"] = (
						(plane["r_boundary_tip"] - plane["r"][0])
						/ (plane["r"][-1] - plane["r"][0])
					)

					# get list of indices corresponding to columns in cell-centred matrices
					columns = np.arange(plane["A"].shape[1])

					# calculate hub boundary layer indices
					indices = np.argmax(
						plane["m_dot_cumulative"]
						>= data["metadata"]["hub_tip_ratio"] * Inputs.BL_threshold
						* plane["m_dot_cumulative"][-1, :],
						axis = 0
					)

					# find node-centred boundary layer position values
					plane["y_boundary_hub"] = np.append(
						y[indices, columns], y[indices[-1], columns[-1] + 1]
					)
					plane["z_boundary_hub"] = np.append(
						z[indices, columns], z[indices[-1], columns[-1] + 1]
					)

					# find mask for hub  boundary layer
					num_rows, num_cols = plane["m_dot_cumulative"].shape
					row_indices = np.arange(num_rows)[:, np.newaxis]
					plane["boundary_hub"] = row_indices <= indices

					# repeat for tip
					indices = np.argmax(
						plane["m_dot_cumulative"]
						>= (1 - Inputs.BL_threshold) * plane["m_dot_cumulative"][-1, :],
						axis = 0
					)
					plane["y_boundary_tip"] = np.append(
						y[indices, columns], y[indices[-1], columns[-1] + 1]
					)
					plane["z_boundary_tip"] = np.append(
						z[indices, columns], z[indices[-1], columns[-1] + 1]
					)

					# find mask for tip boundary layer
					row_indices = np.arange(plane["m_dot_cumulative"].shape[0])[:, np.newaxis]
					plane["boundary_tip"] = row_indices >= indices

					# get cell-centred y- and z-values
					plane["y_cell"] = 0.25 * (y[1:, 1:] + y[1:, :-1] + y[:-1, 1:] + y[:-1, :-1])
					plane["z_cell"] = 0.25 * (z[1:, 1:] + z[1:, :-1] + z[:-1, 1:] + z[:-1, :-1])

					# get cell-centred entropy values
					s = plane["s"]
					plane["s_cell"] = 0.25 * (s[1:, 1:] + s[1:, :-1] + s[:-1, 1:] + s[:-1, :-1])

				# calculate axial velocity density ratio based on mass-averaged mass flux
				rovx_in_mean = (
					np.sum(inlet["rovx_cell"]**2 * inlet["A"])
					/ np.sum(inlet["rovx_cell"] * inlet["A"])
				)
				rovx_out_mean = (
					np.sum(outlet["rovx_cell"]**2 * outlet["A"])
					/ np.sum(outlet["rovx_cell"] * outlet["A"])
				)
				blade_row["AVDR"] = rovx_out_mean / rovx_in_mean

				# calculate pitch-to-chord distribution
				blade_row["outlet"]["pitch_to_chord"] = (
					2 * np.pi * outlet["r"] / (outlet["chord"] * blade_row["no_of_blades"])
				)

				# calculate hub entropy flux
				blade_row["outlet"]["s_flux_hub"] = (
					np.sum(
						blade_row["outlet"]["rovx_cell"] * blade_row["outlet"]["s_cell"]
						* blade_row["outlet"]["A"] * blade_row["outlet"]["boundary_hub"]
					) - np.sum(
						blade_row["inlet"]["rovx_cell"] * blade_row["inlet"]["s_cell"]
						* blade_row["inlet"]["A"] * blade_row["inlet"]["boundary_hub"]
					)
				)

				# calculate tip entropy flux
				blade_row["outlet"]["s_flux_tip"] = (
					np.sum(
						blade_row["outlet"]["rovx_cell"] * blade_row["outlet"]["s_cell"]
						* blade_row["outlet"]["A"] * blade_row["outlet"]["boundary_tip"]
					) - np.sum(
						blade_row["inlet"]["rovx_cell"] * blade_row["inlet"]["s_cell"]
						* blade_row["inlet"]["A"] * blade_row["inlet"]["boundary_tip"]
					)
				)

				# calculate midspan entropy flux
				blade_row["outlet"]["s_flux_midspan"] = (
					np.sum(
						blade_row["outlet"]["rovx_cell"] * blade_row["outlet"]["s_cell"]
						* blade_row["outlet"]["A"]
						* (~blade_row["outlet"]["boundary_hub"] & ~blade_row["outlet"]["boundary_tip"])
					) - np.sum(
						blade_row["inlet"]["rovx_cell"] * blade_row["inlet"]["s_cell"]
						* blade_row["inlet"]["A"]
						* (~blade_row["inlet"]["boundary_hub"] & ~blade_row["inlet"]["boundary_tip"])
					)
				)

				# blade row is a rotor
				if blade_row["rpm"] != 0:

					# calculate Lieblein diffusion factor across span
					DF = (
						1 - outlet["M_rel_avg"] / inlet["M_rel_avg"]
						* np.sqrt(outlet["T_avg"] / inlet["T_avg"]) + 0.5 * np.abs(
							np.sin(utils.deg_to_rad(inlet["beta_avg"]))
							- outlet["M_rel_avg"] / inlet["M_rel_avg"]
							* np.sqrt(outlet["T_avg"] / inlet["T_avg"])
							* np.sin(utils.deg_to_rad(outlet["beta_avg"]))
						) * outlet["pitch_to_chord"]
					)

					# calculate stagnation pressure loss coefficient across span
					Y_p = (
						(
							inlet["p_0_rel_avg"] * np.power(
								outlet["T_0_rel_avg"] / inlet["T_0_rel_avg"],
								utils.gamma / (utils.gamma - 1)
							) - outlet["p_0_rel_avg"])
						/ (inlet["p_0_rel_avg"] - inlet["p_avg"])
					)

					# calculate deviation across span
					deviation = outlet["metal_angle"] - outlet["beta_avg"]

					# calculate incidence across span
					incidence = inlet["beta_avg"] - inlet["metal_angle"]

				# blade row is a stator
				else:

					# calculate Lieblein diffusion factor across span
					DF = (
						1 - outlet["M_avg"] / inlet["M_avg"]
						* np.sqrt(outlet["T_avg"] / inlet["T_avg"]) + 0.5 * np.abs(
							np.sin(utils.deg_to_rad(inlet["alpha_avg"]))
							- outlet["M_avg"] / inlet["M_avg"]
							* np.sqrt(outlet["T_avg"] / inlet["T_avg"])
							* np.sin(utils.deg_to_rad(outlet["alpha_avg"]))
						) * outlet["pitch_to_chord"]
					)

					# calculate stagnation pressure loss coefficient across span
					Y_p = (
						(inlet["p_0_avg"] - outlet["p_0_avg"])
						/ (inlet["p_0_avg"] - inlet["p_avg"])
					)

					# calculate deviation across span
					deviation = outlet["alpha_avg"] - outlet["metal_angle"]

					# calculate incidence across span
					incidence = inlet["alpha_avg"] - inlet["metal_angle"]

				# store diffusion factor distribution and max. diffusion factor
				blade_row["outlet"]["diffusion_factor_avg"] = DF
				blade_row["max_diffusion_factor"] = np.max(DF)

				# store distribution of and max. stagnation pressure loss coefficient
				blade_row["outlet"]["Y_p_avg"] = Y_p
				blade_row["max_Y_p"] = np.max(Y_p)

				# store deviation distribution and max. deviation
				blade_row["outlet"]["deviation_avg"] = deviation
				blade_row["max_deviation"] = np.max(deviation)

				# store incidence distribution and max. incidence
				blade_row["outlet"]["incidence_avg"] = incidence
				blade_row["max_incidence"] = np.max(incidence)

				# calculate inlet cumulative mass flow rate
				inlet["m_dot_avg"] = (
					utils.cumulative_trapezoid(inlet["r"], 2 * np.pi * inlet["r"] * inlet["rovx_avg"])
				)
				outlet["m_dot_avg"] = (
					utils.cumulative_trapezoid(outlet["r"], 2 * np.pi * outlet["r"] * outlet["rovx_avg"])
				)

				# get downstream radial positions and mass fluxes via interpolation
				outlet["r_mass"] = np.interp(inlet["m_dot_avg"], outlet["m_dot_avg"], outlet["r"])
				outlet["rovx_mass_avg"] = np.interp(inlet["m_dot_avg"], outlet["m_dot_avg"], outlet["rovx_avg"])
				outlet["s_mass_avg"] = np.interp(inlet["m_dot_avg"], outlet["m_dot_avg"], outlet["s_avg"])

				# calculate entropy flux per blade row, non-dimensionalised by total input power
				T_0 = data["metadata"]["T_0"]
				outlet["loss_function_avg"] = (
					np.maximum(
						utils.cumulative_trapezoid(
							inlet["r"],
							2 * np.pi * inlet["r"] * T_0 * (
								outlet["rovx_mass_avg"] * outlet["s_mass_avg"] - inlet["rovx_avg"] * inlet["s_avg"]
							)
						), 0
					)
					/ np.sum([blade_row["power"] for blade_row in data["blade_rows"]])
				)

				# calculate second derivative of mass flux
				inlet["d2_rovx_dr2_avg"] = (
					np.gradient(np.gradient(inlet["rovx_avg"], inlet["r"]), inlet["r"])
				)
				outlet["d2_rovx_dr2_avg"] = (
					np.gradient(np.gradient(outlet["rovx_avg"], outlet["r"]), outlet["r"])
				)

				# find all crossing points for determination of boundary layer
				sign_changes = np.diff(np.sign(outlet["d2_rovx_dr2_avg"])) != 0
				idx = np.where(sign_changes)[0]
				
				# linear interpolation formulas to find exact x-crossings
				x0, x1 = outlet["r"][idx], outlet["r"][idx + 1]
				y0, y1 = outlet["d2_rovx_dr2_avg"][idx], outlet["d2_rovx_dr2_avg"][idx + 1]
				
				# find crossing points
				x_crossings = x0 - y0 * (x1 - x0) / (y1 - y0)

				# convert to span
				span_crossings = (x_crossings - outlet["r"][0]) / (outlet["r"][-1] - outlet["r"][0])

				# define boundary layer points
				outlet["boundary_layer_thickness_hub"] = span_crossings[1]
				outlet["boundary_layer_thickness_tip"] = span_crossings[-2]

	def calculate_thrust(self):
		"""Calculates the actual thrust produced by the CFD test cases via the Engine class."""
		# loop for each dataset
		for data in self.data:

			# create Flight_scenario object
			scenario = Flight_scenario(
				"",
				data["metadata"]["altitude"],
				data["metadata"]["flight_speed"],
				data["metadata"]["diameter"],
				data["metadata"]["hub_tip_ratio"],
				data["metadata"]["thrust"],
			)

			# create Engine object
			engine = Engine(
				scenario,
				data["metadata"]["no_of_stages"],
				data["phi"] * np.ones(data["metadata"]["no_of_stages"]),
				data["psi"] * np.ones(data["metadata"]["no_of_stages"]),
				data["metadata"]["vortex_exponent"],
				data["metadata"]["pressure_loss_coefficient"],
				data["metadata"]["blade_row_area_ratio"],
				data["metadata"]["inlet_mach_number"]
			)
			engine.design()

			# solve for engine geometry
			engine.geometry = {
				"aspect_ratio": data["metadata"]["aspect_ratio"],
				"diffusion_factor": data["metadata"]["diffusion_factor"],
				"design_parameter": data["metadata"]["design_parameter"]
			}
			engine.empirical_design()

			# run detailed thrust breakdown calculations
			engine.calc_thrust()

			# store engine in data object
			data["engine_design"] = engine

			# create a copy of the engine
			engine = copy.deepcopy(data["engine_design"])

			# set nozzle inlet conditions to (dimensionless) compressor exit conditions
			engine.nozzle.inlet.M = data["blade_rows"][-1]["outlet"]["M_avg"]
			engine.nozzle.inlet.alpha = utils.deg_to_rad(data["blade_rows"][-1]["outlet"]["alpha_avg"])
			engine.nozzle.inlet.T_0 = data["blade_rows"][-1]["outlet"]["T_0_avg"]
			engine.nozzle.inlet.p_0 = data["blade_rows"][-1]["outlet"]["p_0_avg"]
			engine.nozzle.inlet.s = data["blade_rows"][-1]["outlet"]["s_avg"] / (utils.gamma * utils.R)
			engine.nozzle.inlet.rr = (
				data["blade_rows"][-1]["outlet"]["r"] / data["blade_rows"][0]["outlet"]["r"][-1]
			)
			engine.nozzle.inlet.v_x = (
				data["blade_rows"][-1]["outlet"]["M_avg"]
				* np.cos(utils.deg_to_rad(data["blade_rows"][-1]["outlet"]["alpha_avg"]))
				* np.sqrt(data["blade_rows"][-1]["outlet"]["T_avg"])
			)
			engine.nozzle.inlet.v_theta = (
				data["blade_rows"][-1]["outlet"]["M_avg"]
				* np.sin(utils.deg_to_rad(data["blade_rows"][-1]["outlet"]["alpha_avg"]))
				* np.sqrt(data["blade_rows"][-1]["outlet"]["T_avg"])
			)

			# calculate and store static properties
			engine.nozzle.inlet.T = (
				engine.nozzle.inlet.T_0 * utils.stagnation_temperature_ratio(engine.nozzle.inlet.M)
			)
			engine.nozzle.inlet.p = (
				engine.nozzle.inlet.p_0 * utils.stagnation_pressure_ratio(engine.nozzle.inlet.M)
			)

			# delete lists of blade rows and stages from engine
			engine.blade_rows = []
			engine.stages = []

			# calculate nozzle exit conditions
			engine.design()
			engine.evaluate()
			engine.dimensional_values()

			"""if np.isnan(engine.C_th):

				plt.close("all")

				print(f"engine.nozzle.inlet.v_theta: {engine.nozzle.inlet.v_theta}")

				fig, axes = plt.subplots(1, 2)
				axes[0].plot(engine.nozzle.inlet.v_x, engine.nozzle.inlet.rr, color = "C0")
				axes[1].plot(engine.nozzle.exit.v_x, engine.nozzle.exit.rr, color = "C0")
				axes[0].plot(engine.nozzle.inlet.v_theta, engine.nozzle.inlet.rr, color = "C1")
				axes[1].plot(engine.nozzle.exit.v_theta, engine.nozzle.exit.rr, color = "C1")
				axes[0].grid()
				axes[1].grid()
				plt.show()"""

			# store input power as engine attribute
			engine.P_in = np.sum([blade_row["power"] for blade_row in data["blade_rows"]])

			# run detailed thrust breakdown calculations
			engine.calc_thrust()

			# store engine in data object
			data["engine"] = engine

		# get index of desired test case to investigate
		"""phi, psi = Inputs.contour_ij
		index = next(
			i for i, d in enumerate(self.data)
			if (np.abs(d["phi"] - phi) < 1e-6) and (np.abs(d["psi"] - psi) < 1e-6)
		)

		# store variables for convenience
		data = self.data[index]
		engine = copy.deepcopy(data["engine"])

		# create thrust sankey chart
		fig, ax = engine.plot_thrust()

		# put design vs. actual values in title
		ax.set_title(
			f"Design / actual thrust (%): {100 * actual_thrust / design_thrust:.4g}\n"
			f"Design / actual input power (%): {100 * actual_power / design_power:.4g}",
			fontsize = Inputs.titlesize
		)

		# save plot
		figname = (
			f"plot_thrust_phi_{phi}_psi_{psi}".replace(".", "_")
		)
		fig.savefig(
			os.path.join(Inputs.folder, figname),
			dpi = Inputs.dpi, bbox_inches = "tight"
		)

		# close plot
		Inputs.saved_figures.append(f"{figname}.png")
		print(f"Plot saved as {utils.Colours.GREEN}{figname}.png{utils.Colours.END}!")
		plt.close()"""

	def plot_contours(self, attribute, label = "", limits = None):
		"""Creates a contour plot on axes of flow and stage loading coefficient."""
		# store x-, y- and z-values separately for convenience
		xx = np.array([data["phi"] for data in self.data])
		yy = np.array([data["psi"] for data in self.data])
		zz = np.array([
			data[attribute]
			if attribute in data
			else [blade_row[attribute] for blade_row in data["blade_rows"]]
			if attribute in data["blade_rows"][0]
			else data["metadata"][attribute]
			if attribute in data["metadata"]
			else data["metadata"]["M"][attribute]
			if attribute in data["metadata"]["M"]
			else getattr(data["engine"], attribute)
			for data in self.data
		])

		# transpose data array and ensure at least 2D
		zz = np.atleast_2d(np.transpose(zz))

		# create fine grid of x- and y-values over which to interpolate
		xx_fine = np.linspace(np.min(xx), np.max(xx), Inputs.N)
		yy_fine = np.linspace(np.min(yy), np.max(yy), Inputs.N)

		# create meshed grid of x- and y-values
		xx_grid, yy_grid = np.meshgrid(xx_fine, yy_fine, indexing = "ij")

		# loop for each blade row, if attribute given is stored per blade row
		for index in range(zz.shape[0]):

			# mask out NaNs
			mask = ~np.isnan(zz[index])

			print(f"{attribute} min.: {np.min(zz[index][mask]):.4g}")
			print(f"{attribute} max.: {np.max(zz[index][mask]):.4g}")

			# Construct the Clough-Tocher interpolator object
			interp = CloughTocher2DInterpolator(
				points = np.column_stack((xx[mask], yy[mask])), 
				values = zz[index][mask]
			)

			# Evaluate on your grid
			grid = interp(xx_grid, yy_grid)

			# create plot
			fig, ax = plt.subplots(figsize = Inputs.figsize)

			# no data limits are provided
			if limits is None:

				# every instance of data is an integer
				if np.all(zz[index][mask] == np.floor(zz[index][mask])):

					# set integer contour bar range
					levels = np.arange(np.min(zz[index][mask]), np.max(zz[index][mask]), 1)

				# all other cases
				else:

					# set colour bar levels to be between minimum and maximum data values
					levels = np.linspace(
						np.min(zz[index][mask]), np.max(zz[index][mask]),
						Inputs.levels
					)

			# use provided limits
			else:

				# get data range rouned to 2 sig. fig.s
				"""data_range = limits[1] - limits[0]
				order = int(np.floor(np.log10(abs(data_range))))
				rounded = round(data_range, 1 - order)

				# start at the nearest power of 10 below "rounded"
				start_power = 10 ** int(np.floor(np.log10(rounded)))

				# define repeating step multipliers
				multipliers = [5, 2, 1]

				# scale factor down by 10 every time we exhaust the 5, 2, 1 cycle
				scale_factor = start_power 
				chosen_step = None

				# loop until exit condition is met
				while True:

					# loop for each multiplier
					for m in multipliers:

						# generate the current step size
						current_step = m * scale_factor
							
						# test the criteria: divide "rounded" by the current step
						divisions = rounded / current_step
						
						# check if the result is a clean integer division AND greater than Inputs.levels
						if np.isclose(divisions, np.round(divisions)) and divisions > Inputs.levels:
							chosen_step = current_step
							break
							
					# if the inner loop found a match, break the outer while loop
					if chosen_step is not None:
						break
						
					# drop down an order of magnitude for the next cycle
					scale_factor /= 10

				# use provided limits
				levels = np.linspace(*limits, int(rounded / chosen_step) + 1)"""

				levels = np.linspace(*limits)

			# add 
			cs = ax.contour(
				xx_grid, 
				yy_grid, 
				np.ma.masked_invalid(grid), 
				levels = levels,
				cmap="viridis", 
				extend="both"
			)

			# 3. Update the colorbar to map to the line contour object
			#plt.colorbar(cs, ax=ax, label=label, extend="both")

			# add text labels directly onto the contour lines
			ax.clabel(cs, inline = True, fontsize = Inputs.fontsize, fmt='%1.2g')

			# plot contours of interpolated data and individual datapoints
			cf = ax.contourf(
				xx_grid, yy_grid, np.ma.masked_invalid(grid), levels = levels, cmap = "viridis",
				extend = "both", alpha = Inputs.alpha
			)
			plt.colorbar(cf, ax = ax, label = label, extend = "both")
			ax.scatter(xx[mask], yy[mask], c = "k", s = 10, zorder = 5, label = "Datapoints")
			ax.scatter(xx[~mask], yy[~mask], c = "red", s = 10, zorder = 5)

			# set axis labels
			ax.set_xlabel('Flow Coefficient, φ')
			ax.set_ylabel('Stage Loading Coefficient, ψ')
			
			# construct title
			title = (
				f"No. of Stages: {self.data[0]['metadata']['no_of_stages']} | "
				f"Inlet Mach Number: {self.data[0]['metadata']['inlet_mach_number']}"
			)
			if zz.shape[0] > 1:

				# append blade row to title if contour plot is specific to one row
				title += f"\n{'Rotor' if index % 2 == 0 else 'Stator'} {index // 2 + 1}"

			# set title
			ax.set_title(title, fontsize = Inputs.titlesize)

			# save plot
			figname = (
				f"plot_contours_row_{index}_{attribute}".replace(".", "_")
			)
			fig.savefig(
				os.path.join(Inputs.folder, figname),
				dpi = Inputs.dpi, bbox_inches = "tight"
			)

			# close plot
			Inputs.saved_figures.append(f"{figname}.png")
			print(f"Plot saved as {utils.Colours.GREEN}{figname}.png{utils.Colours.END}!")
			plt.close()

	def rotate(self, inlet, j):
		"""Calculates the rotated coordinates for a given cut and index to rotate to vertical."""
		# find coordinates of centre-line through blade passage section
		y1 = inlet["y"][0, j]
		z1 = inlet["z"][0, j]
		y2 = inlet["y"][-1, j]
		z2 = inlet["z"][-1, j]

		# get angle to rotate coordinates by
		theta = np.pi / 2 - np.arctan2(z2 - z1, y2 - y1)

		# define rotation matrix
		R = np.array([
			[np.cos(theta), -np.sin(theta)],
			[np.sin(theta),  np.cos(theta)]
		])

		# stack coordinates
		coords = np.stack([inlet["y"], inlet["z"]], axis = 0)

		# multiply the matrix by the coordinates
		coords_rotated = np.einsum('ij,j...', R, coords)

		# unpack rotated 2D arrays
		inlet["y"] = coords_rotated[..., 0]
		inlet["z"] = coords_rotated[..., 1]

	def section_view(self, attribute, label = ""):
		"""Creates a plot of a section view through the compressor."""
		# get index of desired test case to investigate
		phi, psi = Inputs.contour_ij
		index = next(
			i for i, d in enumerate(self.data)
			if (np.abs(d["phi"] - phi) < 1e-6) and (np.abs(d["psi"] - psi) < 1e-6)
		)

		# initial values for axis limits
		ymin = np.inf
		zmin = np.inf
		ymax = -np.inf
		zmax = -np.inf

		# loop for each blade row
		for blade_row in self.data[index]["blade_rows"]:

			# loop for both inlet and outlet
			for c in ("inlet", "outlet"):

				# update min-max values for axis limits
				ymin = min(ymin, np.min(blade_row[c]["y"]))
				zmin = min(zmin, np.min(blade_row[c]["z"]))
				ymax = max(ymax, np.max(blade_row[c]["y"]))
				zmax = max(zmax, np.max(blade_row[c]["z"]))

		corners = np.array([
			[ymin, zmin],
			[ymin, zmax],
			[ymax, zmin],
			[ymax, zmax]
		])

		# loop for each blade row
		for i, blade_row in enumerate(self.data[index]["blade_rows"]):

			# create side-by-side figure
			fig, axes = plt.subplots(1, 2, figsize = Inputs.figsize)

			# get inlet and outlet planes
			inlet = blade_row["inlet"]
			outlet = blade_row["outlet"]

			# find coordinates of centre-line through blade passage section
			"""j = int(np.floor(inlet["y"].shape[1] / 2))

			# generate rotated coordinates
			theta = self.rotate(inlet, j)

			# repeat for outlet
			theta = self.rotate(outlet, j)"""

			# get data lmits
			vmin = min(np.array(inlet[attribute]).min(), np.array(outlet[attribute]).min())
			vmax = max(np.array(inlet[attribute]).max(), np.array(outlet[attribute]).max())

			# set contour levels
			levels = np.linspace(vmin, vmax, Inputs.levels)
			
			# plot inlet slice
			cf1 = axes[0].contourf(
				inlet["y"], inlet["z"], inlet[attribute], levels = levels, cmap = "viridis"
			)
			
			# plot outlet slice and create colour bar
			cf2 = axes[1].contourf(
				outlet["y"], outlet["z"], outlet[attribute], levels = levels, cmap = "viridis"
			)
			cbar = fig.colorbar(cf2, ax = [axes[0], axes[1]])
			cbar.set_label(
				f"{attribute} ({Inputs.units[attribute] if attribute in Inputs.units else '-'})"
			)

			# calculate mass flux below which region is classed as a boundary layer
			m_dot = self.data[index]["metadata"]["mass_flow_rate"]
			d = self.data[index]["metadata"]["diameter"]
			HTR = self.data[index]["metadata"]["hub_tip_ratio"]
			A = np.pi * (d / 2)**2 * (1 - HTR**2)
			rovx = 0.95 * m_dot / A

			# add contour of approximate boundary layer thickness
			axes[0].plot(
				inlet["y_boundary_hub"], inlet["z_boundary_hub"], color = "k"
			)
			axes[0].plot(
				inlet["y_boundary_tip"], inlet["z_boundary_tip"], color = "k"
			)
			axes[1].plot(
				outlet["y_boundary_hub"], outlet["z_boundary_hub"], color = "k"
			)
			axes[1].plot(
				outlet["y_boundary_tip"], outlet["z_boundary_tip"], color = "k"
			)

			# set subplot titles
			axes[0].set_title(
				f"{'Rotor' if i % 2 == 0 else 'Stator'} {i // 2 + 1} Inlet",
				fontsize = Inputs.fontsize
			)
			axes[1].set_title(
				f"{'Rotor' if i % 2 == 0 else 'Stator'} {i // 2 + 1} Exit",
				fontsize = Inputs.fontsize
			)

			# set figure title
			fig.suptitle(
				rf"Contours of {label} for $\phi$ = {phi}, $\psi$ = {psi}", fontsize = Inputs.titlesize,
				y = 1.01
			)

			# loop for each subplot
			for ax in axes:

				# set aspect ratio to equal
				ax.set_aspect("equal")

				# remove x- and y-axis ticks
				#ax.axis("off")

			# save plot
			figname = (
				f"section_view_phi_{phi}_psi_{psi}_row_{i}_{attribute}".replace(".", "_")
			)
			fig.savefig(
				os.path.join(Inputs.folder, figname),
				dpi = Inputs.dpi, bbox_inches = "tight"
			)

			# close plot
			Inputs.saved_figures.append(f"{figname}.png")
			print(f"Plot saved as {utils.Colours.GREEN}{figname}.png{utils.Colours.END}!")
			plt.close()

	def plot_span(self, attribute, label = "", hold = False):
		"""Creates a plot of the spanwise variation of mass-averaged flow properties."""
		# get list of colours
		colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

		# get index of desired test case to investigate
		phi, psi = Inputs.contour_ij
		index = next(
			i for i, d in enumerate(self.data)
			if (np.abs(d["phi"] - phi) < 1e-6) and (np.abs(d["psi"] - psi) < 1e-6)
		)

		# store engine separately for convenience
		engine = self.data[index]["engine_design"]

		# check that there are no open figures
		if plt.get_fignums() == []:

			# create figure with one subplot per blade row + 1
			fig, axes = plt.subplots(ncols = len(self.data[index]["blade_rows"]) + 1, figsize = Inputs.figsize)

			# set colour counter to 0
			self.colour_index = 0

		# figure already exists
		else:

			# retrieve figure and axes object instances
			fig = plt.gcf()
			axes = fig.axes

			# increment colour counter
			self.colour_index += 1

		# check if attribute is contained within CFD data
		inlet = self.data[index]["blade_rows"][0]["inlet"]
		if f"{attribute}_avg" in inlet:

			# calculate dimensionless span
			span = (
				(inlet["r"] - np.min(inlet["r"]))
				/ (np.max(inlet["r"]) - np.min(inlet["r"]))
			)

			# plot rotor inlet
			axes[0].plot(
				inlet[f"{attribute}_avg"], span, color = colours[self.colour_index % len(colours)],
				label = f"{label} (CFD)"
			)

		# loop for each blade row
		for i, (blade_row, ax) in enumerate(zip(self.data[index]["blade_rows"], axes[1:])):

			# get outlet plane
			outlet = blade_row["outlet"]

			# check if attribute is contained within CFD data
			if f"{attribute}_avg" in outlet:

				# calculate dimensionless span
				span = (
					(outlet["r"] - np.min(outlet["r"]))
					/ (np.max(outlet["r"]) - np.min(outlet["r"]))
				)

				# plot spanwise variation of given property
				ax.plot(
					outlet[f"{attribute}_avg"], span, color = colours[self.colour_index % len(colours)],
					label = f"{label} (CFD)"
				)

				# plot boundary layer thicknesses
				ax.axhline(outlet["span_boundary_hub"], color = "k")
				ax.axhline(outlet["span_boundary_tip"], color = "k")

		# check if attribute is contained with design tool data
		if hasattr(engine.blade_rows[0].inlet, attribute):

			# given attribute is an angle
			if any(s in attribute for s in ["angle", "alpha" or "beta"]):

				# convert from radians to degrees
				zz = utils.rad_to_deg(getattr(engine.blade_rows[0].inlet, attribute))

			# all other properties
			else:

				# plot as is
				zz = getattr(engine.blade_rows[0].inlet, attribute)

			# calculate dimensionless span
			rr = engine.blade_rows[0].inlet.rr
			span = (rr - np.min(rr)) / (np.max(rr) - np.min(rr))

			# plot rotor inlet
			axes[0].plot(
				zz, span, linestyle = '--',
				color = colours[self.colour_index % len(colours)],
				label = f"{label} (Design)"
			)

		# loop for each blade row
		for i, (blade_row, ax) in enumerate(zip(engine.blade_rows, axes[1:])):

			# check if attribute is contained with design tool data
			if hasattr(blade_row.exit, attribute):

				# given attribute is an angle
				if any(s in attribute for s in ["angle", "alpha", "beta"]):

					# convert from radians to degrees
					zz = utils.rad_to_deg(getattr(blade_row.exit, attribute))

				# all other properties
				else:

					# plot as is
					zz = getattr(blade_row.exit, attribute)

				# calculate dimensionless span
				rr = blade_row.exit.rr
				span = (rr - np.min(rr)) / (np.max(rr) - np.min(rr))

				# plot spanwise variation of given property
				ax.plot(
					zz, span, linestyle = '--', color = colours[self.colour_index % len(colours)],
					label = f"{label} (Design)"
				)

		# figure is to be saved and closed
		if hold == False:

			# set axis labels
			axes[0].set_title("Compressor Inlet")

			# loop for all but first axis
			for i, ax in enumerate(axes[1:]):

				if i % 2 == 0:
					ax.set_title(f"Rotor {i // 2 + 1}")
				else:
					ax.set_title(f"Stator {i // 2 + 1}")

			# loop for each axis
			for ax in axes:

				# get boolean values for different criteria checking if axis is populated
				has_lines = len(ax.lines) > 0
				has_collections = len(ax.collections) > 0
				has_images = len(ax.images) > 0
				has_containers = len(ax.containers) > 0
				
				# If all are empty, delete the axis from the figure layout
				if not (has_lines or has_collections or has_images or has_containers):
					ax.remove()

			axes = np.array([ax for ax in axes if ax.figure is not None])

			# get axis limits
			xlims = [ax.get_xlim() for ax in axes]
			xmin = min(lim[0] for lim in xlims)
			xmax = max(lim[1] for lim in xlims)

			# container for storing legend handles and labels
			handles, labels = [], []

			# loop for each axis
			for i, ax in enumerate(axes):

				# get unique legend entries
				h, l = ax.get_legend_handles_labels()
				handles.extend(h)
				labels.extend(l)

				# set axis limits and grid
				ax.set_xlim(xmin, xmax)
				ax.set_ylim(0, 1)
				ax.grid()

				# all but LH axis
				if i > 0:

					# remove y-axis ticks
					ax.yaxis.set_tick_params(left=False, labelleft=False)

				# x-axis scale is very large
				if (xmax - xmin) > 1e3:

					# clip between (-1, 1)
					ax.set_xlim(-1, 1)

			# set y-axis label
			axes[0].set_ylabel("Dimensionless Span")

			# create legend underneath plot
			unique = dict(zip(labels, handles))
			plt.subplots_adjust(bottom = 0.1 * len(unique))
			axes[-1].legend(
				unique.values(), unique.keys(), fontsize = Inputs.fontsize,
				loc = "center", bbox_to_anchor = (0.5, 0.05), bbox_transform = fig.transFigure
			)

			# save plot
			figname = (
				f"plot_span_phi_{phi}_psi_{psi}_{attribute}".replace(".", "_")
			)
			fig.savefig(
				os.path.join(Inputs.folder, figname),
				dpi = Inputs.dpi, bbox_inches = "tight"
			)

			# close plot
			Inputs.saved_figures.append(f"{figname}.png")
			print(f"Plot saved as {utils.Colours.GREEN}{figname}.png{utils.Colours.END}!")
			plt.close()

	def quantify_loss(self):
		"""Creates a pie chart breaking down the relative contributions to the entropy loss function."""
		# get index of desired test case to investigate
		phi, psi = Inputs.contour_ij
		index = next(
			i for i, d in enumerate(self.data)
			if (np.abs(d["phi"] - phi) < 1e-6) and (np.abs(d["psi"] - psi) < 1e-6)
		)
		
		# retrieve relevant data object
		data = self.data[index]

		# ALTERNATIVE PLOT
		# calculate total entropy flux
		outlet = data["blade_rows"][-1]["outlet"]
		S_dot_total = np.sum(outlet["rovx_cell"] * outlet["s_cell"] * outlet["A"])

		# get entropy flux totals for each blade row
		S_dot_blade = [
			blade_row["outlet"]["s_flux_hub"] + blade_row["outlet"]["s_flux_midspan"]
			+ blade_row["outlet"]["s_flux_tip"]
			for blade_row in data["blade_rows"]
		]

		# list of chart values
		chart_values = [S_dot_total - np.sum(S_dot_blade)] + S_dot_blade
		chart_values = np.maximum(0, chart_values)
		chart_labels = ["Mixing"] + [f"Blade {i}" for i in range(len(data["blade_rows"]))]
		
		# create pie chart
		fig, ax = plt.subplots(figsize = utils.Defaults.figsize)

		# create pie chart
		wedges, texts, autotexts = ax.pie(
			chart_values, 
			labels = chart_labels, 
			autopct = '%1.2g%%',
			startangle = 140,
			wedgeprops = dict(width = 0.6, edgecolor = 'white', linewidth = 2)
		)

		# 1. Calculate entropy fluxes
		outlet_last = data["blade_rows"][-1]["outlet"]
		S_dot_total = np.sum(outlet_last["rovx_cell"] * outlet_last["s_cell"] * outlet_last["A"])

		# Track outer ring (Total per category) and inner ring (Sub-breakdown)
		outer_values = [0]
		outer_labels = ["Mixing"]
		inner_values = [0]
		inner_labels = ["Mixing"]

		# Colors: define a unified color palette for clean visual grouping
		colors_outer = ["#7f7f7f"]  # Gray for mixing
		# Example color families for blades: Blues for Blade 0, Oranges for Blade 1, etc.
		blade_color_maps = [["#1f77b4", "#aec7e8", "#17becf"], ["#ff7f0e", "#ffbb78", "#dbdb8d"]] 
		colors_inner = ["#a6a6a6"]  # Gray for inner mixing segment

		for i, blade_row in enumerate(data["blade_rows"]):
			h = blade_row["outlet"]["s_flux_hub"]
			m = blade_row["outlet"]["s_flux_midspan"]
			t = blade_row["outlet"]["s_flux_tip"]
			total_blade = h + m + t
			
			# Outer Ring Data
			outer_values.append(total_blade)
			outer_labels.append(f"Blade {i}")
			colors_outer.append(blade_color_maps[i % len(blade_color_maps)][0])
			
			# Inner Ring Data
			inner_values.extend([h, m, t])
			inner_labels.extend(["Hub", "Mid", "Tip"])
			colors_inner.extend(blade_color_maps[i % len(blade_color_maps)])

		# Calculate the actual mixing value for the outer ring
		mixing_value = S_dot_total - np.sum(outer_values[1:])
		outer_values[0] = mixing_value
		inner_values[0] = mixing_value

		# 2. Plotting the Nested Donut
		fig, ax = plt.subplots(figsize = utils.Defaults.figsize)

		# clip values
		outer_values = np.maximum(0, outer_values)
		inner_values = np.maximum(0, inner_values)

		# --- OUTER RING (Total Blade and Mixing) ---
		wedges_out, texts_out = ax.pie(
			outer_values, 
			labels = outer_labels, 
			radius = 1.0,
			colors = colors_outer,
			startangle = 140,
			wedgeprops = dict(width=0.3, edgecolor='white', linewidth=2),
			pctdistance = 0.85
		)

		# --- INNER RING (Hub, Midspan, Tip Breakdowns) ---
		wedges_in, texts_in, autotexts_in = ax.pie(
			inner_values, 
			labels = inner_labels,
			labeldistance = 0.55,
			radius = 0.7,
			colors = colors_inner,
			startangle = 140,
			autopct = '%1.0f%%',
			pctdistance = 0.35,
			wedgeprops=dict(width = 0.3, edgecolor = 'white', linewidth = 1)
		)

		# Clean up label sizes/formatting so they don't overlap
		plt.setp(autotexts_in, size = 8, weight = "bold")
		plt.setp(texts_in, size = 8)

		ax.set(aspect="equal")

		# save plot
		figname = (
			f"quantify_loss_{phi}_psi_{psi}".replace(".", "_")
		)
		fig.savefig(
			os.path.join(Inputs.folder, figname),
			dpi = Inputs.dpi, bbox_inches = "tight"
		)

		# close plot
		Inputs.saved_figures.append(f"{figname}.png")
		print(f"Plot saved as {utils.Colours.GREEN}{figname}.png{utils.Colours.END}!")
		plt.close()

	def generate_report(self):
		"""Generates a .pdf report of all generated plots relating to a given test case."""
		# create document
		doc = Document(geometry_options = {"margin": "1in"})
		doc.packages.append(Package("graphicx"))
		doc.packages.append(Package("float"))

		# loop for each saved figure
		for figure in Inputs.saved_figures:

			# create figure
			with doc.create(Figure(position = "H")) as fig:

				# add line to Latex doc with \includegraphics and specify caption
				fig.add_image(os.path.join(Inputs.folder, figure), width = NoEscape(r"0.9\textwidth"))
				fig.add_caption(f"{figure}")

		# compile as pdf
		phi, psi = Inputs.contour_ij
		file_name = f"report_phi_{phi}_psi_{psi}".replace(".", "_")
		doc.generate_pdf(
			os.path.join(Inputs.folder, file_name), clean_tex = False,
			compiler = "pdflatex"
		)

		# print user feedback and load report
		print(f"Output report saved as {utils.Colours.GREEN}{file_name}{utils.Colours.END}!")
		subprocess.run(["xdg-open", os.path.join(Inputs.folder, f"{file_name}.pdf")])

# main function
def main():

	# create post-processing object
	post = Post()

	# calculate post-processed values
	post.calc_secondary()
	post.analyse()
	post.calculate_thrust()		

	# create contour plots of efficiency and number of blades
	post.plot_contours("eta_poly", "Polytropic Efficiency", [0.7, 0.94, 13])
	post.plot_contours("no_of_blades", "No. of Blades")
	post.plot_contours("min_skewness_angle", "Min. Skewness Angle")
	post.plot_contours("C_th", r"Thrust Coefficient, $C_{\text{þ}}$", [0, 0.05])
	post.plot_contours("eta_overall", r"Overall Efficiency, $\eta_{\text{overall}}$", [0.6, 0.84, 13])
	post.plot_contours("eta_comp", r"Compressor Efficiency, $\eta_{\text{comp}}$", [0.7, 0.94, 13])
	post.plot_contours("eta_prop", r"Propulsive Efficiency, $\eta_{\text{prop}}$", [0.84, 1, 9])
	post.plot_contours("eta_swirl", r"Swirl Efficiency, $\eta_{\text{swirl}}$", [0.84, 1, 9])
	#post.plot_contours("max_expansion_ratio", "Max. Expansion Ratio")
	#post.plot_contours("n_aid", "Azimuthal Gridpoints")
	#post.plot_contours("max_diffusion_factor", "Max. Diffusion Factor")
	#post.plot_contours("max_deviation", "Max. Deviation (deg)")
	#post.plot_contours("max_incidence", "Max. Incidence (deg)")

	# phi-psi coordinates for a specific test case exist
	if Inputs.contour_ij is not None:

		# get index of desired test case to investigate
		"""phi, psi = Inputs.contour_ij
		index = next(
			i for i, d in enumerate(post.data)
			if (np.abs(d["phi"] - phi) < 1e-6) and (np.abs(d["psi"] - psi) < 1e-6)
		)"""

		# close any plots that might have been created
		plt.close("all")

		# produce section views
		#post.section_view("rorvt", "Tangential Momentum")
		#post.section_view("v_x", "Axial Velocity")
		post.section_view("M", "Mach Number")
		post.section_view("p_0", "Stagnation Pressure")
		post.section_view("T_0", "Stagnation Temperature")
		post.section_view("s", "Entropy")
		post.section_view("alpha", "Flow Angle (deg)")

		# produce spanwise variation plots
		post.plot_span("rovx", "Axial Momentum")

		post.plot_span("M", "Mach Number", hold = True)
		post.plot_span("M_rel", "Relative Mach Number")

		post.plot_span("p", "Pressure", hold = True)
		post.plot_span("p_0", "Stagnation Pressure")

		post.plot_span("T", "Temperature", hold = True)
		post.plot_span("T_0", "Stagnation Temperature")

		post.plot_span("alpha", "Flow Angle (deg)", hold = True)
		post.plot_span("beta", "Relative Flow Angle (deg)", hold = True)
		post.plot_span("metal_angle", "Blade Metal Angle (deg)")

		post.plot_span("diffusion_factor", "Diffusion Factor")

		post.plot_span("deviation", "Deviation (deg)", hold = True)
		post.plot_span("incidence", "Incidence (deg)")

		post.plot_span("s", "Entropy")
		post.plot_span("loss_function", "Entropy Loss Function")

		# create latex report with all plots produced
		post.generate_report()

# upon script execution
if __name__ == "__main__":

	# no additional input arguments are provided
	if len(sys.argv) < 2:

		# print error message
		print(f"{utils.Colours.RED}Please provide a data folder location!{utils.Colours.END}")

	# input folder is specified
	else:

		# store input folder
		Inputs.folder = os.path.join(Inputs.folder_path, sys.argv[1])
		print(f"Reading from folder: {utils.Colours.GREEN}{Inputs.folder}{utils.Colours.END}")

	# a specific test case is provided
	if len(sys.argv) > 2:

		# store phi-psi coordinates of test case
		Inputs.contour_ij = ast.literal_eval(sys.argv[2])
		print(f"Analysing test case with phi-psi coordinates: {utils.Colours.GREEN}{Inputs.contour_ij}{utils.Colours.END}")

	# no specific test case is provided
	else:

		# set to none
		Inputs.contour_ij = None

	# run main()
	main()
