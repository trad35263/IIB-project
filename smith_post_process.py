# smith_post_process.py
# 25 May 2026

# import modules
import numpy as np
import mat73
import itertools
import colorsys
from scipy.interpolate import CloughTocher2DInterpolator

# import matplotlib
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

#
from adjustText import adjust_text

# import system modules
import os
import subprocess
import ast
import copy

# import latex
from pylatex import Document, Figure, NoEscape, Package

# import high speed solver
import sys
from engine import Engine
from flight_scenario import Flight_scenario
from motor_database import Database
import utils

# override matplotlib default behaviour to render negative contours as dashed lines
import matplotlib as mpl
mpl.rcParams["contour.negative_linestyle"] = "solid"

# font manager
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
		"phi_local": r"\phi_\text{loc}"
	}

	# empty list of saved figures
	saved_figures = []

	# mass flow rate threshold for classification as a boundary layer
	BL_threshold = 0.1

	# dynamic viscosity
	mu = 1.716e-5				# (Pa s)

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

		# initialise variable for appending text to as a summary in the report
		self.text = ""

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
		# loop for each test case
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
					plane["s"] = (
						Inputs.c_p * np.log(T_0 / np.mean(data["blade_rows"][0]["inlet"]["T_0"]))
						- Inputs.R * np.log(p_0 / np.mean(data["blade_rows"][0]["inlet"]["p_0"]))
					)

					# rotors only
					if blade_row["rpm"] > 0:

						# get local flow coefficient, non-dimensionalised by blade tip speed
						plane["phi_local"] = (
							plane["v_x"]
							/ (blade_row["rpm"] * 2 * np.pi / 60 * plane["r"][-1])
						)

					# stators
					else:

						# set local flow coefficient to gi
						plane["phi_local"] = (
							plane["v_x"]
							/ (data["blade_rows"][i - 1]["rpm"] * 2 * np.pi / 60 * plane["r"][-1])
						)

	def analyse(self):
		"""Carries out preliminary analysis for each loaded test case."""
		# loop for each test case
		for data in self.data:

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
							#data["metadata"]["hub_tip_ratio"] * Inputs.BL_threshold
							Inputs.BL_threshold
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
						#>= data["metadata"]["hub_tip_ratio"] * Inputs.BL_threshold
						>= Inputs.BL_threshold
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

					# find mask for hub boundary layer
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
						* blade_row["outlet"]["A"] * blade_row["no_of_blades"]
						* blade_row["outlet"]["boundary_hub"]
					) - np.sum(
						blade_row["inlet"]["rovx_cell"] * blade_row["inlet"]["s_cell"]
						* blade_row["inlet"]["A"] * blade_row["no_of_blades"]
						* blade_row["inlet"]["boundary_hub"]
					)
				)

				# calculate tip entropy flux
				blade_row["outlet"]["s_flux_tip"] = (
					np.sum(
						blade_row["outlet"]["rovx_cell"] * blade_row["outlet"]["s_cell"]
						* blade_row["outlet"]["A"] * blade_row["no_of_blades"]
						* blade_row["outlet"]["boundary_tip"]
					) - np.sum(
						blade_row["inlet"]["rovx_cell"] * blade_row["inlet"]["s_cell"]
						* blade_row["inlet"]["A"] * blade_row["no_of_blades"]
						* blade_row["inlet"]["boundary_tip"]
					)
				)

				# calculate midspan entropy flux
				blade_row["outlet"]["s_flux_midspan"] = (
					np.sum(
						blade_row["outlet"]["rovx_cell"] * blade_row["outlet"]["s_cell"]
						* blade_row["outlet"]["A"] * blade_row["no_of_blades"]
						* (~blade_row["outlet"]["boundary_hub"] & ~blade_row["outlet"]["boundary_tip"])
					) - np.sum(
						blade_row["inlet"]["rovx_cell"] * blade_row["inlet"]["s_cell"]
						* blade_row["inlet"]["A"] * blade_row["no_of_blades"]
						* (~blade_row["inlet"]["boundary_hub"] & ~blade_row["inlet"]["boundary_tip"])
					)
				)

				# calculate entire blade row entropy flux
				blade_row["outlet"]["s_flux"] = (
					np.sum(
						blade_row["outlet"]["rovx_cell"] * blade_row["outlet"]["s_cell"]
						* blade_row["outlet"]["A"] * blade_row["no_of_blades"]
					) - np.sum(
						blade_row["inlet"]["rovx_cell"] * blade_row["inlet"]["s_cell"]
						* blade_row["inlet"]["A"] * blade_row["no_of_blades"]
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

			# calculate machine entropy flux
			data["s_flux"] = (
				np.sum(
					data["blade_rows"][-1]["outlet"]["rovx_cell"] * data["blade_rows"][-1]["outlet"]["s_cell"]
					* data["blade_rows"][-1]["outlet"]["A"] * data["blade_rows"][-1]["no_of_blades"]
				) - np.sum(
					data["blade_rows"][0]["inlet"]["rovx_cell"] * data["blade_rows"][0]["inlet"]["s_cell"]
					* data["blade_rows"][0]["inlet"]["A"] * data["blade_rows"][0]["no_of_blades"]
				)
			)

			# calculate mixing contribution to entropy flux
			data["s_flux_mixing"] = (
				data["s_flux"]
				- np.sum([blade_row["outlet"]["s_flux"] for blade_row in data["blade_rows"]])
			)

			# calculate machine entropy loss function
			data["loss_function"] = (
				data["metadata"]["T_0"] * data["s_flux"]
				/ np.sum([blade_row["power"] for blade_row in data["blade_rows"]])
			)

			data["loss_function"] = 1 / (1 + data["loss_function"])

	def calculate_Re(self):
		"""Calculates the Reynolds number of each test case."""
		# loop for each data object
		for data in self.data:

			# calculate design Re based on blade speed, engine diameter and inlet stagnation density
			data["design_Re"] = (
				data["metadata"]["p_0"] / (utils.R * data["metadata"]["T_0"])
				* utils.velocity_function(data["metadata"]["inlet_mach_number"])
				* np.sqrt(utils.c_p * data["metadata"]["T_0"])
				* data["metadata"]["diameter"] / Inputs.mu
			)

			# calculate Re based on blade speed, engine diameter and inlet stagnation density
			data["Re"] = (
				data["metadata"]["p_0"] / (utils.R * data["metadata"]["T_0"]) * np.mean(
					data["blade_rows"][0]["inlet"]["rovx_avg"]
					/ data["blade_rows"][0]["inlet"]["ro_avg"]
				) * data["metadata"]["diameter"] / Inputs.mu
			)

			if data["Re"] > 1500000:

				print(f"data['phi']: {data['phi']}")
				print(f"data['psi']: {data['psi']}")

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

			# calculate engine dimensional values
			engine.dimensional_values()

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

			# store input power as engine attribute
			engine.P_in = np.sum([blade_row["power"] for blade_row in data["blade_rows"]])

			# run detailed thrust breakdown calculations
			engine.calc_thrust()

			# store engine in data object
			data["engine"] = engine

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

			# Construct the Clough-Tocher interpolator object
			interp = CloughTocher2DInterpolator(
				points = np.column_stack((xx[mask], yy[mask])), 
				values = zz[index][mask]
			)

			# Evaluate on your grid
			grid = interp(xx_grid, yy_grid)

			# create plot
			fig, ax = plt.subplots(figsize = utils.Defaults.figsize)

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

				# evenly-spaced given limits
				levels = np.linspace(*limits)

			# add contours
			cs = ax.contour(
				xx_grid, 
				yy_grid, 
				np.ma.masked_invalid(grid), 
				levels = levels,
				cmap="viridis", 
				extend="both"
			)

			# add text labels directly onto the contour lines
			ax.clabel(cs, inline = True, fontsize = utils.Defaults.fontsize, fmt='%1.2g')

			# plot contours of interpolated data and individual datapoints
			cf = ax.contourf(
				xx_grid, yy_grid, np.ma.masked_invalid(grid), levels = levels, cmap = "viridis",
				extend = "both", alpha = Inputs.alpha
			)
			plt.colorbar(cf, ax = ax, label = label, extend = "both")
			ax.scatter(xx[mask], yy[mask], c = "k", s = 8, zorder = 5, label = "Datapoints")
			ax.scatter(xx[~mask], yy[~mask], c = "red", s = 8, zorder = 5)

			# configure plot
			ax.set_xlabel(r"Flow Coefficient, $\phi$")
			ax.set_ylabel(r"Stage Loading Coefficient, $\psi$")

			# construct title
			title = (
				"Smith Chart for "
				rf"$N$ = {self.data[0]['metadata']['no_of_stages']}, "
				rf"$M_1$ = {self.data[0]['metadata']['inlet_mach_number']}"
			)
			if zz.shape[0] > 1:

				# append blade row to title if contour plot is specific to one row
				title += f"\n{'Rotor' if index % 2 == 0 else 'Stator'} {index // 2 + 1}"

			# set title
			ax.set_title(title, fontsize = Inputs.titlesize)

			# configure plot
			ax.set_xlim(0.4, 1)
			ax.set_ylim(0.05, 0.6)

			# save plot
			figname = (
				f"plot_contours_row_{index}_{attribute}".replace(".", "_")
			)
			fig.savefig(
				os.path.join(Inputs.folder, figname),
				dpi = utils.Defaults.dpi, bbox_inches = "tight"
			)

			# close plot
			Inputs.saved_figures.append(f"{figname}.png")
			print(f"Plot saved as {utils.Colours.GREEN}{figname}.png{utils.Colours.END}!")
			plt.close()

			# update summary text
			self.text += f"{attribute.replace('_', ' ')}: {np.nanmin(zz):.4g} $\\to$ {np.nanmax(zz):.4g} \\\\ "

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

	def section_view(self, attribute, label = "", symbol = ""):
		"""Creates a plot of a section view through the compressor."""
		# get index of desired test case to investigate
		phi, psi = Inputs.contour_ij
		index = next(
			i for i, d in enumerate(self.data)
			if (np.abs(d["phi"] - phi) < 1e-6) and (np.abs(d["psi"] - psi) < 1e-6)
		)

		# get relevant data object
		data = self.data[index]

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

		# loop for each blade row
		for i, blade_row in enumerate(data["blade_rows"]):

			# create side-by-side figure
			fig, axes = plt.subplots(1, 2, figsize = utils.Defaults.figsize)

			# get inlet and outlet planes
			inlet = blade_row["inlet"]
			outlet = blade_row["outlet"]

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
			cbar.set_label(symbol)

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
				fontsize = utils.Defaults.fontsize
			)
			axes[1].set_title(
				f"{'Rotor' if i % 2 == 0 else 'Stator'} {i // 2 + 1} Exit",
				fontsize = utils.Defaults.fontsize
			)

			# set figure title
			fig.suptitle(
				(
					rf"Contours of {label}, {symbol}" "\n"
					rf"$N$ = {data['metadata']['no_of_stages']}, "
					rf"$M_1$ = {data['metadata']['inlet_mach_number']}, $\phi$ = {phi}, "
					rf"$\psi$ = {psi}"
				),
				fontsize = Inputs.titlesize,
				y = 1.01
			)

			# loop for each subplot
			for ax in axes:

				# set aspect ratio to equal
				ax.set_aspect("equal")

				# remove x- and y-axis ticks
				ax.axis("off")

			# save plot
			figname = (
				f"section_view_phi_{phi}_psi_{psi}_row_{i}_{attribute}".replace(".", "_")
			)
			fig.savefig(
				os.path.join(Inputs.folder, figname),
				dpi = utils.Defaults.dpi, bbox_inches = "tight"
			)

			# close plot
			Inputs.saved_figures.append(f"{figname}.png")
			print(f"Plot saved as {utils.Colours.GREEN}{figname}.png{utils.Colours.END}!")
			plt.close()

	def rotor_section(self):
		"""Plots a section view through the rotor mean-line and tip showing contours of M_rel."""
		# get index of desired test case to investigate
		phi, psi = Inputs.contour_ij
		index = next(
			i for i, d in enumerate(self.data)
			if (np.abs(d["phi"] - phi) < 1e-6) and (np.abs(d["psi"] - psi) < 1e-6)
		)

		# get relevant data object
		data = self.data[index]
	
		# create plot
		fig, axes = plt.subplots(1, 2, figsize = utils.Defaults.figsize)

		# set min-max and levels
		min = 0
		max = 1.3
		levels = np.linspace(min, max, 201)

		# define colour map of red and blue
		blues = cm.get_cmap("Blues_r", 100)(np.linspace(0.4, 0.8, 100))
		reds = cm.get_cmap("Reds", 100)(np.linspace(0.2, 0.6, 100))
		custom_colours = np.vstack((blues, reds))
		colour_map = mcolors.ListedColormap(custom_colours)
		norm = mcolors.TwoSlopeNorm(vcenter = 1, vmin = min, vmax = max)

		# loop for both plots
		for (section, ax) in zip(("section1", "section2"), axes):

			# colour convex hull of selected mesh block in black
			g = data[section]["g5"]
			x = np.ravel(g["x"])
			z = np.ravel(g["z"])
			points = np.column_stack((x, z))
			hull = ConvexHull(points)
			vertices = points[hull.vertices]
			poly = Polygon(vertices, facecolor="black", edgecolor="black", zorder=1)
			ax.add_patch(poly)

			# loop for each mesh block
			for key, g in data[section].items():

				# plot contours of M_rel
				cf = ax.contourf(g["x"], g["z"], g["M_rel"], levels = levels, cmap = colour_map, norm = norm)

			# set axis title
			r_tip = 0.5 * data["metadata"]["diameter"]
			r_hub = 0.5 * data["metadata"]["diameter"] * data["metadata"]["hub_tip_ratio"]
			span = (
				(data[section][key]["r"] - r_hub)
				/ (r_tip - r_hub)
			)
			ax.set_title(f"{100 * span:.2g}% Span", fontsize = utils.Defaults.fontsize)

			# configure plot
			ax.axis("off")
			ax.set_aspect("equal")
			ax.set_xlim(0.045, 0.065)
			ax.set_ylim(-0.005, 0.015)

		# add colour bar
		cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.6, pad=0.05)
		cbar.set_label("Relative Mach Number, $M_{rel}$")
		cbar.set_ticks([min, 1, max])

		# set figure title
		N = data["metadata"]["no_of_stages"]
		M_1 = data["metadata"]["inlet_mach_number"]
		fig.suptitle(
			r"Rotor Blade Section View, Contours of $M_\text{rel}$" + "\n"
			rf"$N$ = {N:.4g}, $M_1$ = {M_1:.4g}, $\phi$ = {phi:.4g}, $\psi$ = {psi:.4g}",
			y = 1.02, fontsize = utils.Defaults.titlesize
		)

		# save plot
		figname = (
			f"rotor_section".replace(".", "_")
		)
		fig.savefig(
			os.path.join(Inputs.folder, figname),
			dpi = utils.Defaults.dpi, bbox_inches = "tight"
		)

		# close plot
		Inputs.saved_figures.append(f"{figname}.png")
		print(f"Plot saved as {utils.Colours.GREEN}{figname}.png{utils.Colours.END}!")
		plt.close()

		plt.show()

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
			fig, all_axes = plt.subplots(
				nrows = 2,
				ncols = len(self.data[index]["blade_rows"]) + 1,
				figsize = utils.Defaults.figsize,
				#layout = "constrained",
				gridspec_kw = {"height_ratios": [0.9, 0.1]}
			)

			# separate into axes for plotting and for legend
			all_axes = np.asarray(all_axes)
			axes = all_axes[0, :]
			legend_ax = all_axes[1, 0]

			# for all legend axes
			for ax in all_axes[1, :]:

				# hide axes
				ax.axis("off")

			# set colour counter to 0
			self.colour_index = 0

		# figure already exists
		else:

			# retrieve figure and axes object instances
			fig = plt.gcf()
			all_axes = fig.axes

			# separate into axes for plotting and for legend
			num_cols = len(self.data[index]["blade_rows"]) + 1
			all_axes = np.asarray(all_axes).reshape(2, num_cols)
			axes = all_axes[0, :]
			legend_ax = all_axes[1, 0]

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
				if any(s in attribute for s in ["angle", "alpha", "beta", "deviation"]):

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
			axes[0].set_title("Fan Inlet")

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
				
				# if all are empty, delete the axis from the figure layout
				if not (has_lines or has_collections or has_images or has_containers):
					ax.remove()

			# update list of axes
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
					ax.yaxis.set_tick_params(left = False, labelleft = False)

			# set y-axis label
			axes[0].set_ylabel("Dimensionless Span")

			# create legend underneath plot
			unique = dict(zip(labels, handles))
			legend_ax.legend(
                unique.values(), 
                unique.keys(), 
                fontsize = utils.Defaults.fontsize,
                ncol = 2,
				loc = "upper center",
				bbox_to_anchor = (0.5, 0.1),
				bbox_transform = fig.transFigure,
                frameon = False
            )
			"""axes[-1].legend(
				unique.values(), unique.keys(), fontsize = utils.Defaults.fontsize,
				loc = "center", bbox_to_anchor = (1 + 0.03 * len(axes), 0.5),
				bbox_transform = fig.transFigure, frameon = False
			)"""

			# save plot
			figname = (
				f"plot_span_phi_{phi}_psi_{psi}_{attribute}".replace(".", "_")
			)
			fig.savefig(
				os.path.join(Inputs.folder, figname),
				dpi = utils.Defaults.dpi, bbox_inches = "tight"
			)

			# close plot
			Inputs.saved_figures.append(f"{figname}.png")
			print(f"Plot saved as {utils.Colours.GREEN}{figname}.png{utils.Colours.END}!")
			plt.close()

	def loss_breakdown(self):
		"""Creates a pie chart breaking down the relative contributions to the entropy loss function."""
		# get index of desired test case to investigate
		phi, psi = Inputs.contour_ij
		index = next(
			i for i, d in enumerate(self.data)
			if (np.abs(d["phi"] - phi) < 1e-6) and (np.abs(d["psi"] - psi) < 1e-6)
		)
		
		# retrieve relevant data object
		data = self.data[index]

		# create plot
		fig, ax = plt.subplots(figsize = utils.Defaults.figsize)

		# set pie chart radius and add circle
		R = 1.6
		my_circle = plt.Circle((0, 0), R, edgecolor = 'none', facecolor='grey')
		ax.add_patch(my_circle)

		# get list of values for outer pie
		outer_values = (
			[blade_row["outlet"]["s_flux"] for blade_row in data["blade_rows"]]
			+ [data["s_flux_mixing"]]
		)

		# prepare lists of values and labels for inner pie and loop for each blade row
		inner_values = []
		for blade_row in data["blade_rows"]:

			# append hub, mid-span and tip values and label
			inner_values.extend([
				blade_row["outlet"]["s_flux_hub"],
				blade_row["outlet"]["s_flux_midspan"],
				blade_row["outlet"]["s_flux_tip"],
			])

		inner_values.append(data["s_flux_mixing"])

		# clip values - hub/midspan/tip values can be -ve
		outer_values = np.maximum(0, outer_values)
		inner_values = np.maximum(0, inner_values)

		# non-dimensionalise values
		outer_values = outer_values / np.sum(outer_values)
		inner_values = inner_values / np.sum(inner_values)

		# get list of labels for outer pie
		outer_labels = []
		for i, blade_row in enumerate(data["blade_rows"]):

			# even indices are rotors
			if i % 2 == 0:

				# append rotor label
				outer_labels.append(f"Rotor {i // 2 + 1}\n{100 * outer_values[i]:.1f}%")

			# odd indices are stators
			else:

				# append stator label
				outer_labels.append(f"Stator {i // 2 + 1}\n{100 * outer_values[i]:.1f}%")

		# add mixing label
		outer_labels.append(f"Mixing\n{100 * outer_values[-1]:.2g}%")

		# get cycle of default colours + grey for mixing loss
		default_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
		colours = [default_colours[i % len(default_colours)] for i in range(len(data["blade_rows"]))]
		colours.append("none")

		# list of all text objects
		all_texts = []

		# plot outer ring
		_, texts = ax.pie(
			outer_values, 
			labels = outer_labels, 
			radius = R,
			wedgeprops = dict(width = R / 2, edgecolor = "white", linewidth = 0.5),
			textprops = dict(horizontalalignment = "center"),
			labeldistance = 1.3,
			startangle = 90,
			counterclock = False,
			colors = colours
		)
		all_texts.extend(texts)

		# get new colours cycle
		default_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
		colour_cycle = itertools.cycle(default_colours)
		base_colours = [next(colour_cycle) for _ in range(len(data["blade_rows"]))]
		
		# create empty list of colours and loop for each base colour
		colours = []
		for colour in base_colours:

			# convert colour to rgb
			rgb = mcolors.to_rgb(colour)

			# convert rgb to hue, lightness, saturation
			h, l, s = colorsys.rgb_to_hls(*rgb)

			# loop for 3 new lightness values required
			for i in [0.25, 0.5, 0.75]:

				# append lighter colour
				colours.append(colorsys.hls_to_rgb(h, l + (1 - l) * i, s))

		# append none to have the mixing loss not appear
		colours.append("none")

		# plot inner ring
		_, texts = ax.pie(
			inner_values,
			radius = R / 2,
			wedgeprops = dict(width = R / 2, edgecolor = "white", linewidth = 0.5),
			textprops = dict(horizontalalignment = "center"),
			startangle = 90,
			counterclock = False,
			colors = colours
		)
		all_texts.extend(texts)

		labels = ["Hub", "Midspan", "Tip"]

		# reshape lists of values and colours
		values = np.reshape(inner_values[:-1], (-1, 3))
		colours = np.reshape(colours[:-1], (-1, 3, 3))

		# loop for each blade row
		for i, (value, colour) in enumerate(zip(values, colours)):

			# normalise values
			value = value / np.sum(value)

			# append percentage values to labels
			label = [f"{l}\n{100 * v:.1f}%" for (l, v) in zip(labels, value)]

			# find centre position
			centre = (6 * np.cos(np.pi / 4 * (1 - 2 * i)), 2 * np.sin(np.pi / 4 * (1 - 2 * i)))

			# add mini pie chart detailing blade row loss breakdown
			_, texts = ax.pie(
				value,
				labels = label,
				radius = 0.5,
				center = centre,
				wedgeprops = dict(width = 0.5, edgecolor = "white", linewidth = 0.5),
				textprops = dict(horizontalalignment = "center"),
				labeldistance = 1.8,
				startangle = 90,
				counterclock = False,
				colors = colour
			)
			all_texts.extend(texts)

		# configure plot
		ax.set_aspect("equal")
		ax.set_xlim(-4, 4)
		ax.set_ylim(-2, 2)
		ax.axis("off")

		# figure title
		fig.suptitle(
			rf"Relative Contributions to Entropy Loss Function, $\xi$" + "\n"
			rf"$N$ = {data['metadata']['no_of_stages']}, $M_1$ = "
			rf"{data['metadata']['inlet_mach_number']}, $\phi$ = {phi}, $\psi$ = {psi}" + "\n"
			rf"$\eta_\text{{poly}}$ = {data['eta_poly']:.4g}",
			y = 1.2, fontsize = utils.Defaults.titlesize
		)

		# save plot
		figname = (
			f"loss_breakdown_{phi}_psi_{psi}".replace(".", "_")
		)
		fig.savefig(
			os.path.join(Inputs.folder, figname),
			dpi = utils.Defaults.dpi, bbox_inches = "tight"
		)

		# close plot
		Inputs.saved_figures.append(f"{figname}.png")
		print(f"Plot saved as {utils.Colours.GREEN}{figname}.png{utils.Colours.END}!")
		plt.close()

	def thrust_breakdown(self):
		"""Creates a horizontal bar chart detailing the losses to flight power."""
		# get index of desired test case to investigate
		phi, psi = Inputs.contour_ij
		index = next(
			i for i, d in enumerate(self.data)
			if (np.abs(d["phi"] - phi) < 1e-6) and (np.abs(d["psi"] - psi) < 1e-6)
		)

		# bar half-width
		width = 0.4

		# get relevant data and engine objects
		data = self.data[index]
		engine = data["engine"]

		print(f"engine.eta_overall: {engine.eta_overall}")
		print(f"engine.eta_thrust: {engine.eta_thrust}")
		print(f"engine.eta_poly: {engine.eta_poly}")
		print(f"engine.eta_prop: {engine.eta_prop}")
		print(f"engine.eta_swirl: {engine.eta_swirl}")

		# create plot
		fig, ax = plt.subplots(figsize = utils.Defaults.figsize)
		
		# plot polytropic efficiency
		ax.barh([1], engine.eta_poly, color = "C1", alpha = 0.7)
		ax.text(
			0.5 * engine.eta_poly, 1 - 5e-2, rf"$\eta_\text{{poly}} = {engine.eta_poly:.3g}$",
			fontsize = utils.Defaults.titlesize
		)

		# plot overall efficiency
		ax.barh(
			[0], engine.eta_overall,
			label = f"eta_overall: {engine.eta_overall:.3g}",
			color = "C0", alpha = 0.7
		)
		ax.text(
			0.5 * engine.eta_overall, -5e-2, rf"$\eta_\text{{overall}} = {engine.eta_overall:.3g}$",
			fontsize = utils.Defaults.titlesize
		)

		# plot breakdown of loss
		ax.barh(
			[0], engine.eta_thrust - engine.eta_overall, left = engine.eta_overall,
			linewidth = 0.5,
			label = f"eta_thrust: {engine.eta_thrust:.3g}",
			color = (1, 0.7, 0.7)
		)
		ax.barh(
			[0], engine.eta_poly - engine.eta_thrust, left = engine.eta_thrust,
			linewidth = 0.5,
			label = f"eta_poly: {engine.eta_poly:.3g}",
			color = (0.8, 0, 0)
		)
		ax.barh(
			[0], engine.eta_prop * (1 - engine.eta_poly),
			linewidth = 0.5,
			left = engine.eta_poly,
			label = f"eta_prop: {engine.eta_prop:.3g}",
			color = (0.9, 0.5, 0.5)
		)
		ax.barh(
			[0], 1 - engine.eta_poly - engine.eta_prop * (1 - engine.eta_poly),
			linewidth = 0.5,
			left = engine.eta_poly + engine.eta_prop * (1 - engine.eta_poly),
			label = f"eta_swirl: {engine.eta_swirl:.3g}",
			color = (0.8, 0.35, 0.35)
		)

		# list of line vertical heights
		heights = [-1, 1.6, 2.2, -1.6]

		# text left-align position
		x1 = 0.4
		x2 = 0.5 * (engine.eta_poly + 1)
		x3 = 1.05

		# add vertical lines to each loss component
		ax.vlines(
			[engine.eta_overall, engine.eta_thrust],
			heights[0], width, linewidth = 0.5, color = "k"
		)
		ax.vlines(
			[engine.eta_thrust, engine.eta_poly],
			-width, heights[1], linewidth = 0.5, color = "k"
		)
		ax.vlines(
			[engine.eta_poly, engine.eta_poly + engine.eta_prop * (1 - engine.eta_poly)],
			-width, heights[2], linewidth = 0.5, color = "k"
		)
		ax.vlines(
			[engine.eta_poly + engine.eta_prop * (1 - engine.eta_poly), 1],
			heights[3], width, linewidth = 0.5, color = "k"
		)

		# add arrows to each loss component
		ax.annotate(
			"",
			xy = (x1, heights[0]), 
			xytext = (engine.eta_overall, heights[0]),
			arrowprops = dict(arrowstyle = "<|-", color = "k", linewidth = 1, shrinkA = 0, shrinkB = 0)
		)
		ax.annotate(
			"",
			xy = (x2, heights[0]), 
			xytext = (engine.eta_thrust, heights[0]),
			arrowprops = dict(arrowstyle = "<|-", color = "k", linewidth = 1, shrinkA = 0, shrinkB = 0)
		)
		ax.annotate(
			"", 
			xy = (x1, heights[1]), 
			xytext = (engine.eta_thrust, heights[1]),
			arrowprops = dict(arrowstyle = "<|-", color = "k", linewidth = 1, shrinkA = 0, shrinkB = 0)
		)
		ax.annotate(
			"", 
			xy = (x2, heights[1]), 
			xytext = (engine.eta_poly, heights[1]),
			arrowprops = dict(arrowstyle = "<|-", color = "k", linewidth = 1, shrinkA = 0, shrinkB = 0)
		)
		ax.annotate(
			"", 
			xy = (x1, heights[2]), 
			xytext = (engine.eta_poly, heights[2]),
			arrowprops = dict(arrowstyle = "<|-", color = "k", linewidth = 1, shrinkA = 0, shrinkB = 0)
		)
		ax.annotate(
			"", 
			xy = (x3, heights[2]), 
			xytext = (engine.eta_poly + engine.eta_prop * (1 - engine.eta_poly), heights[2]),
			arrowprops = dict(arrowstyle = "<|-", color = "k", linewidth = 1, shrinkA = 0, shrinkB = 0)
		)
		ax.annotate(
			"", 
			xy = (x1, heights[3]), 
			xytext = (engine.eta_poly + engine.eta_prop * (1 - engine.eta_poly), heights[3]),
			arrowprops = dict(arrowstyle = "<|-", color = "k", linewidth = 1, shrinkA = 0, shrinkB = 0)
		)
		ax.annotate(
			"", 
			xy = (x3, heights[3]), 
			xytext = (1, heights[3]),
			arrowprops = dict(arrowstyle = "<|-", color = "k", linewidth = 1, shrinkA = 0, shrinkB = 0)
		)

		# add text to arrows
		offset = 0.08
		ax.text(x1, heights[0] + offset, f"Propulsive loss {100 * engine.eta_prop * (1 - engine.eta_poly):.2g}%")
		ax.text(x1, heights[1] + offset, f"Non-uniformity loss {100 * (engine.eta_poly - engine.eta_thrust):.2g}%")
		ax.text(x1, heights[2] + offset, f"Fan irreversibility {100 * (engine.eta_poly - engine.eta_overall):.2g}%")
		ax.text(
			x1, heights[3] + offset,
			f"Loss due to swirl {100 * (1 - (engine.eta_poly + engine.eta_prop * (1 - engine.eta_poly))):.2g}%"
		)

		# configure plot
		ax.set_xlim(0, x3)
		ax.set_ylim(-2, 3)
		ax.set_xlabel(r"Contribution to Engine Efficiency, $\eta$")

		# remove top and right-hand sides of bounding box
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)

		# move the remaining left and bottom spines to the origin
		ax.spines['left'].set_position('zero')
		ax.spines['bottom'].set_position(('data', -2))

		# remove all ticks and tick labels
		ax.set_xticks([0, 1])
		ax.set_yticks([])

		# set title
		ax.set_title(
			rf"Thrust Power Breakdown" + "\n"
			rf"$N$ = {data['metadata']['no_of_stages']}, "
			rf"$M_1$ = {data['metadata']['inlet_mach_number']}, $\phi$ = {phi:.4g}, "
			rf"$\psi$ = {psi:.4g}"
		)

		# save plot
		figname = (
			f"thrust_breakdown_phi_{phi}_psi_{psi}".replace(".", "_")
		)
		fig.savefig(
			os.path.join(Inputs.folder, figname),
			dpi = utils.Defaults.dpi, bbox_inches = "tight"
		)

		# close plot
		Inputs.saved_figures.append(f"{figname}.png")
		print(f"Plot saved as {utils.Colours.GREEN}{figname}.png{utils.Colours.END}!")
		plt.close()

	def summary(self, attribute, label = ""):
		"""Creates a summary plot of all test cases."""
		# create plot
		fig, ax = plt.subplots(figsize = utils.Defaults.figsize)

		# get list of polytropic efficiency values for plot x-axis
		xx = np.array([data["eta_poly"] for data in self.data])

		# get y-axis data
		yy = np.array([
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

		# get unique phi values
		phi = np.unique([data["phi"] for data in self.data])
		phi = np.unique(phi)

		# loop for each unique phi-value
		for p in phi:

			# mask only the indexes matching the specific phi
			mask = (np.array([data["phi"] for data in self.data]) == p)
		
			# Plot just this group
			ax.plot(
				xx[mask], 
				yy[mask], 
				marker = ".", 
				markersize = 8, 
				linestyle = "",
				label = f"$\phi$: {p}"
			)

		# configure plot
		ax.set_xlabel(r"Polytropic Efficiency, $\eta_\text{poly}$")
		ax.set_ylabel(label if label else attribute)
		ax.legend()

		# save plot
		figname = (
			f"summary_{attribute}".replace(".", "_")
		)
		fig.savefig(
			os.path.join(Inputs.folder, figname),
			dpi = utils.Defaults.dpi, bbox_inches = "tight"
		)

		# close plot
		Inputs.saved_figures.append(f"{figname}.png")
		print(f"Plot saved as {utils.Colours.GREEN}{figname}.png{utils.Colours.END}!")
		plt.close()

	def generate_report(self):
		"""Generates a .pdf report of all generated plots relating to a given test case."""
		# get index of desired test case to investigate
		phi, psi = Inputs.contour_ij
		index = next(
			i for i, d in enumerate(self.data)
			if (np.abs(d["phi"] - phi) < 1e-6) and (np.abs(d["psi"] - psi) < 1e-6)
		)

		# get relevant data object
		data = self.data[index]

		# create document
		doc = Document(geometry_options = {"margin": "1in"})
		doc.packages.append(Package("graphicx"))
		doc.packages.append(Package("float"))

		# helper function
		def add(string):

			# adds a string to the document with an endline appended
			doc.append(NoEscape(f"{string} \\\\ "))

		# add summary text
		add(f"Design Re: {data['design_Re']:.4g}")
		add(f"Re: {data['Re']:.4g}")
		add(f"Design thrust: {data['engine_design'].thrust:.4g} N")
		add(f"Thrust: {data['engine'].thrust:.4g} N")

		# add collected text
		add(self.text)

		# add some space
		doc.append(NoEscape(r"\vspace{2cm}"))

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

	def motors(self):
		""""""
		# create plot
		fig, ax = plt.subplots(figsize = utils.Defaults.figsize)

		# load motors database
		database = Database()
		print(f"database: {database}")

		for motor in database.motors:

			motor["area"] = (
				np.pi * (motor["diameter_mm"] * 1e-3 / 2)**2
				* (1 / self.data[0]["metadata"]["hub_tip_ratio"]**2 - 1)
			)

		# plot motor power / area against area
		ax.plot(
			[motor["area"] for motor in database.motors],
			[
				motor["max_power_W"] / motor["area"]
				for motor in database.motors
			],
			marker = ".", linestyle = "", color = "C3",
			label = "NeuMotors Database"
		)

		# loop for each data
		for data in self.data:

			data["metadata"]["area"] = (
				np.pi * (data["metadata"]["diameter"] / 2)**2
				* (1 - data["metadata"]["hub_tip_ratio"]**2)
			)

			data["power"] = np.sum([blade_row["power"] for blade_row in data["blade_rows"]])

		ax.plot(
			[data["metadata"]["area"] for data in self.data],
			[
				np.sum([blade_row["power"] for blade_row in data["blade_rows"]])
				/ data["metadata"]["area"]
				for data in self.data
			],
			marker = ".", linestyle = "", color = "C0",
			label = "CFD Data"
		)

		# configure plot
		ax.set_xlabel("Engine area (m^2)")
		ax.set_ylabel("Power per area (W / m^2)")
		ax.legend()
		ax.grid()

		plt.show()

# main function
def main():

	# create post-processing object
	post = Post()

	#post.motors()

	# calculate post-processed values
	post.calc_secondary()
	post.analyse()
	post.calculate_Re()
	post.calculate_thrust()

	# close any plots that might have been created
	plt.close("all")

	# create summary plots
	post.summary("eta_overall", r"Overall Efficiency, $\eta_\text{overall}$")

	# create contour plots of efficiency and number of blades
	#post.plot_contours("eta_poly", "Polytropic Efficiency", [0.7, 0.94, 13])
	post.plot_contours("no_of_blades", "No. of Blades")
	#post.plot_contours("min_skewness_angle", "Min. Skewness Angle")
	post.plot_contours("C_th", r"Thrust Coefficient, $C_{\text{þ}}$", [0, 0.05])
	#post.plot_contours("eta_overall", r"Overall Efficiency, $\eta_{\text{overall}}$", [0.6, 0.84, 13])
	post.plot_contours("eta_thrust", r"Thrust-averaged Polytropic Efficiency, $\eta_{\text{thrust}}$", [0.7, 0.94, 13])
	post.plot_contours("eta_poly", r"Polytropic Efficiency, $\eta_{\text{poly}}$", [0.7, 0.94, 13])
	post.plot_contours("eta_prop", r"Propulsive Efficiency, $\eta_{\text{prop}}$", [0.84, 1, 9])
	post.plot_contours("eta_swirl", r"Swirl Efficiency, $\eta_{\text{swirl}}$", [0.84, 1, 9])

	# phi-psi coordinates for a specific test case exist
	if Inputs.contour_ij is not None:

		# rotor relative Mach number plot
		#post.rotor_section()

		# produce loss breakdown
		#post.loss_breakdown()

		# produce thrust breakdown
		post.thrust_breakdown()

		return

		# produce section views
		post.section_view("phi_local", "Local Flow Coefficient", r"$\phi_\text{loc}$")
		post.section_view("rovx", "Axial Momentum")
		post.section_view("p_0", "Stagnation Pressure")
		post.section_view("s", "Entropy")
		post.section_view("alpha", "Flow Angle (°)")

		# produce spanwise variation plots
		post.plot_span("rovx", "Axial Momentum")

		post.plot_span("M", "Mach Number", hold = True)
		post.plot_span("M_rel", "Relative Mach Number")

		post.plot_span("p", "Pressure", hold = True)
		post.plot_span("p_0", "Stagnation Pressure")

		post.plot_span("T", "Temperature", hold = True)
		post.plot_span("T_0", "Stagnation Temperature")

		post.plot_span("alpha", "Flow Angle (°)", hold = True)
		post.plot_span("beta", "Relative Flow Angle (°)", hold = True)
		post.plot_span("metal_angle", "Blade Metal Angle (°)")

		post.plot_span("diffusion_factor", "Diffusion Factor")

		post.plot_span("deviation", "Deviation (°)", hold = True)
		post.plot_span("incidence", "Incidence (°)")

		post.plot_span("s", "Entropy")

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
