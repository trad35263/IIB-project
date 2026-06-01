# compare_reynolds.py
# 31 May 2026

# import modules
import numpy as np
import matplotlib.pyplot as plt
import os
import utils
from smith_post_process import Post
from smith_post_process import Inputs

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

# main function
def main():
	
	# set inputs folder to M = 0.1 cases
	Inputs.folder = os.path.join(Inputs.folder_path, "N_1_M_0.1")

	# create post-processing object
	post_1 = Post()
	post_1.calc_secondary()
	post_1.analyse()
	post_1.calculate_Re()
	
	# set inputs folder to M = 0.1 cases
	Inputs.folder = os.path.join(Inputs.folder_path, "N_1_M_0.25")

	# create post-processing object
	post_2 = Post()
	post_2.calc_secondary()
	post_2.analyse()
	post_2.calculate_Re()

	# get list of mutual phi and psi values
	pairs_1 = {
		(d["phi"].item(), d["psi"].item())
		for d in post_1.data
		if "phi" in d and "psi" in d
	}
	pairs_2 = {
		(d["phi"].item(), d["psi"].item())
		for d in post_2.data
		if "phi" in d and "psi" in d
	}
	pairs_set = pairs_1 & pairs_2
	pairs_list = list(pairs_set)

	# loop for each pair
	for pair in pairs_list:

		# get index of test case
		phi, psi = pair

		# manually specify
		phi, psi = 0.7, 0.35
		index = next(
			i for i, d in enumerate(post_1.data)
			if (np.abs(d["phi"] - phi) < 1e-3) and (np.abs(d["psi"] - psi) < 1e-3)
		)
		data_1 = post_1.data[index]
		index = next(
			i for i, d in enumerate(post_2.data)
			if (np.abs(d["phi"] - phi) < 1e-3) and (np.abs(d["psi"] - psi) < 1e-3)
		)
		data_2 = post_2.data[index]

		# create plot for rotor
		fig, axes = plt.subplots(1, 2, figsize = utils.Defaults.figsize, sharex = True, sharey = True)

		# get rotor exit planes
		outlet_1 = data_1["blade_rows"][0]["outlet"]
		outlet_2 = data_2["blade_rows"][0]["outlet"]

		# get max. flow coefficient
		phi_max = np.nanmax(outlet_1["phi_local"])
		phi_max = max(phi_max, np.nanmax(outlet_2["phi_local"]))
		phi_max = np.floor(phi_max * 100) / 100
		levels = np.linspace(0, phi_max, 101)

		# create contour plot of local flow coefficient
		contour_1 = axes[0].contourf(
			outlet_1["y"], outlet_1["z"], outlet_1["phi_local"], levels = levels, extend = "max"
		)
		contour_2 = axes[1].contourf(
			outlet_2["y"], outlet_2["z"], outlet_2["phi_local"], levels = levels, extend = "max"
		)

		# loop for each axes
		for ax in axes:

			# remove axes and set aspect ratio to equal
			ax.axis("off")
			ax.set_aspect("equal")

		# add colour bar
		fig.colorbar(contour_2, ax = axes, ticks = levels[::10], label = r"$\phi_\text{local}$")

		# set axis titles
		axes[0].set_title(
			rf"Re = {data_1['Re']:.0f}" + "\n"
			rf"$\eta_\text{{poly}}$ = {data_1['eta_poly']:.4g}"
		)
		axes[1].set_title(
			rf"Re = {data_2['Re']:.0f}" + "\n"
			rf"$\eta_\text{{poly}}$ = {data_2['eta_poly']:.4g}"
		)

		# set figure title
		fig.suptitle(
			rf"Rotor Exit Axial Section, Contours of $\phi_\text{{local}}$" + "\n"
			rf"$N$ = 1, $\phi$ = {phi:.4g}, $\psi$ = {psi:.4g}",
			y = 1.2, fontsize = utils.Defaults.titlesize
		)

		# save figure
		fig.savefig("exports/compare_reynolds_rotor.png", dpi = utils.Defaults.dpi, bbox_inches = "tight")

		# create plot for stator
		fig, axes = plt.subplots(1, 2, figsize = utils.Defaults.figsize, sharex = True, sharey = True)

		# get stator exit planes
		outlet_1 = data_1["blade_rows"][1]["outlet"]
		outlet_2 = data_2["blade_rows"][1]["outlet"]

		# get max. flow coefficient
		phi_max = np.nanmax(outlet_1["phi_local"])
		phi_max = max(phi_max, np.nanmax(outlet_2["phi_local"]))
		phi_max = np.floor(phi_max * 100) / 100
		levels = np.linspace(0, phi_max, 101)

		# create contour plot of local flow coefficient
		contour_1 = axes[0].contourf(
			outlet_1["y"], outlet_1["z"], outlet_1["phi_local"], levels = levels, extend = "max"
		)
		contour_2 = axes[1].contourf(
			outlet_2["y"], outlet_2["z"], outlet_2["phi_local"], levels = levels, extend = "max"
		)

		# loop for each axes
		for ax in axes:

			# remove axes and set aspect ratio to equal
			ax.axis("off")
			ax.set_aspect("equal")

		# add colour bar
		fig.colorbar(contour_2, ax = axes, ticks = levels[::10], label = r"$\phi_\text{local}$")

		# set axis titles
		axes[0].set_title(
			rf"Re = {data_1['Re']:.0f}" + "\n"
			rf"$\eta_\text{{poly}}$ = {data_1['eta_poly']:.4g}"
		)
		axes[1].set_title(
			rf"Re = {data_2['Re']:.0f}" + "\n"
			rf"$\eta_\text{{poly}}$ = {data_2['eta_poly']:.4g}"
		)

		# set figure title
		fig.suptitle(
			rf"Stator Exit Axial Section, Contours of $\phi_\text{{local}}$" + "\n"
			rf"$N$ = 1, $\phi$ = {phi:.4g}, $\psi$ = {psi:.4g}",
			y = 1.2, fontsize = utils.Defaults.titlesize
		)

		# save figure
		fig.savefig("exports/compare_reynolds_stator.png", dpi = utils.Defaults.dpi, bbox_inches = "tight")

		break

# upon script execution
if __name__ == "__main__":

	# run main and show all plots
	main()
	plt.show()
