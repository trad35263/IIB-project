# CFD_validation2.py
# 31 May 2026

# import modules
import numpy as np
import matplotlib.pyplot as plt
import os
import utils
from smith_post_process import Post
from smith_post_process import Inputs
from matplotlib.lines import Line2D

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

def span(rr):

	return (rr - rr[0]) / (rr[-1] - rr[0])

# main function
def main():
	
	# set inputs folder to M = 0.1 cases
	Inputs.folder = os.path.join(Inputs.folder_path, "N_2_M_0.25")

	# create post-processing object
	post = Post()
	post.calc_secondary()
	post.analyse()
	post.calculate_Re()
	post.calculate_thrust()

	# get index of desired test case
	phi, psi = 0.7, 0.25
	index = next(
		i for i, d in enumerate(post.data)
		if (np.abs(d["phi"] - phi) < 1e-6) and (np.abs(d["psi"] - psi) < 1e-6)
	)

	# get relevant data object
	data = post.data[index]

	# create plot
	fig, axes = plt.subplots(1, 4, figsize = (9, 3.2))

	# plot CFD absolute flow angles
	axes[0].plot(
		data["blade_rows"][0]["outlet"]["alpha_avg"], span(data["blade_rows"][0]["outlet"]["r"]), color = "C0"
	)
	axes[1].plot(
		data["blade_rows"][1]["outlet"]["alpha_avg"], span(data["blade_rows"][1]["outlet"]["r"]), color = "C0"
	)
	axes[2].plot(
		data["blade_rows"][2]["outlet"]["alpha_avg"], span(data["blade_rows"][2]["outlet"]["r"]), color = "C0"
	)
	axes[3].plot(
		data["blade_rows"][3]["outlet"]["alpha_avg"], span(data["blade_rows"][3]["outlet"]["r"]), color = "C0"
	)

	# plot CFD relative flow angles
	axes[0].plot(
		data["blade_rows"][0]["outlet"]["beta_avg"], span(data["blade_rows"][0]["outlet"]["r"]), color = "C1"
	)
	axes[2].plot(
		data["blade_rows"][2]["outlet"]["beta_avg"], span(data["blade_rows"][2]["outlet"]["r"]), color = "C1"
	)

	# get engine
	engine = data["engine_design"]

	# plot design tool absolute flow angles
	axes[0].plot(
		utils.rad_to_deg(engine.blade_rows[0].exit.alpha), span(engine.blade_rows[0].exit.rr),
		color = "C0", linestyle = "--"
	)
	axes[1].plot(
		utils.rad_to_deg(engine.blade_rows[1].exit.alpha), span(engine.blade_rows[1].exit.rr),
		color = "C0", linestyle = "--"
	)
	axes[2].plot(
		utils.rad_to_deg(engine.blade_rows[2].exit.alpha), span(engine.blade_rows[2].exit.rr),
		color = "C0", linestyle = "--"
	)
	axes[3].plot(
		utils.rad_to_deg(engine.blade_rows[3].exit.alpha), span(engine.blade_rows[3].exit.rr),
		color = "C0", linestyle = "--"
	)

	# plot design tool relative flow angles
	axes[0].plot(
		utils.rad_to_deg(engine.blade_rows[0].exit.beta), span(engine.blade_rows[0].exit.rr),
		color = "C1", linestyle = "--"
	)
	axes[2].plot(
		utils.rad_to_deg(engine.blade_rows[2].exit.beta), span(engine.blade_rows[2].exit.rr),
		color = "C1", linestyle = "--"
	)

	# plot design tool metal angles
	axes[0].plot(
		utils.rad_to_deg(engine.blade_rows[0].exit.metal_angle), span(engine.blade_rows[0].exit.rr),
		color = "C2", linestyle = "--"
	)
	axes[1].plot(
		utils.rad_to_deg(engine.blade_rows[1].exit.metal_angle), span(engine.blade_rows[1].exit.rr),
		color = "C2", linestyle = "--"
	)
	axes[2].plot(
		utils.rad_to_deg(engine.blade_rows[2].exit.metal_angle), span(engine.blade_rows[2].exit.rr),
		color = "C2", linestyle = "--"
	)
	axes[3].plot(
		utils.rad_to_deg(engine.blade_rows[3].exit.metal_angle), span(engine.blade_rows[3].exit.rr),
		color = "C2", linestyle = "--"
	)

	axes[0].text(
		0.45, -0.03,
		"Angle (°)",
		transform=fig.transFigure
	)

	# configure plot
	for ax in axes:

		ax.grid()
		ax.set_ylim(0, 1)

	axes[0].set_ylabel("Dimensionless Span")
	#axes[2].set_xlabel("Angle (°)")


	# set titles
	axes[0].set_title("Rotor 1")
	axes[1].set_title("Stator 1")
	axes[2].set_title("Rotor 2")
	axes[3].set_title("Stator 2")
	
	# switch off y-axis labels for all but first axis
	axes[1].tick_params(axis = "y", labelleft = False)
	axes[2].tick_params(axis = "y", labelleft = False)
	axes[3].tick_params(axis = "y", labelleft = False)

	# figure title
	fig.suptitle(
		r"Design Tool vs. CFD Data for $N$ = 2, $M_1$ = 0.25, $\phi$ = 0.7, $\psi$ = 0.25",
		y = 1.05
	)

	# save figure
	fig.savefig("exports/CFD_validation1.png", dpi = utils.Defaults.dpi, bbox_inches = "tight")

	# custom legend handles
	custom_handles = [
		Line2D([0], [0], color = "C0", linestyle = "--", label = "Flow Angle (Design)"),
		Line2D([0], [0], color = "C1", linestyle = "--", label = "Relative Flow Angle (Design)"),
		Line2D([0], [0], color = "C2", linestyle = "--", label = "Metal Angle (Design)"),
		Line2D([0], [0], color = "C0", label = "Flow Angle (CFD)"),
		Line2D([0], [0], color = "C1", label = "Relative Flow Angle (CFD)")
	]

	# custom legend
	axes[0].legend(
		handles = custom_handles, 
		ncol = 2,
		loc = "upper center",
		bbox_to_anchor = (0.5, -0.05),
		bbox_transform = fig.transFigure,
		frameon = False
	)

	# save figure
	fig.savefig("exports/CFD_validation2.png", dpi = utils.Defaults.dpi, bbox_inches = "tight")

# upon script execution
if __name__ == "__main__":

	# run main and show all plots
	main()
	plt.show()
