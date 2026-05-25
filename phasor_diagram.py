# phasor_diagram.py
# 24 May 2026

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

	# default phasor parameters
	E = 80.0		# back-emf magnitude (V)
	I = 15.0		# current magnitude (A)
	R = 0.5			# stator resistance (Ohm)
	X_s = 1.8		# synchronous reactance (Ohm)
	beta = 20		# torque angle (deg)

# draw_phasor function
def draw_phasor(ax, start, end, label, color, text_offset = (5, 5)):
	"""Helper function to annotate phasors as arrows on Im-Re axes."""
	# annotate arrow
	ax.annotate(
		"", 
		xy=end, 
		xytext=start,
		arrowprops=dict(arrowstyle="->", lw=2, color=color, shrinkA=0, shrinkB=0)
	)

	# also plot invisible datatpoints to ensure axis limits are adjusted
	ax.plot([start[0], end[0]], [start[1], end[1]], linestyle = "")

	# place text, slightly offset from the tip of the arrow
	ax.annotate(
		label, 
		xy=end, 
		xytext=text_offset, 
		textcoords="offset points", 
		color=color
	)

# main function
def main():

	# create plot
	fig, ax = plt.subplots(figsize = utils.Defaults.figsize)
	
	# Load inputs
	E_ph = Inputs.E
	I_ph = Inputs.I
	R = Inputs.R
	X_s = Inputs.X_s
	beta = np.radians(Inputs.beta)
	
	# convert emf magnitude to a phasor with no imaginary part
	E = complex(E_ph, 0)
	
	# calculate current phasor with phase advance beta
	I = I_ph * np.exp(1j * beta)
	
	# calculate voltage drops
	IR = I * R
	IX_s = I * (1j * X_s)
	
	# calculate terminal voltage
	V = E + IR + IX_s
	
	# plotting coordinates
	origin = (0, 0)
	p_E = (E.real, E.imag)
	p_IR = (p_E[0] + IR.real, p_E[1] + IR.imag)
	p_V = (V.real, V.imag)
	p_I = (I.real, I.imag)
	
	# draw phasors
	draw_phasor(ax, origin, p_E, r"$\vec{E}$", "blue", text_offset = (5, -15))
	draw_phasor(ax, origin, p_V, r"$\vec{V}$", "red", text_offset = (5, 5))
	draw_phasor(ax, origin, p_I, r"$\vec{I}$", "darkgreen", text_offset = (5, 5))
	draw_phasor(ax, p_E, p_IR, r"$\vec{I}R$", "purple", text_offset = (5, -5))
	draw_phasor(ax, p_IR, p_V, r"$j\vec{I}X_s$", "orange", text_offset = (-25, 10))
	
	# configure plot
	#ax.set_title("Sinusoidal BLDC Motor Phasor Diagram")
	ax.set_xlabel("Real Axis")
	ax.set_ylabel("Imaginary Axis")
	
	# equal aspect ratio
	ax.set_aspect('equal', adjustable='box')
	
	# Dynamically set limits with padding
	"""all_x = [0, p_E[0], p_V[0], p_I_scaled[0]]
	all_y = [0, p_E[1], p_V[1], p_I_scaled[1]]
	ax.set_xlim(min(all_x) - 10, max(all_x) + 20)
	ax.set_ylim(min(all_y) - 10, max(all_y) + 20)"""
	
	ax.grid(True)
	
	# save figure
	fig.savefig("exports/phasor_diagram.png", dpi = utils.Defaults.dpi, bbox_inches = "tight")
	
# upon script execution
if __name__ == "__main__":

	# run main
	main()
	plt.show()
