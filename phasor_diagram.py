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
	E = 80			# back-emf magnitude (V)
	I = 30			# current magnitude (A)
	R = 0.5			# stator resistance (Ohm)
	X_s = 1.8		# synchronous reactance (Ohm)
	beta = 60		# torque angle (deg)
	k = 1			# emf constant (Vs/rad)

	# plotting parameters
	margin = 0.2
	N = 100

# draw_phasor function
def draw_phasor(ax, start, vec, label, colour = "C0", text_offset = (5, 5)):
	"""Helper function to annotate phasors as arrows on Im-Re axes."""
	# find endpoint
	end = (start[0] + vec[0], start[1] + vec[1])

	# annotate arrow
	ax.annotate(
		"", 
		xy=end, 
		xytext=start,
		arrowprops=dict(arrowstyle="->", lw = 1, color=colour, shrinkA = 0, shrinkB = 0)
	)

	# also plot invisible datatpoints to ensure axis limits are adjusted
	ax.plot([start[0], end[0]], [start[1], end[1]], linestyle = "")

	# place text, slightly offset from the tip of the arrow
	ax.annotate(
		label, 
		xy=end, 
		xytext=text_offset, 
		textcoords="offset points", 
		color=colour
	)

def draw_angle(ax, start, end, radius, label, colour = "C0", text_offset = (10, 0)):
	"""Helper function to annotate angles on phasor diagram."""
	# plot coordinates of curve
	theta = np.linspace(start, end, Inputs.N)
	xx = radius * np.cos(theta)
	yy = radius * np.sin(theta)
	ax.plot(xx, yy, color = "C0")

	# get midpoint radius
	xy = (xx[int(len(xx) / 2)], yy[int(len(yy) / 2)])

	# annotate text
	ax.annotate(
		label,
		xy = xy,
		xytext = text_offset,
		textcoords = "offset points",
		color = colour
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
	I = I_ph * np.exp(1j * (beta - np.pi / 2))
	
	# calculate voltage drops
	IR = I * R
	IX_s = I * (1j * X_s)
	
	# calculate terminal voltage
	V = E + IR + IX_s
	
	# plotting coordinates
	origin = (0, 0)
	p_E = (E.real, E.imag)
	p_V = (V.real, V.imag)
	p_I = (I.real, I.imag)
	p_IR = (IR.real, IR.imag)
	p_IX_s = (IX_s.real, IX_s.imag)
	p_E_IR = (p_E[0] + IR.real, p_E[1] + IR.imag)
	
	# draw phasors
	draw_phasor(ax, origin, p_E, r"$\vec{E}$", "k", (-20, 5))
	draw_phasor(ax, origin, p_V, r"$\vec{V}$", "k", (-30, 0))
	draw_phasor(ax, origin, p_I, r"$\vec{I}$", "k", (5, -10))
	draw_phasor(ax, p_E, p_IR, r"$\vec{I}R$", "k", (-25, -10))
	draw_phasor(ax, origin, p_IX_s, r"$j \omega L_s \vec{I}$", "k", (5, -20))
	draw_phasor(ax, p_E_IR, p_IX_s, r"$j \omega L_s \vec{I}$", "k", (5, -20))

	# draw torque and power factor angles
	draw_angle(ax, 0, beta, 10, r"$\beta$", "C0", (5, 5))
	draw_angle(ax, np.angle(I), np.angle(V), 20, r"$\varphi$", "C0", (5, -10))
	
	# configure plot
	#ax.set_title("Sinusoidal BLDC Motor Phasor Diagram")
	ax.set_xlabel("Real Axis")
	ax.set_ylabel("Imaginary Axis")
	
	# equal aspect ratio
	ax.set_aspect('equal', adjustable='box')
	
	# switch off axes and grid
	ax.axis("off")
	
	# save figure
	fig.savefig("exports/phasor_diagram.png", dpi = utils.Defaults.dpi, bbox_inches = "tight")

	# create another figure
	fig, ax = plt.subplots(figsize = utils.Defaults.figsize)

	# calculate rated torque
	k = Inputs.k
	tau = 3 * k * I_ph

	# calculate rated speed
	omega = E_ph / k
	ax.plot([0, omega], [tau, tau], color = "k")

	# calculate torque-speed characteristic in field-weakening regime
	xx = np.linspace(omega, 2 * omega, Inputs.N)
	yy = tau * omega / xx
	ax.plot(xx, yy, color = "k")

	# annotated field-weakening regime
	ax.plot(omega, tau, marker = ".", markersize = 12, color = "k")
	ax.annotate(
		"(Rated speed, rated torque)",
		xy = (omega, tau),
		xytext = (5, 5),
		textcoords = "offset points",
		color = "C0"
	)
	ax.axvline(omega, linestyle = "--", linewidth = 1, color = "k")
	ax.annotate(
		"Constant torque",
		xy = (omega / 2, tau),
		xytext = (-40, 5),
		textcoords = "offset points",
		color = "C0"
	)
	ax.annotate(
		"Constant power",
		xy = (xx[int(len(xx) / 2)], yy[int(len(yy) / 2)]),
		xytext = (5, 5),
		textcoords = "offset points",
		color = "C0"
	)

	# add arrow and label for field-weakening regime
	ax.annotate(
		"",
		xy = (1.3 * omega, tau / 5),
		xytext = (1.05 * omega, tau / 5),
		arrowprops=dict(arrowstyle="-|>", color="C0", lw=1.5, shrinkA=0, shrinkB=0)
	)
	ax.annotate(
		"Field-weakening",
		xy = (1.1 * omega, tau / 5),
		xytext = (5, 10),
		textcoords = "offset points",
		color = "C0"
	)

	# configure plot
	ax.set_xlabel(r"Rotational Speed, $\Omega$")
	ax.set_ylabel(r"Torque, $\tau$")
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)

	# 2. Move the remaining left and bottom spines to the origin (0,0)
	ax.spines['left'].set_position('zero')
	ax.spines['bottom'].set_position('zero')

	# 3. Remove all ticks and tick labels
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_ylim(0, 1.2 * tau)
		
	# save figure
	fig.savefig("exports/torque_speed_characteristic.png", dpi = utils.Defaults.dpi, bbox_inches = "tight")

	
# upon script execution
if __name__ == "__main__":

	# run main
	main()
	plt.show()
