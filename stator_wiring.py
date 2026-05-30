# stator_wiring.py
# 27 May 2026

# import modules
import numpy as np
import matplotlib.pyplot as plt
import copy

# import high speed solver
from engine import Engine
from flight_scenario import Flight_scenario
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

def draw_wire(ax, r, l, xx, yy, d = 0):
	"""Draws a hatched wire at a position along a given camber line on a given set of axes."""
	# calculate array of cumulative lengths along camber line
	ll = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(xx)**2 + np.diff(yy)**2))])
	ll = ll / ll[-1]

	# find coordinates of fraction x along camber line
	x = np.interp(l, ll, xx)
	y = np.interp(l, ll, yy)

	# calculate gradient at camber line at position of wire
	dx = np.interp(l, ll, np.gradient(xx, ll))
	dy = np.interp(l, ll, np.gradient(yy, ll))
	
	# add normal vector to wire centre position
	mag = np.hypot(dx, dy) + 1e-10
	x = x - d * (dy / mag)
	y = y + d * (dx / mag)

	# add patch
	wire = plt.Circle(
		(x, y), r,
		fill = False, hatch = "xxxxx"
	)
	ax.add_patch(wire)

	# return coordinates of centre of wire
	return x, y

def draw_dovetail(ax, xx, yy, xx_c, yy_c, offset = (0, 0), start = 0.3, width = 0.12, depth = 0.018):
	"""Draws a dovetail groove on the given axes for a given blade shape"""
	# define radii and undercut angle
	r1 = 0.005
	r2 = 0.01
	theta = np.pi / 9
	N = 100

	# Calculate array of cumulative lengths along camber line
	ll = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(xx_c)**2 + np.diff(yy_c)**2))])
	ll = ll / ll[-1]

	# find start and end points on camber line
	a1 = np.array([np.interp(start + width / 2, ll, xx_c), np.interp(start + width / 2, ll, yy_c)])
	a2 = np.array([np.interp(start - width / 2, ll, xx_c), np.interp(start - width / 2, ll, yy_c)])

	# calculate gradient at camber line at centre of dovetail groove
	dx = np.interp(start, ll, np.gradient(xx_c, ll))
	dy = np.interp(start, ll, np.gradient(yy_c, ll))
	
	# add normal vector to wire centre position
	mag = np.hypot(dx, dy) + 1e-10
	x = a1[0] - dy / mag
	y = a1[1] + dx / mag
	b1 = np.array([x, y])

	# find cross products
	v_line = b1 - a1
	v_curve = np.column_stack((xx, yy)) - a1
	cross_products = np.cross(v_line, v_curve)

	# find where sign flips
	idx = np.where(np.diff(np.sign(cross_products)) != 0)[0]

	# linear interpolation
	if len(idx) > 0:
		i = idx[-1]
		t = abs(cross_products[i]) / (abs(cross_products[i]) + abs(cross_products[i+1]))
		x_int = xx[i] + t * (xx[i+1] - xx[i])
		y_int = yy[i] + t * (yy[i+1] - yy[i])

	# get dovetail start point
	p1 = (x_int, y_int)

	# calculate new line for dovetail end point
	x = a2[0] - dy / mag
	y = a2[1] + dx / mag
	b2 = np.array([x, y])

	# find cross products
	v_line = b2 - a2
	v_curve = np.column_stack((xx, yy)) - a2
	cross_products = np.cross(v_line, v_curve)

	# find where sign flips
	idx = np.where(np.diff(np.sign(cross_products)) != 0)[-1]

	# linear interpolation
	if len(idx) > 0:
		i = idx[-1]
		t = abs(cross_products[i]) / (abs(cross_products[i]) + abs(cross_products[i+1]))
		x_int = xx[i] + t * (xx[i+1] - xx[i])
		y_int = yy[i] + t * (yy[i+1] - yy[i])

	# get dovetail end point 
	p2 = (x_int, y_int)

	# get angle between end points
	alpha = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

	# find external radii centres
	c1 = (p2[0] + r1 * np.sin(alpha) + offset[0], p2[1] - r1 * np.cos(alpha) + offset[1])
	c2 = (p1[0] + r1 * np.sin(alpha) + offset[0], p1[1] - r1 * np.cos(alpha) + offset[1])

	# draw external radius
	zz = np.linspace(-np.pi / 2 - alpha - theta, -alpha, N)
	xx = c1[0] + r1 * np.sin(zz)
	yy = c1[1] + r1 * np.cos(zz)
	ax.plot(xx, yy, color = "k")
	q1 = (xx[0], yy[0])

	# find start of internal radius
	q2 = (q1[0] + depth * np.sin(alpha + theta), q1[1] - depth * np.cos(alpha + theta))
	ax.plot([q1[0], q2[0]], [q1[1], q2[1]], color = "k")

	# draw internal radii
	c3 = (q2[0] - r2 * np.cos(alpha + theta), q2[1] - r2 * np.sin(alpha + theta))
	zz = np.linspace(np.pi / 2 - alpha - theta, np.pi - alpha, N)
	xx = c3[0] + r2 * np.sin(zz)
	yy = c3[1] + r2 * np.cos(zz)
	ax.plot(xx, yy, color = "k")

	# repeat for other external radius
	zz = np.linspace(-alpha, np.pi / 2 - alpha + theta, N)
	xx = c2[0] + r1 * np.sin(zz)
	yy = c2[1] + r1 * np.cos(zz)
	ax.plot(xx, yy, color = "k")
	q1 = (xx[-1], yy[-1])

	# find start of internal radius
	q2 = (q1[0] + depth * np.sin(alpha - theta), q1[1] - depth * np.cos(alpha - theta))
	ax.plot([q1[0], q2[0]], [q1[1], q2[1]], color = "k")

	# draw internal radii
	c4 = (q2[0] + r2 * np.cos(alpha - theta), q2[1] + r2 * np.sin(alpha - theta))
	zz = np.linspace(np.pi - alpha, 3 * np.pi / 2 - alpha + theta, N)
	xx = c4[0] + r2 * np.sin(zz)
	yy = c4[1] + r2 * np.cos(zz)
	ax.plot(xx, yy, color = "k")

	# connect halves
	ax.plot(
		[c3[0] + r2 * np.sin(alpha), c4[0] + r2 * np.sin(alpha)],
		[c3[1] - r2 * np.cos(alpha), c4[1] - r2 * np.cos(alpha)], color = "k"
	)

def draw_arrow(ax, x, y, theta, text, l1 = 0.2, l2 = 0.2):
	"""Draws an arrow with text annotating a given axes."""
	# draw first arrow
	ax.annotate(
		"",
		xy = (x, y),
		xytext = (x - l1 * np.sin(theta), y - l1 * np.cos(theta)),
		arrowprops = dict(arrowstyle = "-|>", color = "k", linewidth = 0.5, shrinkA = 0, shrinkB = 0)
	)

	# draw second arrow with no head
	ax.annotate(
		"",
		xy = (
			x - l1 * np.sin(theta) - l2 * np.sign(np.sin(theta)),
			y - l1 * np.cos(theta)
		),
		xytext = (x - l1 * np.sin(theta), y - l1 * np.cos(theta)),
		arrowprops = dict(arrowstyle = "-", color = "k", linewidth = 0.5, shrinkA = 0, shrinkB = 0)
	)

	# add invisible line to update axis limits
	ax.plot(
		[x - l1 * np.sin(theta) - l2 * np.sign(np.sin(theta)), x - l1 * np.sin(theta)],
		[y - l1 * np.cos(theta), y - l1 * np.cos(theta)],
		linestyle = ""
	)

	# define text offset
	delta = 0.01

	# define text alignment
	align = "left" if np.sin(theta) > 0 else "right"

	# display text
	ax.text(
		x - l1 * np.sin(theta) - l2 * np.sign(np.sin(theta)),
		y - l1 * np.cos(theta) + delta,
		text,
		ha = align
	)

def add_detail(ax, centre, radius, scale, location, label = "", theta = -np.pi / 4):

	# draw circles for section view
	circle = plt.Circle(centre, radius, fill = False, linestyle = "-.")
	ax.add_patch(circle)
	circle = plt.Circle(location, scale * radius, fill = False, linestyle = "-.")
	ax.add_patch(circle)

	# add detail view text
	draw_arrow(
		ax, centre[0] - radius * np.sin(theta), centre[1] - radius * np.cos(theta), theta, label
	)
	ax.text(location[0], location[1] - 1.5 * scale * radius, f"DETAIL {label}", ha = "center")

	# loop for each line added
	for line in ax.lines:

		# extract line x-y information
		xx = line.get_xdata()
		yy = line.get_ydata()

		# mask datapoints outside of detail view
		dx = xx - centre[0]
		dy = yy - centre[1]
		mask = (dx**2 + dy**2) <= radius**2

		# check if any part of line is to be plotted
		if np.any(mask):

			# pad mask and get indices of changes from False to True and vice versa
			mask_padded = np.concatenate(([False], mask, [False]))
			idx = np.where(np.diff(mask_padded))[0]
			
			# loop for each index
			for k in range(0, len(idx), 2):

				# get start and end of line segment
				start = idx[k]
				end = idx[k + 1]
				
				# get line segment, centred in detail view, scaled and re-positioned
				x = scale * (xx[start:end] - centre[0]) + location[0]
				y = scale * (yy[start:end] - centre[1]) + location[1]
				
				new_line = plt.Line2D(
					x, y, 
					color=line.get_color(), 
					linestyle=line.get_linestyle(),
					linewidth=line.get_linewidth()
				)
				ax.add_line(new_line)

	# loop for each patch added to the original axis
	for patch in ax.patches:
		
		# Circle instances only
		if isinstance(patch, plt.Circle):
			
			# get distances to detail view centre
			dx = patch.center[0] - centre[0]
			dy = patch.center[1] - centre[1]
			
			# circle is inside detail view
			if (dx**2 + dy**2) <= radius**2:
				
				# duplicate the patch to keep all original styling
				new_circle = copy.copy(patch)
				new_circle.axes = None
				
				# centre in detail view, scale and reposition
				new_cx = scale * (patch.center[0] - centre[0]) + location[0]
				new_cy = scale * (patch.center[1] - centre[1]) + location[1]
				new_circle.center = (new_cx, new_cy)
				new_circle.radius = patch.radius * scale
				
				# add new circle
				ax.add_patch(new_circle)

# main function
def main():

	# define default flight scenario
	scenario = Flight_scenario(
		"",
		utils.Defaults.altitude,
		utils.Defaults.flight_speed,
		utils.Defaults.diameter,
		utils.Defaults.hub_tip_ratio,
		utils.Defaults.thrust
	)

	# create engine with default values
	engine = Engine(scenario)
	engine.design()

	# specify geometry object and complete empirical design process
	engine.geometry = {
		"aspect_ratio": utils.Defaults.aspect_ratio,
		"diffusion_factor": utils.Defaults.diffusion_factor,
		"design_parameter": utils.Defaults.design_parameter
	}
	engine.empirical_design()

	# impose thickness distribution
	engine.geometry["thickness"] = {
		"max_thickness_mm": utils.Defaults.max_thickness_mm,
		"thickness_fraction": utils.Defaults.thickness_fraction
	}
	engine.calculate_thickness()

	# store stator object separately for convenience
	stator = engine.blade_rows[1]

	# store hub chord separately for convenience
	hub_chord = stator.exit.chord[0]

	# store dimensionless hub blade shape separately for convenience
	xx = stator.xx[0] / hub_chord
	yy = stator.yy[0] / hub_chord

	# store hub camber line separately for convenience (already unit length)
	xx_camber = stator.xx_camber[0]
	yy_camber = stator.yy_camber[0]

	# create plot
	fig, ax = plt.subplots(figsize = utils.Defaults.figsize)

	# define offset between two blades shown
	offset = [0.6, -0.2]

	# draw stator hub and tip sections non-dimensionalised by chord
	ax.plot(xx, yy, color = "k")
	ax.plot( xx + offset[0], yy + offset[-1], color = "k")

	# add wires to first configuration
	r_wire = 0.018
	draw_wire(ax, r_wire, 0.22, xx_camber, yy_camber)
	centre = draw_wire(ax, r_wire, 0.3, xx_camber, yy_camber)
	draw_wire(ax, r_wire, 0.38, xx_camber, yy_camber)

	# define detail view for configuration A
	radius = 0.12
	scale = 3
	location = (-0.5, 0.05)

	# detail view for configuration A
	add_detail(ax, centre, radius, scale, location, label = "A", theta = 5 * np.pi / 4)
	
	# annotated detail view
	d = 0.05
	draw_arrow(
		ax, location[0] + d, location[1] - d, -np.pi / 4, "Material thickness critical", l1 = 0.7, l2 = 1.3
	)

	# draw dovetail groove on second configuration
	draw_dovetail(ax, xx, yy, xx_camber, yy_camber, offset)

	# add wires to second configuration
	d = -0.01
	_ = draw_wire(ax, r_wire, 0.26, xx_camber + offset[0], yy_camber + offset[1], d)
	centre = draw_wire(ax, r_wire, 0.3, xx_camber + offset[0], yy_camber + offset[1], d)
	_ = draw_wire(ax, r_wire, 0.34, xx_camber + offset[0], yy_camber + offset[1], d)

	# define detail view for configuration B
	radius = 0.12
	scale = 3
	location = (2, 0.05)

	# detail view for configuration B
	add_detail(ax, centre, radius, scale, location, label = "B")
	
	# make annotations
	d = 0.06
	draw_arrow(
		ax, location[0] - d, location[1] + d, 3 * np.pi / 4, "Dovetail groove", l1 = 0.6, l2 = 0.9
	)
	d = 0.07
	draw_arrow(
		ax, location[0], location[1] - d, np.pi / 4, "Apply speed tape", l1 = 1, l2 = 0.85
	)

	# configure plot
	ax.axis("off")
	ax.set_aspect("equal", adjustable = "box")

	# save figure
	fig.savefig("exports/stator_wiring.png", dpi = utils.Defaults.dpi, bbox_inches = "tight")

# design_wiring function
def design_wiring():

	# define default flight scenario
	scenario = Flight_scenario(
		"",
		utils.Defaults.altitude,
		utils.Defaults.flight_speed,
		utils.Defaults.diameter,
		utils.Defaults.hub_tip_ratio,
		utils.Defaults.thrust
	)

	# create engine with values selected from best design
	engine = Engine(
		scenario, 2, 0.7, 0.25, utils.Defaults.vortex_exponent, utils.Defaults.Y_p,
		utils.Defaults.area_ratio, 0.25
	)
	engine.design()

	# specify geometry object and complete empirical design process
	engine.geometry = {
		"aspect_ratio": utils.Defaults.aspect_ratio,
		"diffusion_factor": utils.Defaults.diffusion_factor,
		"design_parameter": utils.Defaults.design_parameter
	}
	engine.empirical_design()

	# impose thickness distribution
	engine.geometry["thickness"] = {
		"max_thickness_mm": 1,
		"thickness_fraction": 0.4
	}
	engine.calculate_thickness()

	# determine engine motor requirements
	engine.geometry["motor"] = {
		"motor_power": engine.blade_rows[0].motor_power,
		"motor_rpm": engine.blade_rows[0].motor_rpm,
		"motor_diameter": scenario.diameter * engine.hub_tip_ratio,
		"cable_diameter": 0.6,
		"cables_per_phase": 44
	}
	engine.select_motor()

	# determine min. electric current density
	J_min = np.min([motor["J"] for motor in engine.motors])

	# store stator object separately for convenience
	stator = engine.blade_rows[1]

	# store hub chord separately for convenience
	hub_chord = stator.exit.chord[0]

	# store dimensionless hub blade shape separately for convenience
	xx = stator.xx[0] / hub_chord
	yy = stator.yy[0] / hub_chord

	# store hub camber line separately for convenience (already unit length)
	xx_camber = stator.xx_camber[0]
	yy_camber = stator.yy_camber[0]

	# create plot
	fig, ax = plt.subplots(figsize = utils.Defaults.figsize)

	# draw stator hub and tip sections non-dimensionalised by chord
	ax.plot(xx, yy, color = "k")

	# get wire radius
	r_wire_mm = 0.3
	r_wire = r_wire_mm / (hub_chord * scenario.radius * 1e3)

	# get list of wire positions
	m = 6
	n = 2
	zz = np.linspace(0.25, 0.45, n)

	# offset from camber line
	d = -0.005

	# offset between wires
	D = 0.025
	wire_offsets = D * (np.arange(m) - (m - 1) / 2)

	# loop for each bundle of wires wire
	for z in zz:

		# loop for each wire in bundle
		for o in wire_offsets:

			# add wires to configuration
			centre = draw_wire(ax, r_wire, z + o, xx_camber, yy_camber, d)

		# draw dovetail groove
		draw_dovetail(
			ax, xx, yy, xx_camber, yy_camber, offset = (0, 0), start = z, width = m * D, depth = 0.008
		)

	# annotate first configuration
	draw_arrow(
		ax, *centre, 5 * np.pi / 4,
		rf"{m} $\times$ {n} cables, d = {2 * r_wire_mm} mm" + "\n"
		rf"J = {J_min:.4g} $\text{{A mm}}^{{-2}}$",
		l1 = 0.25, l2 = 0.55
	)
	draw_arrow(
		ax, *centre, 3 * np.pi / 4,
		"Design thickness\n" + r"$t_\text{max}$ = 1 mm",
		l1 = 0.2, l2 = 0.4
	)

	# impose alternative, slightly thicker, thickness distribution
	engine.geometry["thickness"] = {
		"max_thickness_mm": 2,
		"thickness_fraction": 0.3
	}
	engine.calculate_thickness()

	# determine engine motor requirements
	engine.geometry["motor"] = {
		"motor_power": engine.blade_rows[0].motor_power,
		"motor_rpm": engine.blade_rows[0].motor_rpm,
		"motor_diameter": scenario.diameter * engine.hub_tip_ratio,
		"cable_diameter": 1.2,
		"cables_per_phase": 22
	}
	engine.select_motor()

	# determine min. electric current density
	J_min = np.min([motor["J"] for motor in engine.motors])

	# store stator object separately for convenience
	stator = engine.blade_rows[1]

	# store hub chord separately for convenience
	hub_chord = stator.exit.chord[0]

	# store dimensionless hub blade shape separately for convenience
	xx = stator.xx[0] / hub_chord
	yy = stator.yy[0] / hub_chord

	# store hub camber line separately for convenience (already unit length)
	xx_camber = stator.xx_camber[0]
	yy_camber = stator.yy_camber[0]

	# define offset between two blades shown
	offset = [0.6, -0.2]
	ax.plot(xx + offset[0], yy + offset[-1], color = "k")

	# get wire radius
	r_wire_mm = 0.6
	r_wire = r_wire_mm / (hub_chord * scenario.radius * 1e3)

	# get list of wire positions
	m = 6
	n = 1
	zz = np.linspace(0.35, 0.35, n)

	# offset from camber line
	d = -0.012

	# offset between wires
	D = 0.05
	wire_offsets = D * (np.arange(m) - (m - 1) / 2)

	# loop for each bundle of wires wire
	for z in zz:

		# loop for each wire in bundle
		for o in wire_offsets:

			# add wires to configuration
			centre = draw_wire(ax, r_wire, z + o, xx_camber + offset[0], yy_camber + offset[1], d)

		# draw dovetail groove
		draw_dovetail(
			ax, xx, yy, xx_camber, yy_camber, offset = offset, start = z, width = m * D, depth = 0.03
		)

	# annotate second configuration
	draw_arrow(
		ax, *centre, 5 * np.pi / 4,
		rf"{m} $\times$ {n} cables, d = {2 * r_wire_mm} mm" + "\n"
		rf"J = {J_min:.4g} $\text{{A mm}}^{{-2}}$",
		l1 = 0.2, l2 = 0.55
	)
	draw_arrow(
		ax, *centre, -np.pi / 4, r"$t_\text{max}$ = 2 mm",
		l1 = 0.2, l2 = 0.4
	)

	# configure plot
	ax.axis("off")
	ax.set_aspect("equal", adjustable = "box")

	# save figure
	fig.savefig("exports/design_wiring.png", dpi = utils.Defaults.dpi, bbox_inches = "tight")


# upon script execution
if __name__ == "__main__":

	# run main functions and show all plots
	main()
	design_wiring()
	plt.show()
