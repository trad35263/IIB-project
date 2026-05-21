# dimensional_analysis.py
# 20 May 2026

# import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# import high speed solver
from flight_scenario import Flight_scenario
from engine import Engine
import utils

def project(xx, yy, zz, theta = np.pi / 4, phi = np.pi / 4):
    """Projects an array of 3D coordinates onto a 2D plane."""
    # unit projection vector
    n = np.array([np.cos(theta), np.sin(theta), np.sin(phi)])
    n = n / np.linalg.norm(n)

    # world vertical
    k = np.array([0.0, 0.0, 1.0])

    # in-plane "vertical direction"
    e2 = k - np.dot(k, n) * n
    e2 = e2 / np.linalg.norm(e2)
    
    # second in-plane axis
    e1 = np.cross(n, e2)

    # stack input vectors and reshape
    vv = np.stack([xx, yy, zz], axis=-1)
    shape = vv.shape[:-1]
    vv = vv.reshape(-1, 3)
    
    # 2D coordinates in plane
    x2d = np.dot(vv, e1)
    y2d = np.dot(vv, e2)

    # return to original shape and return
    x2d = x2d.reshape(shape)
    y2d = y2d.reshape(shape)
    return x2d, y2d

def offset(x, y, scale_factor):
    """Offsets a closed 2D loop by a scale factor along its unit normals.

    Parameters:
    -----------
    x, y : array-like
        The x and y coordinates of the closed loop.
    scale_factor : float
        The distance to offset the loop.

    Returns:
    --------
    x_offset, y_offset : ndarrays
        The coordinates of the offset closed loop.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # 1. Handle explicit closure (if the last point equals the first point)
    is_explicitly_closed = (
        np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1]) and len(x) > 1
    )

    if is_explicitly_closed:
        # Strip the last point so periodic gradients wrap perfectly
        x_calc = x[:-1]
        y_calc = y[:-1]
    else:
        x_calc = x
        y_calc = y

    # 2. Compute tangents using periodic central differences
    # np.roll(..., -1) is the next point, np.roll(..., 1) is the previous point
    dx = (np.roll(x_calc, -1) - np.roll(x_calc, 1)) / 2.0
    dy = (np.roll(y_calc, -1) - np.roll(y_calc, 1)) / 2.0

    # 3. Compute the normal vector (-dy, dx)
    # This choice points to the LEFT of the direction of travel.
    # For a Counter-Clockwise (CCW) loop, this points INWARD.
    nx = -dy
    ny = dx

    # 4. Normalize to get UNIT normals
    magnitude = np.sqrt(nx**2 + ny**2)
    # Prevent division by zero if two consecutive points are identical
    magnitude[magnitude == 0] = 1.0

    nx_unit = nx / magnitude
    ny_unit = ny / magnitude

    # 5. Apply the scaled offset
    x_offset = x_calc + scale_factor * nx_unit
    y_offset = y_calc + scale_factor * ny_unit

    # 6. Re-close the loop if it was originally explicitly closed
    if is_explicitly_closed:
        x_offset = np.append(x_offset, x_offset[0])
        y_offset = np.append(y_offset, y_offset[0])

    return x_offset, y_offset

def vertical_end_spline(
    p0,
    p1,
    N=200
):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    x0, y0, z0 = p0
    x1, y1, z1 = p1

    s = np.linspace(0.0, 1.0, N)

    dx = x1 - x0

    x = (
        x0
        + 2.0 * dx * s
        - dx * s**2
    )

    dy = y1 - y0

    y = (
        y0
        + 2.0 * dy * s
        - dy * s**2
    )

    z = z0 + (z1 - z0) * s

    return x, y, z

def quadratic(p0, p1, p2, num_samples=100):
    """Fits a single parametric quadratic curve through 3 points in 3D space.

    Parameters:
    -----------
    p0, p1, p2 : array-like of length 3
        The 3D coordinates of the start, middle, and end points.
    num_samples : int
        Number of points to sample along the calculated curve.
    """
    p0 = np.asarray(p0)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    # Calculate the vector coefficients
    A = 2 * p0 - 4 * p1 + 2 * p2
    B = -3 * p0 + 4 * p1 - p2
    # C is simply p0

    # Generate parameter t from 0 to 1
    t = np.linspace(0, 1, num_samples)

    # Evaluate the quadratic equation for all t values
    # np.outer handles multiplying the t-vectors by the coefficient vectors
    curve = (
        np.outer(t**2, A) + np.outer(t, B) + p0
    )  # p0 acts as C added to every row

    return np.transpose(curve)  # Returns an array of shape (num_samples, 3)^T

def plot_blade(blade_row, fig, ax, theta = np.pi / 4, phi = -np.pi / 4, overlap = 0.4):
    """Plots a 2D projection of a 3D blade shape onto a given axis."""
    # extract chord distribution
    chord = blade_row.exit.chord

    # get index of midspan
    index = int(np.round(np.interp(
        0.5 * (blade_row.exit.rr[0] + blade_row.exit.rr[-1]), blade_row.exit.rr,
        np.arange(utils.Defaults.solver_grid)
    )))

    # extract blade_row hub blade geometry
    xx_hub = blade_row.xx[0] - overlap * chord[0] * np.cos(blade_row.inlet.metal_angle[0])
    yy_hub = blade_row.yy[0] - overlap * chord[0] * np.sin(blade_row.inlet.metal_angle[0])
    zz_hub = np.zeros_like(xx_hub)

    # extract blade_row mid-span blade geometry
    xx_mid = blade_row.xx[1] - overlap * chord[index] * np.cos(blade_row.inlet.metal_angle[index])
    yy_mid = blade_row.yy[1] - overlap * chord[index] * np.sin(blade_row.inlet.metal_angle[index])
    zz_mid = np.ones_like(xx_mid) * (blade_row.exit.rr[index] - blade_row.exit.rr[0])

    # extract blade_row tip blade geometry
    xx_tip = blade_row.xx[-1] - overlap * chord[-1] * np.cos(blade_row.inlet.metal_angle[-1])
    yy_tip = blade_row.yy[-1] - overlap * chord[-1] * np.sin(blade_row.inlet.metal_angle[-1])
    zz_tip = np.ones_like(xx_tip) * (blade_row.exit.rr[-1] - blade_row.exit.rr[0])

    # find trailing edge coordinates
    te_hub = (xx_hub[0], yy_hub[0])
    te_mid = (xx_mid[0], yy_mid[0])
    te_tip = (xx_tip[0], yy_tip[0])

    # find leading edge coordinates
    rr_hub = np.sqrt((xx_hub - xx_hub[0])**2 + (yy_hub - yy_hub[0])**2)
    i = np.argmax(rr_hub)
    le_hub = (xx_hub[i], yy_hub[i])
    rr_mid = np.sqrt((xx_mid - xx_mid[0])**2 + (yy_mid - yy_mid[0])**2)
    i = np.argmax(rr_mid)
    le_mid = (xx_mid[i], yy_mid[i])
    rr_tip = np.sqrt((xx_tip - xx_tip[0])**2 + (yy_tip - yy_tip[0])**2)
    i = np.argmax(rr_tip)
    le_tip = (xx_tip[i], yy_tip[i])

    # construct spline between leading edges
    """xx_le, yy_le, zz_le = vertical_end_spline(
        [le_hub[0], le_hub[1], zz_hub[0]],
        [le_tip[0], le_tip[1], zz_tip[0]]
    )

    # construct spline between trailing edges
    xx_te, yy_te, zz_te = vertical_end_spline(
        [te_hub[0], te_hub[1], zz_hub[0]],
        [te_tip[0], te_tip[1], zz_tip[0]]
    )"""

    # construct spline between leading edges
    xx_le, yy_le, zz_le = quadratic(
        [le_hub[0], le_hub[1], zz_hub[0]],
        [le_mid[0], le_mid[1], zz_mid[0]],
        [le_tip[0], le_tip[1], zz_tip[0]]
    )

    # construct spline between trailing edges
    xx_te, yy_te, zz_te = quadratic(
        [te_hub[0], te_hub[1], zz_hub[0]],
        [te_mid[0], te_mid[1], zz_mid[0]],
        [te_tip[0], te_tip[1], zz_tip[0]]
    )

    # clip hub section by intersects with chord
    j = next(
        i for i, x in enumerate(xx_hub)
        if (np.abs(xx_hub[i] - le_hub[0]) < 1e-6) and (np.abs(yy_hub[i] - le_hub[1]) < 1e-6)
    )
    #xx_hub = xx_hub[j:]
    #yy_hub = yy_hub[j:]
    #zz_hub = zz_hub[j:]

    # determine projection of datapoints on an angled plane
    xx_proj_le, yy_proj_le = project(xx_le, yy_le, zz_le, theta, phi)
    xx_proj_te, yy_proj_te = project(xx_te, yy_te, zz_te, theta, phi)
    xx_proj_hub, yy_proj_hub = project(xx_hub, yy_hub, zz_hub, theta, phi)
    xx_proj_tip, yy_proj_tip = project(xx_tip, yy_tip, zz_tip, theta, phi)

    # plot 3D data
    ax.plot(*project(xx_le, yy_le, zz_le, theta, phi), color = "k")
    ax.plot(*project(xx_te, yy_te, zz_te, theta, phi), color = "k")
    ax.plot(*project(xx_hub[j:], yy_hub[j:], zz_hub[j:], theta, phi), color = "k", linestyle = "--")
    ax.plot(*project(xx_hub[:j], yy_hub[:j], zz_hub[:j], theta, phi), color = "k")
    ax.plot(*project(xx_tip, yy_tip, zz_tip, theta, phi), color = "k")

    # define origin and unit vectors
    origin = np.array([1.5 * chord[0], 0, 0.5 * chord[0]])

    # define array of unit vectors
    axes = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    labels = ["x", r"r$\theta$", "r"]

    # loop for each unit vector
    for axis, label in zip(axes, labels):

        # calculate start and end positions and plot projected points
        vector = np.transpose([origin, origin + 0.1 * axis])
        xx, yy = project(*vector, theta, phi)

        # plot coordinates spanned by arrow to force bounding box to encompass arrows
        ax.plot(xx, yy, linestyle = "")

        # annotate arrow corresponding to unit vector
        ax.annotate(
            "",
            xy=(xx[0], yy[0]),
            xytext=(xx[1], yy[1]),
            arrowprops=dict(arrowstyle="<|-", color="C0", lw=1.5, shrinkA=0, shrinkB=0)
        )

        # add arrow text
        ax.text(xx[1], yy[1], f" {label}", va='center', ha='left')

    # plot mid-span section as hidden lines
    ax.plot(*project(xx_mid, yy_mid, zz_mid, theta, phi), color = "k", linestyle = "--")

    # find offset of midspan
    xx_offset, yy_offset = offset(xx_mid, yy_mid, -0.02)

    # plot flow over mid-span as offset of cross-section
    i = np.argmax(rr_mid)
    xx_proj, yy_proj = project(xx_offset[:i], yy_offset[:i], zz_mid[:i], theta, phi)
    ax.plot(xx_proj, yy_proj, color = "C0")
    ax.plot(
        *project(
            xx_offset[0] + chord[0] * np.array([0, np.cos(blade_row.exit.metal_angle[index])]),
            yy_offset[0] + chord[0] * np.array([0, np.sin(blade_row.exit.metal_angle[index])]),
            zz_mid[0] + np.array([0, 0]),
            theta, phi
        ),
        color = "C0"
    )
    ax.plot(
        *project(
            xx_offset[i] - chord[0] * np.array([0, np.cos(blade_row.inlet.metal_angle[index])]),
            yy_offset[i] - chord[0] * np.array([0, np.sin(blade_row.inlet.metal_angle[index])]),
            zz_mid[0] + np.array([0, 0]),
            theta, phi
        ),
        color = "C0"
    )

    """ax.plot(*project(*te_hub, zz_hub[0], theta, phi), marker = ".", markersize = 12)
    ax.plot(*project(*te_mid, zz_mid[0], theta, phi), marker = ".", markersize = 12)
    ax.plot(*project(*te_tip, zz_tip[0], theta, phi), marker = ".", markersize = 12)
    ax.plot(*project(*le_hub, zz_hub[0], theta, phi), marker = ".", markersize = 12)
    ax.plot(*project(*le_mid, zz_mid[0], theta, phi), marker = ".", markersize = 12)
    ax.plot(*project(*le_tip, zz_tip[0], theta, phi), marker = ".", markersize = 12)"""

# main function
def main():

    # create Flight_scenario
    flight_scenario = Flight_scenario(
        "",
        utils.Defaults.altitude,
        utils.Defaults.flight_speed,
        utils.Defaults.diameter,
        utils.Defaults.hub_tip_ratio,
        utils.Defaults.thrust
    )

    # create and design engine
    engine = Engine(flight_scenario)
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

    # store rotor and stator separately for convenience
    rotor = engine.blade_rows[0]
    stator = engine.blade_rows[1]

    # create plot of rotor
    fig, ax = plt.subplots(figsize=utils.Defaults.figsize)
    plot_blade(rotor, fig, ax, theta = np.pi / 2, phi = np.pi / 6)

    # configure plot
    ax.set_aspect("equal")
    ax.grid()
    ax.axis("off")

    # create plot of stator
    fig, ax = plt.subplots(figsize = utils.Defaults.figsize)
    plot_blade(stator, fig, ax, theta = 2 * np.pi / 3, phi = np.pi / 6)

    # configure plot
    ax.set_aspect("equal")
    ax.grid()
    ax.axis("off")

if __name__ == "__main__":

    main()
    plt.show()