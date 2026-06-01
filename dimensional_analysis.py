# dimensional_analysis.py
# 20 May 2026

# import modules
import numpy as np

# import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from matplotlib.colors import ListedColormap

# import high speed solver
from flight_scenario import Flight_scenario
from engine import Engine
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

    # default plotting parameters
    line_width = 1
    line_thin = 0.5

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

def quadratic(p0, p1, p2, no_of_samples=100):
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
    t = np.linspace(0, 1, no_of_samples)

    # Evaluate the quadratic equation for all t values
    # np.outer handles multiplying the t-vectors by the coefficient vectors
    curve = (
        np.outer(t**2, A) + np.outer(t, B) + p0
    )  # p0 acts as C added to every row

    return np.transpose(curve)  # Returns an array of shape (num_samples, 3)^T

def plot_blade(blade_row, tag, ax, theta, phi, lean = 0.4, sweep = 0.03):
    """Plots a 2D projection of a 3D blade shape onto a given axis."""
    # extract chord distribution
    chord = blade_row.exit.chord

    # lists of blade geometries to be populated
    xx = []
    yy = []
    zz = []
    te = []
    le = []
    j_le = []
    #xx_offset = []
    #yy_offset = []

    # loop for each slice produced through the blade row
    for i, index in enumerate(blade_row.indices):

        # extract blade section information
        xx.append(
            blade_row.xx[i] - lean * chord[index] * np.cos(blade_row.inlet.metal_angle[index])
        )
        yy.append(
            blade_row.yy[i] - lean * chord[index] * np.sin(blade_row.inlet.metal_angle[index])
            + sweep * chord[index] * i * (len(blade_row.indices) - 1 - i)
        )
        zz.append(
            np.ones(len(blade_row.xx[i])) * (blade_row.exit.rr[index] - blade_row.exit.rr[0])
        )

        # find trailing edge coordinates
        te.append((xx[i][0], yy[i][0]))

        # find leading edge coordinates
        rr = np.sqrt((xx[i] - xx[i][0])**2 + (yy[i] - yy[i][0])**2)
        j = np.argmax(rr)
        le.append((xx[i][j], yy[i][j]))
        j_le.append(j)

        # for hub index
        if i == 0:

            # plot blade section
            ax.plot(*project(xx[i][:j], yy[i][:j], zz[i][:j], theta, phi), color = "k", lw = Inputs.line_thin)

        # for tip index
        elif i == len(blade_row.indices) - 1:

            ax.plot(*project(xx[i], yy[i], zz[i], theta, phi), color = "k", lw = Inputs.line_thin)

        # mid-span indices
        else:

            # scaling factor
            s = 0.7

            # for rotors
            if blade_row.motor_rpm > 0:

                # get relative velocity vector
                x_rel = xx[i][j] - s * blade_row.inlet.v_x[index]
                y_rel = yy[i][j] - s * blade_row.inlet.v_x[index] * np.tan(blade_row.inlet.beta[index])

                # get absolute velocity vector
                x_abs = x_rel + s * blade_row.inlet.v_x[index]
                y_abs = y_rel - s * blade_row.inlet.v_x[index] * np.tan(blade_row.inlet.alpha[index])

                # draw absolute velocity vector
                ax.annotate(
                    "",
                    xy = project([x_rel], [y_rel], [zz[i][j]], theta, phi),
                    xytext = project([x_abs], [y_abs], [zz[i][j]], theta, phi),
                    arrowprops = dict(arrowstyle = "<|-", color = "C0", lw = Inputs.line_width, shrinkA = 0, shrinkB = 0)
                )
                ax.plot(*project([x_abs, x_rel], [y_abs, y_rel], zz[i][:2], theta, phi), linestyle = "")

                # draw relative velocity vector
                ax.annotate(
                    "",
                    xy = project([x_rel], [y_rel], [zz[i][j]], theta, phi),
                    xytext = project([xx[i][j]], [yy[i][j]], [zz[i][j]], theta, phi),
                    arrowprops = dict(arrowstyle = "<|-", color = "C4", lw = Inputs.line_width, shrinkA = 0, shrinkB = 0)
                )

                # draw blade speed vector
                ax.annotate(
                    "",
                    xy = project([xx[i][j]], [yy[i][j]], [zz[i][j]], theta, phi),
                    xytext = project([x_abs], [y_abs], [zz[i][j]], theta, phi),
                    arrowprops = dict(arrowstyle = "<|-", color = "C3", lw = Inputs.line_width, shrinkA = 0, shrinkB = 0)
                )

                # repeat for exit
                x_rel = xx[i][0] + s * blade_row.exit.v_x[index]
                y_rel = yy[i][0] + s * blade_row.exit.v_x[index] * np.tan(blade_row.exit.beta[index])

                # get absolute velocity vector
                x_abs = xx[i][0] + s * blade_row.exit.v_x[index]
                y_abs = yy[i][0] + s * blade_row.exit.v_x[index] * np.tan(blade_row.exit.alpha[index])

                # draw absolute velocity vector
                ax.annotate(
                    "",
                    xy = project([xx[i][0]], [yy[i][0]], [zz[i][0]], theta, phi),
                    xytext = project([x_abs], [y_abs], [zz[i][0]], theta, phi),
                    arrowprops = dict(arrowstyle = "<|-", color = "C0", lw = Inputs.line_width, shrinkA = 0, shrinkB = 0)
                )

                # draw relative velocity vector
                ax.annotate(
                    "",
                    xy = project([xx[i][0]], [yy[i][0]], [zz[i][0]], theta, phi),
                    xytext = project([x_rel], [y_rel], [zz[i][0]], theta, phi),
                    arrowprops = dict(arrowstyle = "<|-", color = "C4", lw = Inputs.line_width, shrinkA = 0, shrinkB = 0)
                )

                # draw blade speed vector
                ax.annotate(
                    "",
                    xy = project([x_rel], [y_rel], [zz[i][0]], theta, phi),
                    xytext = project([x_abs], [y_abs], [zz[i][0]], theta, phi),
                    arrowprops = dict(arrowstyle = "<|-", color = "C3", lw = Inputs.line_width, shrinkA = 0, shrinkB = 0)
                )
                ax.plot(*project([x_abs, x_rel], [y_abs, y_rel], zz[i][:2], theta, phi), linestyle = "")

            # for stators
            else:

                # get absolute velocity vector
                x_abs = xx[i][j] - s * blade_row.inlet.v_x[index]
                y_abs = yy[i][j] - s * blade_row.inlet.v_x[index] * np.tan(blade_row.inlet.alpha[index])

                # draw absolute velocity vector
                ax.annotate(
                    "",
                    xy = project([x_abs], [y_abs], [zz[i][j]], theta, phi),
                    xytext = project([xx[i][j]], [yy[i][j]], [zz[i][j]], theta, phi),
                    arrowprops = dict(arrowstyle = "<|-", color = "C0", lw = Inputs.line_width, shrinkA = 0, shrinkB = 0)
                )
                ax.plot(*project([x_abs, xx[i][j]], [y_abs, yy[i][j]], zz[i][:2], theta, phi), linestyle = "")

                # repeat for exit
                x_abs = xx[i][0] + s * blade_row.exit.v_x[index]
                y_abs = yy[i][0] + s * blade_row.exit.v_x[index] * np.tan(blade_row.exit.alpha[index])

                # draw absolute velocity vector
                ax.annotate(
                    "",
                    xy = project([xx[i][0]], [yy[i][0]], [zz[i][0]], theta, phi),
                    xytext = project([x_abs], [y_abs], [zz[i][0]], theta, phi),
                    arrowprops = dict(arrowstyle = "<|-", color = "C0", lw = Inputs.line_width, shrinkA = 0, shrinkB = 0)
                )
                ax.plot(*project([x_abs, xx[i][0]], [y_abs, yy[i][0]], zz[i][:2], theta, phi), linestyle = "")

    # plot shadow
    x, y = offset(xx[0], yy[0], -0.1)
    x, y = project(x, y, zz[0], theta, phi)
    vertices = np.transpose(np.vstack((x, y)))
    polygon = patches.Polygon(vertices, closed = True, facecolor = "k", alpha = 0.5, linestyle = "")
    ax.add_patch(polygon)

    # construct spline between leading edges
    xx_le, yy_le, zz_le = quadratic(
        [le[0][0], le[0][1], zz[0][0]],
        [le[2][0], le[2][1], zz[2][0]],
        [le[-1][0], le[-1][1], zz[-1][0]]
    )

    # construct spline between trailing edges
    xx_te, yy_te, zz_te = quadratic(
        [te[0][0], te[0][1], zz[0][0]],
        [te[2][0], te[2][1], zz[2][0]],
        [te[-1][0], te[-1][1], zz[-1][0]]
    )

    # prepare grid for shading contours
    M = 10
    N = 100
    X_grid = np.zeros((M, N))
    Y_grid = np.zeros((M, N))
    Z_grid = np.zeros((M, N))
    
    #
    indices = np.linspace(0, j_le[0] - 1, M).astype(int)

    for i, index in enumerate(indices):

        x, y, z = quadratic(
            [xx[0][index], yy[0][index], zz[0][index]],
            [xx[2][index], yy[2][index], zz[2][index]],
            [xx[-1][index], yy[-1][index], zz[-1][index]],
            no_of_samples = N
        )

        # store results in grid
        X_grid[i, :] = x
        Y_grid[i, :] = y
        Z_grid[i, :] = z

    # calculate derivatives
    dz_du, dz_dv = np.gradient(Z_grid)
    dy_du, dy_dv = np.gradient(Y_grid)
    dx_du, dx_dv = np.gradient(X_grid)

    # 2. Reconstruct the tangent vectors]
    u = np.stack([dx_du, dy_du, dz_du], axis=-1)
    # Tangent vector V (along columns)
    v = np.stack([dx_dv, dy_dv, dz_dv], axis=-1)

    # compute cross product to get the normal vectors
    normals = np.cross(u, v)

    # normalise the vectors
    magnitude = np.linalg.norm(normals, axis=-1, keepdims=True)
    magnitude[magnitude == 0] = 1.0 
    normals = normals / magnitude

    # retrieve x-component    
    x_component = normals[..., 0]

    # plot contours
    N = 10
    grey = matplotlib.colormaps["gray"]
    colors = grey(np.linspace(0.7, 0.95, N))
    grey_map = ListedColormap(colors)
    ax.contourf(*project(X_grid, Y_grid, Z_grid, theta, phi), x_component, levels = N, cmap = grey_map)

    # plot leading and trailing edge splines
    ax.plot(*project(xx_le, yy_le, zz_le, theta, phi), color = "k", lw = Inputs.line_thin)
    ax.plot(*project(xx_te, yy_te, zz_te, theta, phi), color = "k", lw = Inputs.line_thin)

    # define origin and unit vectors
    origin = np.array([
        le[0][0] + 2 * chord[0] * np.cos(blade_row.inlet.metal_angle[0]),
        le[0][1] + 2 * chord[0] * np.sin(blade_row.inlet.metal_angle[0]), 0.6 * chord[0]
    ])

    # define array of unit vectors
    axes = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    labels = ["x", r"r$\theta$", "r"]

    # loop for each unit vector
    for axis, label in zip(axes, labels):

        # calculate start and end positions and plot projected points
        vector = np.transpose([origin, origin + 0.1 * axis])
        x, y = project(*vector, theta, phi)

        # plot coordinates spanned by arrow to force bounding box to encompass arrows
        ax.plot(x, y, linestyle = "")

        # annotate arrow corresponding to unit vector
        ax.annotate(
            "",
            xy=(x[0], y[0]),
            xytext=(x[1], y[1]),
            arrowprops=dict(arrowstyle = "<|-", color = "k", lw = Inputs.line_width, shrinkA = 0, shrinkB = 0)
        )

        # recalculate text positions
        vector = np.transpose([origin, origin + 0.12 * axis])
        x, y = project(*vector, theta, phi)

        if y[1] < y[0]:

            y[1] -= 2e-2

        # add arrow text
        ax.text(x[1], y[1], f"{label}", va = 'center', ha = 'left')

    x, y = project(*le[0], 0, theta, phi)
    ax.text(x + 1e-2, y + 1e-2, f"{tag}")

def annotate(ax, label, position):
    
    #
    ax.annotate(
        label,
        xy = position,
        xytext = (0, 0),
		textcoords = "offset points",
        color = "k"
    )

    # update axis limits
    ax.plot(*position, linestyle = "")

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
    fig, ax = plt.subplots(figsize = utils.Defaults.figsize)
    plot_blade(rotor, "PS", ax, theta = np.pi / 3, phi = np.pi / 8)

    # configure plot
    ax.set_aspect("equal")
    ax.axis("off")

    # annotate rotor plot
    annotate(ax, r"$r_\text{1,tip}$", (-0.19, 0.60))
    annotate(ax, r"$r_\text{2,tip}$", (0.16, 0.62))
    annotate(ax, r"$r_\text{1,hub}$", (-0.3, -0.07))
    annotate(ax, r"$r_\text{2,hub}$", (0.29, -0.05))
    annotate(ax, r"$v_{x1}(r)$", (-0.5, 0.31))
    annotate(ax, r"$U_1(r)$", (-0.23, 0.36))
    annotate(ax, r"$\Delta h_0 (r)$", (-0.05, 0.35))
    annotate(ax, r"$\Delta s(r)$", (-0.03, 0.5))

    # save figure
    fig.savefig("exports/dimensional_analysis_rotor.png", dpi = utils.Defaults.dpi, bbox_inches = "tight")

    # create plot of stator
    fig, ax = plt.subplots(figsize = utils.Defaults.figsize)
    plot_blade(stator, "SS", ax, theta = 2 * np.pi / 3, phi = np.pi / 8)

    # configure plot
    ax.set_aspect("equal")
    ax.axis("off")

    # annotate stator plot
    annotate(ax, r"$\alpha_3(r)$", (0.17, 0.42))
    annotate(ax, r"$\Delta s(r)$", (-0.02, 0.4))

    # save figure
    fig.savefig("exports/dimensional_analysis_stator.png", dpi = utils.Defaults.dpi, bbox_inches = "tight")

if __name__ == "__main__":

    main()
    plt.show()