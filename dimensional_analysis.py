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

def project_points(x, y, z, azimuth, elevation):
    """
    Project 3D points onto a 2D plane for a given viewing direction.

    Parameters
    ----------
    x, y, z : array_like
        Arrays of point coordinates (same shape).

    azimuth : float
        Rotation about global z-axis [radians].

    elevation : float
        Elevation angle above xy-plane [radians].

    Returns
    -------
    xp, yp : ndarray
        2D projected coordinates.
    """

    # Convert to arrays
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # Stack into (N,3)
    p = np.stack([x, y, z], axis=-1)

    # Rotation about z-axis (azimuth)
    ca = np.cos(azimuth)
    sa = np.sin(azimuth)

    Rz = np.array([
        [ ca, -sa, 0],
        [ sa,  ca, 0],
        [  0,   0, 1]
    ])

    # Rotation about x-axis (elevation)
    ce = np.cos(elevation)
    se = np.sin(elevation)

    Rx = np.array([
        [1,  0,   0],
        [0, ce, -se],
        [0, se,  ce]
    ])

    # Combined rotation
    R = Rx @ Rz

    # Rotate points
    p_rot = p @ R.T

    # Orthographic projection:
    # keep x/y, discard z
    xp = p_rot[..., 0]
    yp = p_rot[..., 1]

    return xp, yp

def project_points2(x, y, z, theta, phi):
    """
    Rotate about z-axis by theta, then rotate about an axis in the xy-plane
    (rotated x-axis) by phi, and return 2D projection (xp, yp).
    """

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    pts = np.stack([x, y, z], axis=-1)

    # ------------------------------------------------------------
    # Step 1: rotation about z-axis
    # ------------------------------------------------------------
    ct = np.cos(theta)
    st = np.sin(theta)

    Rz = np.array([
        [ct, -st, 0.0],
        [st,  ct, 0.0],
        [0.0, 0.0, 1.0]
    ])

    pts = pts @ Rz.T

    # ------------------------------------------------------------
    # Step 2: rotated x-axis (axis in xy-plane)
    # ------------------------------------------------------------
    axis = np.array([ct, st, 0.0])
    axis = axis / np.linalg.norm(axis)

    # ------------------------------------------------------------
    # Step 3: Rodrigues rotation about axis (phi)
    # ------------------------------------------------------------
    c = np.cos(phi)
    s = np.sin(phi)
    C = 1.0 - c

    ux, uy, uz = axis

    R = np.array([
        [c + ux*ux*C,      ux*uy*C - uz*s, ux*uz*C + uy*s],
        [uy*ux*C + uz*s,   c + uy*uy*C,    uy*uz*C - ux*s],
        [uz*ux*C - uy*s,   uz*uy*C + ux*s, c + uz*uz*C]
    ])

    pts = pts @ R.T

    # ------------------------------------------------------------
    # 2D projection (orthographic)
    # ------------------------------------------------------------
    xp = pts[..., 0]
    yp = pts[..., 1]

    return xp, yp

def project(xx, yy, zz, theta = np.pi / 4, psi = 0):

    # get projection vector
    n = np.array([np.cos(theta), np.sin(theta), np.sin(psi)])
    n = n / np.linalg.norm(n)

    # reference direction (z-axis projected onto plane)
    k = np.array([0.0, 0.0, 1.0])
    k_proj = k - np.dot(k, n) * n

    pts = np.stack([xx, yy, zz], axis=-1)
    shape = pts.shape[:-1]
    pts = pts.reshape(-1, 3)

    # plane normal
    e2 = n / np.linalg.norm(n)

    # world vertical
    k = np.array([0.0, 0.0, 1.0])

    # in-plane "vertical direction"
    e1 = k - np.dot(k, e2) * e2
    if np.linalg.norm(e1) < 1e-8:
        e1 = np.array([1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)

    # second in-plane axis
    e0 = np.cross(e2, e1)

    # 2D coordinates in plane
    x2d = pts @ e1
    y2d = pts @ e0

    x2d = x2d.reshape(shape)
    y2d = y2d.reshape(shape)

    return x2d, y2d

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

def plot_blade(blade_row, fig, ax, theta = np.pi / 6, phi = -np.pi / 3, psi = 0, overlap = 0.4):
    """Plots a 2D projection of a 3D blade shape onto a given axis."""
    # extract chord distribution
    chord = blade_row.exit.chord

    # extract blade_row hub blade geometry
    xx_hub = blade_row.xx[0] - overlap * chord[0] * np.cos(blade_row.inlet.metal_angle[0])
    yy_hub = blade_row.yy[0] - overlap * chord[0] * np.sin(blade_row.inlet.metal_angle[0])
    zz_hub = np.zeros_like(xx_hub)

    # extract blade_row tip blade geometry
    xx_tip = blade_row.xx[-1] - overlap * chord[-1] * np.cos(blade_row.inlet.metal_angle[-1])
    yy_tip = blade_row.yy[-1] - overlap * chord[-1] * np.sin(blade_row.inlet.metal_angle[-1])
    zz_tip = np.ones_like(xx_tip) * (blade_row.exit.rr[-1] - blade_row.exit.rr[0])

    # find trailing edge coordinates
    te_hub = (xx_hub[0], yy_hub[0])
    te_tip = (xx_tip[0], yy_tip[0])

    # find leading edge coordinates
    rr_hub = np.sqrt((xx_hub - xx_hub[0])**2 + (yy_hub - yy_hub[0])**2)
    i = np.argmax(rr_hub)
    le_hub = (xx_hub[i], yy_hub[i])
    rr_tip = np.sqrt((xx_tip - xx_tip[0])**2 + (yy_tip - yy_tip[0])**2)
    i = np.argmax(rr_tip)
    le_tip = (xx_tip[i], yy_tip[i])

    # construct spline between leading edges
    xx_le, yy_le, zz_le = vertical_end_spline(
        [le_hub[0], le_hub[1], zz_hub[0]],
        [le_tip[0], le_tip[1], zz_tip[0]]
    )

    # construct spline between trailing edges
    xx_te, yy_te, zz_te = vertical_end_spline(
        [te_hub[0], te_hub[1], zz_hub[0]],
        [te_tip[0], te_tip[1], zz_tip[0]]
    )

    # clip hub section by intersects with chord
    index = next(
        i for i, x in enumerate(xx_hub)
        if (np.abs(xx_hub[i] - le_hub[0]) < 1e-6) and (np.abs(yy_hub[i] - le_hub[1]) < 1e-6)
    )
    xx_hub = xx_hub[index:]
    yy_hub = yy_hub[index:]
    zz_hub = zz_hub[index:]

    # determine projection of datapoints on an angled plane
    xx_proj_le, yy_proj_le = project_points(xx_le, yy_le, zz_le, theta, phi)
    xx_proj_te, yy_proj_te = project_points(xx_te, yy_te, zz_te, theta, phi)
    xx_proj_hub, yy_proj_hub = project_points(xx_hub, yy_hub, zz_hub, theta, phi)
    xx_proj_tip, yy_proj_tip = project_points(xx_tip, yy_tip, zz_tip, theta, phi)

    # plot 3D data
    ax.plot(xx_proj_le, yy_proj_le, color = "k")
    ax.plot(xx_proj_te, yy_proj_te, color = "k")
    ax.plot(xx_proj_hub, yy_proj_hub, color = "k")
    ax.plot(xx_proj_tip, yy_proj_tip, color = "k")

    # define origin and unit vectors
    origin = np.array([1.5 * chord[0], 0, 0.5 * (blade_row.exit.rr[-1] - blade_row.exit.rr[0])])
    origin = np.array([0, 0, 0])

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
        xx, yy = project_points(*vector, theta, phi)

        ax.plot(xx, yy, linestyle = "")

        ax.annotate(
            f"{label}",
            xy=(xx[0], yy[0]),
            xytext=(xx[1], yy[1]),
            arrowprops=dict(arrowstyle="<|-", color="C0", lw=1.5, shrinkA=0, shrinkB=0)
        )

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
    plot_blade(rotor, fig, ax)

    # configure plot
    ax.set_aspect("equal")
    ax.grid()
    ax.axis("off")

    # create plot of stator
    fig, ax = plt.subplots(figsize = utils.Defaults.figsize)
    plot_blade(stator, fig, ax)

    # configure plot
    ax.set_aspect("equal")
    ax.grid()
    ax.axis("off")

if __name__ == "__main__":

    main()
    plt.show()