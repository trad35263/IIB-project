# import modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from time import perf_counter as timer

# import high speed solver
from engine import Engine
from flight_scenario import Flight_scenario
import utils

# import matlab
import matlab.engine
from pathlib import Path as FilePath
import io

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

    # matlab script filename
    filename = "DuctedFanDesign_220503"

    # path to matlab_folder
    python_dir = FilePath(__file__).resolve().parent
    parent_dir = python_dir.parent.parent
    matlab_dir = parent_dir / "forSlava"

    alpha = 0.6

def run_matlab():
    """Creates a spanwise flow angle plot with matlab results overlaid."""
    # run matlab engine
    eng = matlab.engine.start_matlab()
    eng.eval("set(0, 'DefaultFigureVisible', 'off')", nargout=0)

    # add MATLAB folder to MATLAB search path
    eng.addpath(str(Inputs.matlab_dir), nargout = 0)

    # create dummy buffers
    out = io.StringIO()
    err = io.StringIO()

    # run script rejecting terminal outputs
    print(
        f"Running MATLAB script {utils.Colours.GREEN}{Inputs.filename}{utils.Colours.END} "
        f"in folder {utils.Colours.GREEN}{Inputs.matlab_dir}{utils.Colours.END}!"
    )
    t1 = timer()
    eng.run(Inputs.filename, nargout = 0, stdout = out, stderr = err)
    t2 = timer()
    print(f"MATLAB script completed in {utils.Colours.GREEN}{t2 - t1:.4g}{utils.Colours.END} s!")

    # extract information
    a = eng.workspace["a"]
    d = eng.workspace["d"]
    g = eng.workspace["g"]

    # convert to numpy arrays
    alpha = np.array(a["alpha"], dtype = float)
    v_x = np.array(a["vx"], dtype = float)
    chi = np.array(g["chi"], dtype = float)

    # store variables for later
    Inputs.m_dot = np.array(eng.workspace["mdot"], dtype = float)[0][0]
    Inputs.rho = np.array(eng.workspace["ro"], dtype = float)[0][0]
    Inputs.rr = np.transpose(np.array(d["r"], dtype = float))[0]

    # extract flight scenario specification
    Inputs.altitude = np.array(eng.workspace["ht"], dtype = float)[0][0]
    Inputs.flight_speed = np.array(eng.workspace["v0"], dtype = float)[0][0]
    Inputs.diameter = np.array(eng.workspace["D"], dtype = float)
    Inputs.thrust = np.array(eng.workspace["T"], dtype = float)[0][0]
    Inputs.hub_tip_ratio = np.array(eng.workspace["q"], dtype = float)

    # extract engine specification
    Inputs.phi = np.array(eng.workspace["phi"], dtype = float)
    Inputs.psi = np.array(eng.workspace["psi"], dtype = float)[0][0]
    Inputs.n = np.array(eng.workspace["n"], dtype = float)

    # extract blade design information
    Inputs.DF = np.array(eng.workspace["DF"], dtype = float)
    Inputs.AR = np.array(eng.workspace["AR"], dtype = float)

    # shut down engine
    eng.quit()

    # store axial velocities
    Inputs.v_x_1 = v_x[:, 0, 0]
    Inputs.v_x_2 = v_x[:, 1, 0]
    Inputs.v_x_3 = v_x[:, 2, 0]

    # store rotor angles
    Inputs.beta_1 = alpha[:, 0, 1]
    Inputs.beta_2 = alpha[:, 1, 1]
    Inputs.chi_2 = chi[:, 1, 0]

    # store stator angles
    Inputs.alpha_2 = alpha[:, 1, 2]
    Inputs.alpha_3 = alpha[:, 2, 2]
    Inputs.chi_3 = chi[:, 1, 1]

    return

    # get array of radii from array of spans
    rr = Inputs.hub_tip_ratio + (1 - Inputs.hub_tip_ratio) * xx

    # store matlab flow angles
    self.blade_rows[0].inlet.matlab_rel_angle = np.interp(xx, xx_matlab, alpha[:, 0, 1])
    self.blade_rows[0].exit.matlab_rel_angle = np.interp(xx, xx_matlab, alpha[:, 1, 1])
    self.blade_rows[0].exit.matlab_angle = np.interp(xx, xx_matlab, alpha[:, 1, 2])

    # store matlab axial velocity
    self.blade_rows[0].inlet.matlab_v_x = np.interp(xx, xx_matlab, v_x[:, 0, 0])
    self.blade_rows[0].exit.matlab_v_x = np.interp(xx, xx_matlab, v_x[:, 1, 0])
    self.blade_rows[1].exit.matlab_v_x = np.interp(xx, xx_matlab, v_x[:, 2, 0])

    # loop over all blade rows
    for blade_row in self.blade_rows:

        # calculate cumulative (dimensional) mass flow rate
        dm_dr = 2 * rho * blade_row.exit.matlab_v_x * rr * self.scenario.A / (1 - self.hub_tip_ratio**2)
        blade_row.exit.m_dot_matlab = utils.cumulative_trapezoid(dm_dr, rr)
        blade_row.exit.m_dot_kg_s = (
            blade_row.exit.m_dot * self.scenario.A * self.scenario.p_0 / np.sqrt(utils.c_p * self.scenario.T_0)
        )

        # calculate dimensional density variation
        blade_row.exit.rho_kg_m_3 = (
            blade_row.exit.p / (utils.R * blade_row.exit.T) * self.scenario.p_0 / self.scenario.T_0
        )
        blade_row.exit.rho_matlab = rho * np.ones(utils.Defaults.solver_grid)

        # calculate solver flow velocity in m_s
        blade_row.exit.v_x_m_s = blade_row.exit.v_x * np.sqrt(utils.gamma * utils.R * self.scenario.T_0)

    # calculate cumulative (dimensional) mass flow rate at inlet
    dm_dr = 2 * rho * self.blade_rows[0].inlet.matlab_v_x * rr * self.scenario.A / (1 - self.hub_tip_ratio**2)
    self.blade_rows[0].inlet.m_dot_matlab = utils.cumulative_trapezoid(dm_dr, rr)
    self.blade_rows[0].inlet.m_dot_kg_s = (
        self.blade_rows[0].inlet.m_dot * self.scenario.A * self.scenario.p_0 / np.sqrt(utils.c_p * self.scenario.T_0)
    )

    # calculate dimensional density variation at inlet
    self.blade_rows[0].inlet.rho_kg_m_3 = (
        self.blade_rows[0].inlet.p / (utils.R * self.blade_rows[0].inlet.T) * self.scenario.p_0 / self.scenario.T_0
    )

    # calculate solver flow velocity in m/s at inlet
    self.blade_rows[0].inlet.v_x_m_s = (
        self.blade_rows[0].inlet.v_x * np.sqrt(utils.gamma * utils.R * self.scenario.T_0)
    )

    # define quantities to plot
    quantities = [
        [
            'matlab_angle', 'MatLab absolute flow angle',
            'matlab_rel_angle', 'MatLab relative flow angle',
            'alpha', 'Absolute flow angle (°)',
            'beta', 'Relative flow angle (°)'
        ],
        [
            'matlab_v_x', 'MatLab axial flow velocity (m/s)',
            'v_x_m_s', 'Axial flow velocity (m/s)'
        ],
        [
            'm_dot_matlab', 'MatLab cumulative mass flow rate (kg/s)',
            'm_dot_kg_s', 'Cumulative mass flow rate (kg/s)'
        ],
        [
            'rho_matlab', 'MatLab density (kg/m^3)',
            'rho_kg_m_3', 'Density (kg/m^3)'
        ]
    ]

    # plot spanwise variation
    #self.plot_spanwise(quantities)

def span(rr):

    return (rr - rr[0]) / (rr[-1] - rr[0])

# main function
def main():

    # run matlab script
    run_matlab()

    # create Flight_scenario
    flight_scenario = Flight_scenario(
        "", Inputs.altitude, Inputs.flight_speed, Inputs.diameter, Inputs.hub_tip_ratio,
        Inputs.thrust
    )

    # calculate matlab inlet mach number
    v_c_p_T_0 = Inputs.v_x_1[0] / np.sqrt(utils.c_p * flight_scenario.T_0)
    Inputs.M_1 = utils.inverse_velocity_function(v_c_p_T_0)

    print(f"Inputs.M_1: {Inputs.M_1}")

    # create Engine
    engine = Engine(flight_scenario, 1, [Inputs.phi], [Inputs.psi], Inputs.n, 0, 1, Inputs.M_1)
    engine.design()

    # define engine blade geometry
    engine.geometry = {
        "aspect_ratio": Inputs.AR,
        "diffusion_factor": Inputs.DF,
        "design_parameter": 1
    }
    engine.empirical_design()

    # comparison plot of axial velocity
    """fig, axes = plt.subplots(1, 3, figsize = utils.Defaults.figsize, sharex = True)

    # plot design tool axial velocity distributions against span
    axes[0].plot(
        engine.blade_rows[0].inlet.v_x * np.sqrt(utils.gamma * utils.R * engine.scenario.T_0),
        span(engine.blade_rows[0].inlet.rr), alpha = Inputs.alpha
    )
    axes[1].plot(
        engine.blade_rows[0].exit.v_x * np.sqrt(utils.gamma * utils.R * engine.scenario.T_0),
        span(engine.blade_rows[0].exit.rr), alpha = Inputs.alpha
    )
    axes[2].plot(
        engine.blade_rows[1].exit.v_x * np.sqrt(utils.gamma * utils.R * engine.scenario.T_0),
        span(engine.blade_rows[1].exit.rr), alpha = Inputs.alpha
    )"""

    # plot matlab code axial velocity distributions against span
    xx = np.linspace(0, 1, Inputs.v_x_1.shape[0])
    """axes[0].plot(Inputs.v_x_1, xx, linestyle = "--", color = "C0")
    axes[1].plot(Inputs.v_x_2, xx, linestyle = "--", color = "C0")
    axes[2].plot(Inputs.v_x_3, xx, linestyle = "--", color = "C0")"""

    # calculate matlab exit mass flow rate:
    matlab_m_dot = (
        2 * np.pi * Inputs.rho * utils.cumulative_trapezoid(
            Inputs.rr, Inputs.rr * Inputs.v_x_3
        )[-1]
    )
    m_dot_matlab_error = 100 * np.abs(matlab_m_dot / Inputs.m_dot - 1)

    # retrieve design tool mass flow rate and error
    m_dot_design = engine.m_dot
    m_dot_design_error = 100 * np.abs(
        engine.blade_rows[-1].exit.m_dot[-1] / engine.blade_rows[0].inlet.m_dot[-1] - 1
    )

    # add title comparing mass flow rates
    """fig.suptitle(
        f"MATLAB inlet mass flow rate: {Inputs.m_dot:.4g} kg/s, "
        f"outlet error %: {m_dot_matlab_error}\n"
        f"Design tool inlet mass flow rate: {m_dot_design:.4g} kg/s, "
        f"outlet error %: {m_dot_design_error:.4g}"
    )

    # loop for each axis
    for ax in axes:

        # add grid
        ax.grid()"""

    # comparison plot of flow and metal angles
    fig, axes = plt.subplots(1, 3, figsize = utils.Defaults.figsize, sharex = True)

    # plot design tool flow and metal angle distributions across span
    axes[0].plot(
        utils.rad_to_deg(engine.blade_rows[0].inlet.beta),
        span(engine.blade_rows[0].inlet.rr),
        color = "C1", alpha = Inputs.alpha
    )
    axes[1].plot(
        utils.rad_to_deg(engine.blade_rows[0].exit.alpha),
        span(engine.blade_rows[0].exit.rr),
        color = "C0", alpha = Inputs.alpha
    )
    axes[1].plot(
        utils.rad_to_deg(engine.blade_rows[0].exit.beta),
        span(engine.blade_rows[0].exit.rr),
        color = "C1", alpha = Inputs.alpha
    )
    axes[1].plot(
        utils.rad_to_deg(engine.blade_rows[0].exit.metal_angle),
        span(engine.blade_rows[0].exit.rr),
        color = "C2", alpha = Inputs.alpha
    )
    axes[2].plot(
        utils.rad_to_deg(engine.blade_rows[1].exit.alpha),
        span(engine.blade_rows[0].exit.rr),
        color = "C0", alpha = Inputs.alpha
    )
    axes[2].plot(
        utils.rad_to_deg(engine.blade_rows[1].exit.metal_angle),
        span(engine.blade_rows[0].exit.rr),
        color = "C2", alpha = Inputs.alpha
    )

    # plot matlab code flow and metal angle distributions against span
    axes[0].plot(Inputs.beta_1, xx, linestyle = "--", color = "C1")
    axes[1].plot(Inputs.beta_2, xx, linestyle = "--", color = "C1")
    axes[1].plot(Inputs.chi_2, xx, linestyle = "--", color = "C2")
    axes[1].plot(Inputs.alpha_2, xx, linestyle = "--", color = "C0")
    axes[2].plot(Inputs.alpha_3, xx, linestyle = "--", color = "C0")
    axes[2].plot(Inputs.chi_3, xx, linestyle = "--", color = "C2")
    
    # add axis labels
    axes[0].set_ylabel("Dimensionless Span", fontsize = utils.Defaults.fontsize)
    axes[1].set_xlabel("Angle (°)")
    #fig.supxlabel("Angle (°)", y = -0.05, fontsize = utils.Defaults.fontsize)
    axes[1].tick_params(axis = "y", labelleft = False)
    axes[2].tick_params(axis = "y", labelleft = False)

    # loop for each axis
    for ax in axes:

        # set grid, y-axis limits, and tick label font size
        ax.grid()
        ax.set_ylim(0, 1)
        ax.tick_params(axis = "both", labelsize = utils.Defaults.fontsize)

    # create custom legend handles for ordering purposes
    custom_handles = [
        Line2D([0], [0], color = "C0", alpha = Inputs.alpha, label = "Flow angle"),
        Line2D([0], [0], color = "C0", linestyle = "--", label = "(Hirono)"),
        Line2D([0], [0], color = "C1", alpha = Inputs.alpha, label = "Relative flow angle"),
        Line2D([0], [0], color = "C1", linestyle = "--", label = "(Hirono)"),
        Line2D([0], [0], color = "C2", alpha = Inputs.alpha, label = "Metal angle"),
        Line2D([0], [0], color = "C2", linestyle = "--", label = "(Hirono)"),
    ]

    # make room for legend and add to plot
    axes[0].legend(
        handles = custom_handles,
        loc = 'center', bbox_to_anchor = (1.08, 0.5), bbox_transform = fig.transFigure, frameon = False
    )

    # set axis titles
    axes[0].set_title("Fan Inlet", fontsize = utils.Defaults.fontsize)
    axes[1].set_title("Rotor Exit", fontsize = utils.Defaults.fontsize)
    axes[2].set_title("Stator Exit", fontsize = utils.Defaults.fontsize)
    
    # figure title
    fig.suptitle(
        r"Design Tool vs. Incompressible Tool by Hirono $et$ $al.$" + "\n"
        rf"$N$ = {engine.no_of_stages}, $M_1$ = {Inputs.M_1:.4g}, $\phi$ = {Inputs.phi}, "
        rf"$\psi$ = {Inputs.psi:.4g}",
        fontsize = utils.Defaults.titlesize, y = 1.1
    )

    # save figure
    fig.savefig("exports/hirono_validation.png", dpi = utils.Defaults.dpi, bbox_inches = "tight")

# upon script execution
if __name__ == "__main__":

    # run main() and show all plots
    main()
    plt.show()
