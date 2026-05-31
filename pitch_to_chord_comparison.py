# pitch_to_chord_comparison.py
# 30 May 2026

# import modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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

    # fan operating point
    phi = 0.7
    psi = 0.25
    M_1 = 0.25
    no_of_stages = 1

    # target diffusion factor
    DF = 0.4

    # min-max
    DF_min = 0.2
    DF_max = 0.6
    N = 200

# main function
def main():

    # create a flight scenario with default parameters
    flight_scenario = Flight_scenario(
        "default",
        utils.Defaults.altitude,
        utils.Defaults.flight_speed,
        utils.Defaults.diameter,
        utils.Defaults.hub_tip_ratio,
        utils.Defaults.thrust
    )

    # create an engine with default parameters
    engine = Engine(
        flight_scenario,
        Inputs.no_of_stages,
        Inputs.phi,
        Inputs.psi,
        utils.Defaults.vortex_exponent,
        utils.Defaults.Y_p,
        utils.Defaults.area_ratio,
        Inputs.M_1
    )
    engine.design()

    # add geometry and calculate blade angles
    engine.geometry = {
        "aspect_ratio": utils.Defaults.aspect_ratio,
        "diffusion_factor": Inputs.DF,
        "design_parameter": utils.Defaults.design_parameter
    }
    engine.empirical_design()

    # create plot
    fig, axes = plt.subplots(1, 3, figsize = utils.Defaults.figsize)

    # retrieve rotor for convenience
    rotor = engine.blade_rows[0]

    # calculate dimensionless span
    span = (rotor.exit.rr - rotor.exit.rr[0]) / (rotor.exit.rr[-1] - rotor.exit.rr[0])

    # plot rotor pitch-to-chord distribution against span
    axes[0].plot(rotor.exit.pitch_to_chord, span, color = "C0", label = f"Design Tool ({rotor.no_of_blades} blades)")
    axes[1].plot(rotor.exit.diffusion_factor, span, color = "C0", label = "Design Tool")
    axes[2].plot(utils.rad_to_deg(rotor.exit.deviation), span, color = "C0", label = "Design Tool")

    # determine pitch-to-chord distribution for constant diffusion factor
    rotor.exit.pitch_to_chord = (
        2 * (
            Inputs.DF - 1 + rotor.exit.M_rel / rotor.inlet.M_rel
            * np.sqrt(rotor.exit.T / rotor.inlet.T)
        ) / np.abs(
            np.sin(rotor.inlet.beta)
            - rotor.exit.M_rel / rotor.inlet.M_rel
            * np.sqrt(rotor.exit.T / rotor.inlet.T)
            * np.sin(rotor.exit.beta)
        )
    )
    rotor.calculate_deviation()

    # plot constant-DF pitch-to-chord distribution
    axes[0].plot(rotor.exit.pitch_to_chord, span, color = "C1", label = "Constant DF")
    axes[1].plot(Inputs.DF * np.ones(len(rotor.exit.rr)), span, color = "C1", label = "Constant DF")
    axes[2].plot(utils.rad_to_deg(rotor.exit.deviation), span, color = "C1", label = "Constant DF")

    # loop through candidate deviation values
    for delta in np.linspace(rotor.exit.deviation[0], 0.001, Inputs.N):

        # calculate pitch-to-chord
        rotor.exit.pitch_to_chord = (
            (
                utils.rad_to_deg(delta) / (
                    (0.23 + np.abs(utils.rad_to_deg(rotor.exit.beta)) / 500)
                    * utils.rad_to_deg(rotor.inlet.beta - delta - rotor.exit.beta)
                )
            )**2
        )

        # recalculate diffusion factor
        rotor.exit.DF = (
            1 - rotor.exit.M_rel / rotor.inlet.M_rel * np.sqrt(rotor.exit.T / rotor.inlet.T)
            + 0.5 * rotor.exit.pitch_to_chord * np.abs(
                np.sin(rotor.inlet.beta)
                - rotor.exit.M_rel / rotor.inlet.M_rel
                * np.sqrt(rotor.exit.T / rotor.inlet.T)
                * np.sin(rotor.exit.beta)
            )
        )

        # if diffusion factor condition has been satisfied
        if np.all(rotor.exit.DF < Inputs.DF):

            # terminate loop
            break

    # recalculate constant deviation
    rotor.calculate_deviation()

    # plot constant-delta pitch-to-chord distribution
    axes[0].plot(rotor.exit.pitch_to_chord, span, color = "C2", label = r"Constant $\delta$")
    axes[1].plot(rotor.exit.DF, span, color = "C2", label = r"Constant $\delta$")
    axes[2].plot(utils.rad_to_deg(rotor.exit.deviation), span, color = "C2", label = r"Constant $\delta$")

    # configure plot
    axes[0].set_xlabel("Pitch-to-chord Ratio")
    axes[1].set_xlabel("Diffusion Factor, DF")
    axes[2].set_xlabel(r"Deviation Angle, $\delta$ (°)")
    axes[0].set_ylabel("Dimensionless Span")
    axes[0].legend(loc = 'center', bbox_to_anchor = (1.08, 0.5), bbox_transform = fig.transFigure, frameon = False)
    fig.suptitle(
        "Comparison of Methodologies for Determining Pitch-to-chord Ratio\n"
        rf"$N$ = {Inputs.no_of_stages}, $M_1$ = {Inputs.M_1}, $\phi$ = {Inputs.phi}, "
        rf"$\psi$ = {Inputs.psi}",
        fontsize = utils.Defaults.titlesize,
        y = 1.03
    )
    axes[1].tick_params(axis = "y", labelleft = False)
    axes[2].tick_params(axis = "y", labelleft = False)

    # loop for each axes
    for ax in axes:

        # set grid
        ax.grid()
        ax.set_ylim(0, 1)

    # save figure
    fig.savefig("exports/pitch_to_chord_comparison.png", dpi = utils.Defaults.dpi, bbox_inches = "tight")

    # create 2nd plot
    fig, ax = plt.subplots(figsize = utils.Defaults.figsize)

    # list of diffusion factors to try
    DF_list = np.linspace(Inputs.DF_min, Inputs.DF_max, Inputs.N)
    N_blades = []

    # loop for each diffusion factor
    for DF in DF_list:

        # add geometry and calculate blade angles
        engine.geometry = {
            "aspect_ratio": utils.Defaults.aspect_ratio,
            "diffusion_factor": DF,
            "design_parameter": utils.Defaults.design_parameter
        }
        engine.empirical_design()

        # append number of blades
        N_blades.append(engine.blade_rows[0].no_of_blades)

    # plot
    ax.plot(DF_list, N_blades, color = "C0")

    # configure plot
    ax.grid()
    ax.set_xlabel("Lieblein Diffusion Factor, DF")
    ax.set_ylabel("Number of Rotor Blades")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(
        "Variation of Rotor Blade Count with Diffusion Factor\n"
        rf"$N$ = {Inputs.no_of_stages}, $M_1$ = {Inputs.M_1}, $\phi$ = {Inputs.phi}, "
        rf"$\psi$ = {Inputs.psi}",
        fontsize = utils.Defaults.titlesize,
        y = 1.03
    )

    # save figure
    fig.savefig("exports/blades_vs_DF.png", dpi = utils.Defaults.dpi, bbox_inches = "tight")

# upon script execution
if __name__ == "__main__":

    # run main and show all plots
    main()
    plt.show()
