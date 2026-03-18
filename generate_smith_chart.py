# import modules
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter as timer
from scipy.interpolate import RegularGridInterpolator
import operator

# import high speed solver
import utils
from engine import Engine
from flight_scenario import Flight_scenario

# Inputs class
class Inputs:

    # initialise empty list for storing engines
    engines = []

    # number of grid points for interpolation
    N = 100

    # thrust for initial guesses
    thrust = 30
    
    # variables to loop over
    stages = [1]
    phis = np.linspace(0.4, 1, 7)
    psis = np.linspace(0.05, 0.4, 8)

    # inlet Mach number
    M_1 = 0.15

    # set guardrails to exclude bad engine designs
    max_no_of_blades = utils.Defaults.max_no_of_blades
    max_deviation = np.pi / 4
    max_chord = 1
    max_diffusion_factor = 0.8

# create_engine function
def create_engine(
        altitude = utils.Defaults.altitude,
        flight_speed = utils.Defaults.flight_speed,
        diameter = utils.Defaults.diameter,
        hub_tip_ratio = utils.Defaults.hub_tip_ratio,
        thrust = utils.Defaults.thrust,
        no_of_stages = utils.Defaults.no_of_stages,
        phi = utils.Defaults.phi,
        psi = utils.Defaults.psi,
        vortex_exponent = utils.Defaults.vortex_exponent,
        Y_p = utils.Defaults.Y_p,
        area_ratio = utils.Defaults.area_ratio,
        M_1 = None
    ):
    """Creates an engine with custom parameters."""
    # start timer
    t1 = timer()
    
    # create flight scenario
    flight_scenario = Flight_scenario(
        "", altitude, flight_speed, diameter, hub_tip_ratio, thrust
    )
    
    # create engine
    engine = Engine(
        flight_scenario, no_of_stages, phi, psi, vortex_exponent, Y_p, area_ratio, M_1
    )
    engine.design()

    # end timer
    t2 = timer()
    print(f"Test engine created in {utils.Colours.GREEN}{t2 - t1:.4g}{utils.Colours.END} s.")

    return engine

def main():

    for no_of_stages in Inputs.stages:

        Inputs.no_of_stages = no_of_stages

        for phi in Inputs.phis:

            Inputs.phi = phi

            for psi in Inputs.psis:

                Inputs.psi = psi

                # get engine
                export_engine()

        # create contour plots
        plot_contours(no_of_stages, "no_of_blades", "No. of Blades")
        plot_contours(no_of_stages, "exit.diffusion_factor", "Max. Diffusion Factor")
        plot_contours(no_of_stages, "exit.deviation", "Max. Deviation (rad)")

def plot_contours(no_of_stages, attribute, label = ""):

    # Collect unique phi and psi values
    phis = Inputs.phis
    psis = Inputs.psis

    # define "getter" to retrieve nested class attributes
    getter = operator.attrgetter(attribute)

    # 2D array of z-values
    Z = np.array([
        [
            next(
                np.max([getter(blade_row) for blade_row in engine.blade_rows])
                for engine in Inputs.engines
                if engine.no_of_stages == no_of_stages
                and engine.phi[0] == phi
                and engine.psi[0] == psi
            )
            for phi in phis
        ]
        for psi in psis
    ])

    # interpolate onto a finer grid
    interp = RegularGridInterpolator((psis, phis), Z, method="linear")

    phi_fine = np.linspace(min(phis), max(phis), Inputs.N)
    psi_fine = np.linspace(min(psis), max(psis), Inputs.N)
    PHI, PSI = np.meshgrid(phi_fine, psi_fine)
    Z_fine = interp((PSI, PHI))

    # Plot contour
    fig, ax = plt.subplots()
    cf = ax.contourf(PHI, PSI, Z_fine, levels = 20, cmap = "viridis")
    cs = ax.contour(PHI, PSI, Z_fine, levels = 20, colors = "white", linewidths = 0.4, alpha = 0.4)
    cbar = fig.colorbar(cf, ax = ax)
    cbar.set_label(f"{label}")

    ax.set_xlabel("Flow Coefficient (φ)")
    ax.set_ylabel("Stage Loading (ψ)")
    ax.set_title(f"Engine {label} | No. of Stages: {no_of_stages}")

    # Mark the original data points
    for psi in psis:

        for phi in phis:

            ax.plot(phi, psi, "w+", markersize = 6, markeredgewidth = 1.2)

def export_engine():

    engine = create_engine(
        no_of_stages = Inputs.no_of_stages, thrust = Inputs.thrust, phi = Inputs.phi,
        psi = Inputs.psi, M_1 = Inputs.M_1
    )

    # add geometry
    engine.geometry = {
        "aspect_ratio": utils.Defaults.aspect_ratio,
        "diffusion_factor": utils.Defaults.diffusion_factor,
        "design_parameter": utils.Defaults.design_parameter
    }

    # calculate blade angles
    engine.empirical_design()

    Inputs.engines.append(engine)

    for blade_row in engine.blade_rows:

        print(f"blade_row.no_of_blades: {blade_row.no_of_blades}")

        if blade_row.no_of_blades > Inputs.max_no_of_blades:

            print(f"Error! Too many blades.")
            return
        
        if np.any(np.abs(blade_row.exit.deviation) > Inputs.max_deviation):

            print(f"Error! Deviation error.")
            #print(blade_row)
            return
        
        if np.any(blade_row.exit.chord > Inputs.max_chord):

            print(f"Error! Chord error.")
            #print(blade_row)
            return
        
        if np.any(blade_row.exit.diffusion_factor > Inputs.max_diffusion_factor):

            print(f"Error! Diffusion factor error.")
            return

    engine.export(f"smith_N_{Inputs.no_of_stages}_phi_{Inputs.phi}_psi_{Inputs.psi}")

if __name__ == "__main__":

    main()

    # show all plots
    plt.show()
