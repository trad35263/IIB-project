# import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from time import perf_counter as timer

# import high speed solver
import utils
from engine import Engine
from flight_scenario import Flight_scenario

# Inputs class
class Inputs:

    # motor parameters
    #power = 3000
    #rpm = 10000

    # thrust and nozzle area ratio
    thrust = 30
    
    # variables to loop over
    stages = [1, 2, 3]
    phis = [0.6, 0.75, 0.9]
    psis = [0.1, 0.15, 0.2]

    # inlet Mach number
    M_1 = 0.15

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
        area_ratio = utils.Defaults.area_ratio
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
        flight_scenario, no_of_stages, phi, psi, vortex_exponent, Y_p, area_ratio
    )

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

def export_engine():

    def specify_motor(vars, no_of_stages, phi, psi):

        # thrust is input variable
        thrust = vars[0]

        # create engine
        Inputs.engine = create_engine(no_of_stages = no_of_stages, thrust = thrust, phi = phi, psi = psi)

        # residual is mass flow rate delta
        residuals = [Inputs.engine.M_1 - Inputs.M_1]
        return residuals

    # set initial guess
    x0 = [Inputs.thrust]

    # solve for appropriate thrust and stage loading coefficients iteratively
    sol = least_squares(
        specify_motor,
        x0, args = (Inputs.no_of_stages, Inputs.phi, Inputs.psi),
        xtol=1e-6,      # tolerance on change in x
        ftol=1e-6,      # tolerance on change in residual
        gtol=1e-6       # tolerance on gradient
    )

    # print user feedback
    print(f"sol: {sol}")
    print(Inputs.engine)

    # add geometry
    Inputs.engine.geometry = {
        "aspect_ratio": utils.Defaults.aspect_ratio,
        "diffusion_factor": utils.Defaults.diffusion_factor,
        "design_parameter": utils.Defaults.design_parameter
    }

    # calculate blade angles and export engine
    Inputs.engine.empirical_design()
    Inputs.engine.plot_section()
    Inputs.engine.export(f"N_{Inputs.no_of_stages}_phi_{Inputs.phi}_psi_{Inputs.psi}")

if __name__ == "__main__":

    main()

    # show all plots
    plt.show()