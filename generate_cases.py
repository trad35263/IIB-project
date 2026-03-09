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
    power = 3000
    rpm = 10000
    
    # variables to loop over
    stages = [1, 2, 3]
    #phis = [0.6, 0.75, 0.9]
    phis = [0.75]

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

            Inputs.flow_coefficient = phi

            # get engine
            export_engine()

def export_engine():

    def specify_motor(vars, no_of_stages, flow_coefficient):

        # convert input variables to dimensional values
        thrust = vars[0] * utils.Defaults.thrust
        psi = list(vars[1::2] * utils.Defaults.psi)
        phi = list(flow_coefficient * np.concatenate(([1], vars[2::2])))

        # create engine
        Inputs.engine = create_engine(no_of_stages = no_of_stages, thrust = thrust, phi = phi, psi = psi)

        # create empty list of residuals
        residuals = []

        # loop for each stage
        for index, stage in enumerate(Inputs.engine.stages):

            # append dimensionless error in motor power and rpm to residuals list
            residuals.append(getattr(Inputs.engine, f"rotor_{index + 1}_power") / Inputs.power - 1)
            residuals.append(getattr(Inputs.engine, f"rotor_{index + 1}_rpm") / Inputs.rpm - 1)

        return residuals

    # set initial guess
    x0 = np.ones(2 * Inputs.no_of_stages)

    # solve for appropriate thrust and stage loading coefficients iteratively
    sol = least_squares(
        specify_motor,
        x0, args = (Inputs.no_of_stages, Inputs.flow_coefficient),
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
    Inputs.engine.export(f"N_{Inputs.no_of_stages}_phi_{Inputs.flow_coefficient}")

if __name__ == "__main__":

    main()