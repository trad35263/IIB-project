# import modules
import numpy as np
import pytest
import sys
from pathlib import Path
from time import perf_counter as timer

# adjust sys.path to import your modules
sys.path.insert(0, str(Path(__file__).parent))

# import high speed solver modules
from engine import Engine
from flight_scenario import Flight_scenario
import utils

# Assert class
class Assert:
    """Stores several functions relating to engine evaluation."""
    def no_nans(engine):
        """Pytest helper function for identifying NaNs in an engine."""
        # loop over each blade row
        for blade_row in engine.blade_rows:

            # loop for inlet and exit
            for annulus in [blade_row.inlet, blade_row.exit]:

                # assert no nans in any primary variables
                assert np.any(np.isnan(annulus.M)) == False
                assert np.any(np.isnan(annulus.alpha)) == False
                assert np.any(np.isnan(annulus.T_0)) == False
                assert np.any(np.isnan(annulus.p_0)) == False

    def constant_mass_flow_rate(engine, epsilon = 1e-1):
        """Pytest helper function for identifying non-conserved mass flow in the engine."""
        # store dimensionless mass flow rate for convenience
        m_dot = utils.mass_flow_function(engine.M_1)

        # loop over each blade row
        for blade_row in engine.blade_rows:

            # loop for inlet and exit
            for annulus in [blade_row.inlet, blade_row.exit]:

                # check for constant mass flow rate within some tolerance
                assert np.abs(1 - annulus.m_dot[-1] / m_dot) < epsilon

    def physical_values(engine):
        """Pytest helper function for checking for out-of-bounds engine metrics."""
        # test criteria
        assert engine.T_0_ratio > 1
        assert engine.p_0_ratio > 1
        assert engine.nozzle_area_ratio > 0
        assert engine.C_th > 0
        assert engine.jet_velocity_ratio > 0
        assert engine.jet_velocity_ratio <= 1
        assert engine.eta_comp > 0
        assert engine.eta_comp <= 1
        assert engine.eta_prop > 0
        assert engine.eta_prop <= 1
        assert engine.eta_overall > 0
        assert engine.eta_overall <= 1

# TestEngine class
class TestEngine:
    """Test suite for engine creation and basic configuration."""
    def create_engine(
            self,
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

    @pytest.mark.parametrize("altitude", [0, 10000])
    @pytest.mark.parametrize("flight_speed", [20, 100])
    @pytest.mark.parametrize("diameter", [0.2])
    @pytest.mark.parametrize("hub_tip_ratio", [0.3])
    @pytest.mark.parametrize("thrust", [10, 50])
    @pytest.mark.parametrize("no_of_stages", [2])
    @pytest.mark.parametrize("phi", [0.75])
    @pytest.mark.parametrize("psi", [0.1, 0.2])
    @pytest.mark.parametrize("vortex_exponent", [0.5])
    @pytest.mark.parametrize("Y_p", [0.05])
    @pytest.mark.parametrize("area_ratio", [1])
    def test_engine_matrix(
        self, altitude, flight_speed, diameter, hub_tip_ratio, thrust,
        no_of_stages, phi, psi, vortex_exponent, Y_p, area_ratio
    ):
        engine = self.create_engine(
            altitude = altitude,
            flight_speed = flight_speed,
            diameter = diameter,
            hub_tip_ratio = hub_tip_ratio,
            thrust = thrust,
            no_of_stages = no_of_stages,
            phi = phi,
            psi = psi,
            vortex_exponent = vortex_exponent,
            Y_p = Y_p,
            area_ratio = area_ratio
        )
        Assert.no_nans(engine)
        Assert.constant_mass_flow_rate(engine)
        Assert.physical_values(engine)
        #assert_engine_fundamentals(engine)
