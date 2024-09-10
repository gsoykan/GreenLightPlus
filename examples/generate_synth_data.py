import os
from datetime import datetime
from typing import Dict

from core.green_light_model import GreenLightModel
from result_analysis.plot_green_light import plot_green_light
from service_functions.funcs import extract_last_value_from_nested_dict, calculate_energy_consumption
import random

# Set simulation parameters
season_length = 10  # Length of the growth cycle (days), can be fractional
season_interval = 10  # Time interval for each model run (days), can be fractional, e.g., 1/24/4 means 15 minutes
first_day = 91  # The first day of the growth cycle (day of the year)

PLOT_EACH_RUN = True
NUM_RUNS = 10 # some failed, some with very fews steps, with 0 yield, some 700MB (original is 200.9 MB)
SEED = 42

# TODO: raise for 0 yield and abnormal num of simulation steps...

def generate_random_plausible_init_state(return_original: bool = False) -> Dict[str, Dict[str, float]]:
    """
    Generates a random plausible initialization state for the greenhouse model.

    Returns:
        dict: init_state, A dictionary containing random values for greenhouse parameters.
    """

    if return_original:
        return {
            "p": {
                # Greenhouse structure settings
                'psi': 22,  # Average slope of greenhouse cover (degrees)
                'aFlr': 4e4,  # Floor area (m^2)
                'aCov': 4.84e4,  # Cover area, including side walls (m^2)
                'hAir': 6.3,  # Height of main area (m) (ridge height is 6.5m, screen is 20cm below)
                'hGh': 6.905,  # Average greenhouse height (m)
                'aRoof': 0.1169 * 4e4,  # Maximum roof ventilation area (m^2)
                'hVent': 1.3,  # Vertical dimension of a single ventilation opening (m)
                'cDgh': 0.75,  # Discharge coefficient for ventilation (dimensionless)
                'lPipe': 1.25,  # Length of pipe-rail heating system (m/m^2)
                'phiExtCo2': 7.2e4 * 4e4 / 1.4e4,  # CO2 injection capacity for the entire greenhouse (mg/s)
                'pBoil': 300 * 4e4,  # Boiler capacity for the entire greenhouse (W)

                # Control settings
                'co2SpDay': 1000,  # CO2 setpoint during light period (ppm)
                'tSpNight': 18.5,  # Temperature setpoint during dark period (°C)
                'tSpDay': 19.5,  # Temperature setpoint during light period (°C)
                'rhMax': 87,  # Maximum relative humidity (%)
                'ventHeatPband': 4,  # P-band for ventilation at high temperature (°C)
                'ventRhPband': 50,  # P-band for ventilation at high relative humidity (% humidity)
                'thScrRhPband': 10,  # P-band for screen opening at high relative humidity (% humidity)
                'lampsOn': 0,  # Time to turn on lights (h)
                'lampsOff': 18,  # Time to turn off lights (h)
                'lampsOffSun': 400,  # Global radiation above which lamps are turned off (W/m^2)
                'lampRadSumLimit': 10  # Predicted daily sum of solar radiation below which lamps are used (MJ/m^2/day)
            }
        }
    else:
        aflr = random.uniform(2e4, 5e4)

        # TODO: @gsoykan - not sure about randomization of phiExtCo2 because it is somehow dependant on aFlr for sure
        #  at least in this codebase BUT "greenlight matlab" sets in the following way:
        #  addParam(gl, 'phiExtCo2', 7.2e4);

        init_state = {
            "p": {
                # Greenhouse structure settings
                'psi': random.uniform(15, 30),  # Average slope of greenhouse cover (degrees)
                'aFlr': aflr,  # Floor area (m^2)
                'aCov': random.uniform(2.5e4, 6e4),  # Cover area, including side walls (m^2)
                'hAir': random.uniform(5, 7),  # Height of main area (m)
                'hGh': random.uniform(5.5, 7.5),  # Average greenhouse height (m)
                'aRoof': random.uniform(0.05, 0.15) * aflr,  # Roof ventilation area (m^2)
                'hVent': random.uniform(0.8, 1.8),  # Vertical dimension of ventilation opening (m)
                'cDgh': random.uniform(0.6, 0.9),  # Discharge coefficient for ventilation (dimensionless)
                'lPipe': random.uniform(1, 2),  # Pipe-rail heating length (m/m^2)
                'phiExtCo2': random.uniform(6e4, 8e4),  # CO2 injection capacity (mg/s)
                'pBoil': random.uniform(200, 400) * aflr,  # Boiler capacity (W)

                # Control settings
                'co2SpDay': random.uniform(800, 1200),  # CO2 setpoint during light period (ppm)
                'tSpNight': random.uniform(16, 20),  # Temperature setpoint during dark period (°C)
                'tSpDay': random.uniform(18, 24),  # Temperature setpoint during light period (°C)
                'rhMax': random.uniform(70, 90),  # Maximum relative humidity (%)
                'ventHeatPband': random.uniform(2, 6),  # P-band for ventilation at high temperature (°C)
                'ventRhPband': random.uniform(30, 70),  # P-band for ventilation at high relative humidity (%)
                'thScrRhPband': random.uniform(5, 15),  # P-band for screen opening at high relative humidity (%)
                'lampsOn': random.uniform(0, 10),  # Time to turn on lights (h)
                'lampsOff': random.uniform(12, 20),  # Time to turn off lights (h)
                'lampsOffSun': random.uniform(300, 600),  # Global radiation above which lamps are turned off (W/m²)
                'lampRadSumLimit': random.uniform(5, 20)  # Solar radiation sum below which lamps are used (MJ/m²/day)
            }
        }

        return init_state


if __name__ == '__main__':
    random.seed(SEED)

    for i in range(NUM_RUNS):
        print(f"\n***** RUNNING SIMULATION, NUM: {str(i)} *****\n")

        # Create a GreenLight model instance
        # Parameter Descriptions:
        # - first_day: Start date of the simulation (day of the year)
        # - isMature: Indicates whether the crop is mature
        # - epw_path: Path to the weather data file
        model = GreenLightModel(first_day=first_day,
                                isMature=True,
                                epw_path="../data/NLD_Amsterdam.062400_IWEC.epw")

        # Base path where the IO records should be stored
        io_record_path = "../data/io_records/synth"
        os.makedirs(io_record_path, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        io_record_path = os.path.join(io_record_path, timestamp)
        os.makedirs(io_record_path, exist_ok=True)

        # Initialize cumulative variables
        total_yield = 0  # Total yield (kg/m2)
        lampIn = 0  # Lighting energy consumption (MJ/m2)
        boilIn = 0  # Heating energy consumption (MJ/m2)

        # generate the original values for the first run, the rest is perturbated dataset
        if i == 0:
            init_state = generate_random_plausible_init_state(return_original=True)
        else:
            init_state = generate_random_plausible_init_state()

        # Run the model based on growth cycle and time interval
        for current_step in range(int(season_length // season_interval)):
            # run the model and get results
            gl = model.run_model(gl_params=init_state,
                                 season_length=season_length,
                                 season_interval=season_interval,
                                 step=current_step)

            current_io_record_path = os.path.join(io_record_path, f"io_record_step_{str(current_step)}.csv")
            model.io_recorder.save_dataset(current_io_record_path)

            init_state = gl
            dmc = 0.06  # dry matter content

            # calculate the print and current yield (kg/m2)
            current_yield = 1e-6 * calculate_energy_consumption(gl, 'mcFruitHar') / dmc
            print(f"Current yield: {current_yield:.2f} kg/m2")

            # Accumulate fruit yield (kg/m2)
            total_yield += current_yield

            # Calculate and accumulate energy consumption from lighting and heating (MJ/m2)
            lampIn += 1e-6 * calculate_energy_consumption(gl, "qLampIn", "qIntLampIn")
            boilIn += 1e-6 * calculate_energy_consumption(gl, "hBoilPipe", "hBoilGroPipe")

        # Print final results
        print(f"Total yield: {total_yield:.2f} kg/m2")
        print(f"Lighting energy consumption: {lampIn:.2f} MJ/m2")
        print(f"Heating energy consumption: {boilIn:.2f} MJ/m2")
        print(f"Energy consumption per unit: {(lampIn + boilIn) / total_yield:.2f} MJ/kg")

        if PLOT_EACH_RUN:
            # Plot model results
            plot_green_light(gl)
