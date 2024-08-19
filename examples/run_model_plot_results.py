import os
from datetime import datetime

from core.green_light_model import GreenLightModel
from result_analysis.plot_green_light import plot_green_light
from service_functions.funcs import extract_last_value_from_nested_dict, calculate_energy_consumption

# Set simulation parameters
season_length = 10  # Length of the growth cycle (days), can be fractional
season_interval = 10  # Time interval for each model run (days), can be fractional, e.g., 1/24/4 means 15 minutes
first_day = 91  # The first day of the growth cycle (day of the year)

if __name__ == '__main__':
    # Create a GreenLight model instance
    # Parameter Descriptions:
    # - first_day: Start date of the simulation (day of the year)
    # - isMature: Indicates whether the crop is mature
    # - epw_path: Path to the weather data file
    model = GreenLightModel(first_day=first_day,
                            isMature=True,
                            epw_path="../data/NLD_Amsterdam.062400_IWEC.epw")

    # Base path where the IO records should be stored
    io_record_path = "../data/io_records"
    os.makedirs(io_record_path, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    io_record_path = os.path.join(io_record_path, timestamp)
    os.makedirs(io_record_path, exist_ok=True)

    # Initialize cumulative variables
    total_yield = 0  # Total yield (kg/m2)
    lampIn = 0  # Lighting energy consumption (MJ/m2)
    boilIn = 0  # Heating energy consumption (MJ/m2)

    # Initialize model state and parameters
    init_state = {
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

# Plot model results
plot_green_light(gl)

# results with season interval 2
# Start with a mature crop
# Current yield: 0.57 kg/m2
# Current yield: 0.61 kg/m2
# Current yield: 0.66 kg/m2
# Current yield: 0.69 kg/m2
# Current yield: 0.69 kg/m2
# Total yield: 3.22 kg/m2
# Lighting energy consumption: 33.94 MJ/m2
# Heating energy consumption: 33.46 MJ/m2
# Energy consumption per unit: 20.92 MJ/kg

# results with season interval 10
# Start with a mature crop
# Current yield: 3.22 kg/m2
# Total yield: 3.22 kg/m2
# Lighting energy consumption: 34.02 MJ/m2
# Heating energy consumption: 33.68 MJ/m2
# Energy consumption per unit: 21.00 MJ/kg
