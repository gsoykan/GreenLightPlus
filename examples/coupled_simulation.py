import sys

from core.greenlight_energyplus_simulation import GreenhouseSimulation
# docs: https://energyplus.readthedocs.io/en/latest/api.html
from pyenergyplus.api import EnergyPlusAPI
from service_functions.convert_epw2csv import convert_epw2csv

"""
to use pyenergyplus API you can do one of the followings:
1) Modify the Python Path in PyCharm:
Add the EnergyPlus installation directory to the Python path in PyCharm.
Go to File > Settings > Project: [Your Project Name] > Project Structure.
Click Add Content Root and navigate to the /Applications/EnergyPlus-24-1-0 directory.
Click OK to add it to your project structure.

2) 
energyplus_path = "/Applications/EnergyPlus-24-1-0"
sys.path.insert(0, energyplus_path)
"""

if __name__ == '__main__':
    # Create an EnergyPlus API Object
    api = EnergyPlusAPI()

    # set the paths for the weather file and model file
    epw_path = "../data/NLD_Amsterdam.062400_IWEC.epw"
    idf_path = "../data/model_files/greenhouse_half_circle.idf"

    # convert the EPW weather file to a CSV file for easier processing
    csv_path = convert_epw2csv(epw_path=epw_path, time_step=1)

    # set the output directory
    output_directory = "../data/energyPlus/outputs"

    # set the start date and number of days for the simulation
    first_day = 91
    season_length = 7

    # create an instance of the GreenhouseSimulation class
    simulation = GreenhouseSimulation(api,
                                      epw_path,
                                      idf_path,
                                      csv_path,
                                      output_directory,
                                      first_day,
                                      season_length,
                                      isMature=True)

    # run the simulation
    simulation.run()

    # get the simulation results
    total_yield, lampIn, boilIn = simulation.get_results()
    print(f"Total yield: {total_yield} kg/m2, "
          f"Lamp input: {lampIn} MJ/m2, "
          f"Boiler input: {boilIn} MJ/m2\n")
