Notes on
Mini-greenhouse by Efraim:
features = ['time', 'global out', 'temp out', 'rh out', 'co2 out', 'ventilation', 'toplights', 'heater']
target_variables = ['global in', 'temp in', 'rh in', 'co2 in']

### Mini-Greenhouse to GreenLight Mapping

| **Mini-Greenhouse** | **Description**                                                 | **GreenLight Corresponding Input/Output**             |
|---------------------|-----------------------------------------------------------------|--------------------------------------------------------|
| `time`              | Timestamp of the data                                             | `x_time`                                               |
| `global out`        | Outdoor global irradiation [W m^{-2}]                            | `d.iGlob`                                              |
| `temp out`          | Outdoor air temperature [°C]                                     | `d.tOut`                                               |
| `rh out`            | Outdoor relative humidity [%]                                    | `d.vpOut` (not exactly this, comes from Weather data)                                           |
| `co2 out`           | Outdoor CO2 concentration [ppm]                                  | `d.co2Out` (x.co2Air?)                                            |
| `ventilation`       | Ventilation status or rate [unit unspecified]                    | (No direct equivalent; could be related to `d.wind` or other ventilation parameters in GreenLight) |
| `toplights`         | Status or intensity of top lights [unit unspecified]             | (No direct equivalent; could be related to `d.tSky` or other lighting parameters) |
| `heater`            | Heater status or intensity [unit unspecified]                     | (No direct equivalent; related to heating and temperature control parameters) |

**Target Variables:**

| **Mini-Greenhouse** | **Description**                                                 | **GreenLight Corresponding Input/Output**             |
|---------------------|-----------------------------------------------------------------|--------------------------------------------------------|
| `global in`         | Indoor global irradiation [W m^{-2}]                             | (No direct equivalent; could be related to internal light modeling) |
| `temp in`           | Indoor temperature [°C]                                          | (No direct equivalent; might correspond to specific internal temperature modeling) |
| `rh in`             | Indoor relative humidity [%]                                     | (No direct equivalent; could be related to internal humidity modeling) |
| `co2 in`            | Indoor CO2 concentration [ppm]                                   | (No direct equivalent; could be related to internal CO2 concentration modeling) |


- The equivalent values for most of above (except global-in) (some calculated, some derived from auxiliary states) can be seen in the `plot_green_light` function.
- [ ] Check if indoor global radiation can be calculated with what we have? 
- [ ] Design a mini-greenhouse NN model using those values.


### Seljaar Weather Data
Columns:

A - time since beginning of year [s]
B - outdoor global radiation [W m^{-2}]
C - wind [m/s]
D - air temperature [�C]
E - sky temperature [�C]
F - ?????
G - CO2 concentration [ppm] - assumed to be constant 320, not included in the original Breuer&Braak paper
H - day number
I - relative humidity [%]

### Notes

1. **Ventilation, Toplights, and Heater**: These features in Mini-greenhouse may have indirect effects on parameters in 
GreenLight, such as air temperature, CO2 concentration, or other climate controls. 
GreenLight might model these through complex interactions or external conditions that are not directly mapped.

2. **Global In, Temp In, RH In, CO2 In**: These target variables from Mini-greenhouse are 
specific to the internal climate of the greenhouse. 
GreenLight's model might predict these based on input features, 
but exact mappings could be more complex and involve multiple parameters.

3. **Direct Mapping**: Some features and target variables in Mini-greenhouse do not have direct one-to-one mappings 
in GreenLight due to differences in model focus and parameterization. 
For instance, GreenLight may focus more on internal climate modeling and complex interactions 
which are not directly captured in Mini-greenhouse’s simpler feature set.

### Variables in State Space (from GreenLight Plus)
| Parameter Number | Parameter Name      | Description                                                                 |
|------------------|---------------------|-----------------------------------------------------------------------------|
| 1                | Day of the year (day_of_year)  | Reflects seasonal changes, affecting light intensity and temperature differences |
| 2                | Night temperature setpoint (tSpNight)  | Target temperature for the greenhouse at night                                |
| 3                | Day temperature setpoint (tSpDay)  | Target temperature for the greenhouse during the day                          |
| 4                | Day CO2 setpoint (co2SpDay)     | Target CO2 concentration during the day                                      |
| 5                | Air CO2 concentration (co2Air)  | Current CO2 level inside the greenhouse                                      |
| 6                | Air vapor pressure (vpAir)      | Humidity conditions inside the greenhouse                                    |
| 7                | Air temperature (tAir)          | Actual temperature inside the greenhouse                                     |
| 8                | Fruit dry weight (cFruit)       | Growth status of the crop fruits                                             |
| 9                | Total maintenance respiration rate (mcOrgAir) | Crop's respiration metabolic rate                                             |
| 10               | Net photosynthetic rate (mcAirBuf) | Crop's photosynthetic efficiency                                             |
| 11               | Global radiation (iGlob)        | Intensity of solar radiation received in the greenhouse                      |
| 12               | Outdoor temperature (tOut)      | Environmental temperature outside the greenhouse                             |
| 13               | Lamp energy consumption (lampIn) | Energy consumption of artificial lighting                                    |
| 14               | Boiler energy consumption (boilIn) | Energy consumption of the heating system                                    |

# GreenLight "Indoor" Inputs

The function `set_gl_states_init(gl, weather_datenum, indoor=None)` allows the initialization of indoor conditions for the GreenLight model.

### `indoor` (Optional[np.ndarray])
An optional 3-column matrix where:

- `indoor[:, 0]`: **Timestamps** of the input [s], in regular intervals of 300, starting with 0.
- `indoor[:, 1]`: **Temperature** [°C], representing the indoor air temperature.
- `indoor[:, 2]`: **Vapor Pressure** [Pa], representing the indoor vapor concentration.
- `indoor[:, 3]`: **CO2 Concentration** [mg m^{-3}], representing the indoor CO2 concentration.

### Example Usage:
```python
if indoor is not None and len(indoor) > 0:
    # Use provided indoor data to set initial conditions
    gl["x"]["co2Air"] = indoor[0, 3]  # Set initial indoor CO2 concentration
    gl["x"]["tAir"] = indoor[0, 1]    # Set initial indoor air temperature
    gl["x"]["vpAir"] = indoor[0, 2]   # Set initial indoor vapor pressure
else:
    # Set default initial conditions when indoor data is not provided
    gl["x"]["tAir"] = gl["p"]["tSpNight"]  # Default indoor air temperature (night setpoint)
    gl["x"]["vpAir"] = gl["p"]["rhMax"] / 100 * satVp(gl["x"]["tAir"])  # Calculate vapor pressure using max relative humidity
    gl["x"]["co2Air"] = gl["d"]["co2Out"][0, 1]  # Set CO2 concentration to outdoor level
```

- Interestingly, the docstring for the function indicates that:

> If `indoor` is not provided, the initial values are set to default values of 21°C for indoor temperature, 1000 Pa for vapor pressure, and 400 mg/m³ for CO2 concentration.

However, **this is not the case** in the actual implementation. Instead:

- The indoor temperature is set to the **night setpoint** (`tSpNight`).
- The vapor pressure is calculated using the **maximum relative humidity** (`rhMax`).
- The CO2 concentration is set to the **outdoor CO2 level**.


# GreenLight Input Descriptions

### X: Inputs in GreenLight
| Parameter   | Description                                          | Unit          | Min Value | Max Value |
|-------------|------------------------------------------------------|---------------|-----------|-----------|
| `x_co2Air`  | CO2 concentration in the air                         | ppm           | 250       | 2000      |
| `x_co2Top`  | CO2 concentration at the top                         | ppm           | 250       | 3000      |
| `x_tAir`    | Air temperature                                      | °C            | -40       | 50        |
| `x_tTop`    | Temperature at the top                               | °C            | -40       | 50        |
| `x_tCan`    | Canopy temperature                                   | °C            | -10       | 50        |
| `x_tCovIn`  | Temperature of the internal cover                    | °C            | -10       | 50        |
| `x_tThScr`  | Temperature of the thermal screen                    | °C            | -10       | 50        |
| `x_tFlr`    | Floor temperature                                    | °C            | -10       | 50        |
| `x_tPipe`   | Temperature of the pipe                              | °C            | -10       | 100       |
| `x_tCovE`   | Temperature of the external cover                    | °C            | -40       | 50        |
| `x_tSo1`    | Temperature of soil layer 1                          | °C            | -30       | 40        |
| `x_tSo2`    | Temperature of soil layer 2                          | °C            | -30        | 40        |
| `x_tSo3`    | Temperature of soil layer 3                          | °C            | -30        | 40        |
| `x_tSo4`    | Temperature of soil layer 4                          | °C            | -30        | 40        |
| `x_tSo5`    | Temperature of soil layer 5                          | °C            | -30        | 40        |
| `x_vpAir`   | Vapor pressure in the air                            | Pa            | 0         | 7300      |
| `x_vpTop`   | Vapor pressure at the top                            | Pa            | 0         | 7300      |
| `x_tCan24`  | Average canopy temperature over 24 hours             | °C            | -10       | 50        |
| `x_time`    | Time                                                 | s             | N/A         | N/A       |
| `x_tLamp`   | Temperature of the lamp                              | °C            | -10       | 100       |
| `x_tGroPipe`| Temperature of the growth pipe                       | °C            | -10       | 100       |
| `x_tIntLamp`| Temperature of the internal lamp                     | °C            | -10       | 100       |
| `x_tBlScr`  | Temperature of the blind screen                      | °C            | -10       | 50        |
| `x_cBuf`    | Carbohydrates in the buffer                          | mg CH₂O m⁻²   | 0         | 20000     |
| `x_cLeaf`   | Carbohydrates in the leaves                          | mg CH₂O m⁻²   | 0         | 100000     |
| `x_cStem`   | Carbohydrates in the stem                            | mg CH₂O m⁻²   | 0         | 300000     |
| `x_cFruit`  | Carbohydrates in the fruit                           | mg CH₂O m⁻²   | 0         | 300000    |
| `x_tCanSum` | Accumulated canopy temperature                       | °C days       | 0         | 3500      |

- **For Soil Temperature**: https://www.weather.gov/ncrfc/LMI_SoilTemperatureDepthMaps (-30 & 40)
- **x_cBuf**: 19632,164791189300 is the max in the current dataset, but may depend on the plant
- **x_cLeaf**: 98162,84787508580 max value in the dataset
- **x_cStem**: 275872,100691447 max value in the dataset
- **x_cFruit**: 287017,21149939 max value in the dataset
- **x_tCanSum**: 3204,87624137959 max value in the dataset
- **x_tIntLamp**: maybe broken? because we have impossible values in the simulation output => 2.458691596984865e+99
- [ ] This certainly needs debugging... 
- SIMILAR ISSUE WITH THE **x_tBlScr** and **2.458691596984863e+307**

Some issues
- i predicted that x_co2Air would be 1000 at max, but in the dataset we have 1847.5690588668872 so i change it to 2000
- similar issue with x_co2Top the max is 2697.289469324912 so i change it to 3000

### Sensible Rationale for Min and Max Values:
- **Carbohydrates**: The buffer values are estimated from plant biology research on carbon storage and transfer between plant organs.
- **Accumulated Canopy Temperature**: Accumulated temperature sums help in understanding growing degree days (GDD), influencing plant development.


### D: Uncontrolled Inputs (Disturbances)

- **Global Irradiance (iGlob)**: Check max values and dive deeper into the [wiki link](https://en.wikipedia.org/wiki/Solar_irradiance).
  
- **Temperature**:
  - Reasonable Range for Normalization: -40°C to 50°C
  - You can apply Min-Max scaling: \((t - \text{min}) / (\text{max} - \text{min})\)

- **Outdoor Vapor Pressure (d_vpOut)**:
  - At 40°C (104°F), the maximum possible vapor pressure is around 73 mb or 7.3 kPa.
  - Check the formula and ensure it’s correct for your use case.

- **Outdoor CO2 Concentration (d_co2Out)**:
  - [Source 1](https://www.co2meter.com/blogs/news/carbon-dioxide-indoor-levels-chart?srsltid=AfmBOopwsMiefU1B2OYGQBcQ0Mo1pbiQPSbmc8gYvyT2GeGoSBKXZkYS) and [Source 2](https://www.kane.co.uk/knowledge-centre/what-are-safe-levels-of-co-and-co2-in-rooms)
  - Range: 250-500 ppm, but values around 800 ppm have been recorded, so consider using 250-1000 ppm for the range.

- **Outdoor Wind Speed (d_wind)**:
  - Maximum wind speed can be roughly 100 mph (45 m/s).

- **Outdoor Temperature (d_tOut)**:
  - Reasonable Range: -50°C to 30°C for normalization.

- **Outdoor Soil Temperature (d_tSoOut)**:
  - Based on [this source](https://eos.com/blog/soil-temperature/#:~:text=The%20average%20soil%20temperatures%20for,of%20soluble%20substances%2C%20and%20metabolism.):
    - Optimal range is 10°C to 24°C.
    - For normalization, use a broader range: -5°C to 35°C.

- **Daily Radiation Sum (d_dayRadSum)**:
  - 0 MJ m⁻² day⁻¹ corresponds to no solar radiation (e.g., polar night, extreme cloud cover).
  - 40 MJ m⁻² day⁻¹ corresponds to intense solar radiation (e.g., desert or tropical regions).
  - Further research is needed, but [this paper, An Estimation of Solar Radiation using Robust Linear Regression Method](https://pdf.sciencedirectassets.com/277910/1-s2.0-S1876610212X00092/1-s2.0-S1876610212009277/main.pdf) can provide more insights.

| Parameter | Description                                          | Unit          | Min Value | Max Value |
|-----------|------------------------------------------------------|---------------|-----------|-----------|
| `d_iGlob` | Global irradiance                                    | W m^{-2}      | 0         | 1200      |
| `d_tOut`  | Outdoor temperature                                  | °C            | -40       | 50        |
| `d_vpOut` | Outdoor vapor pressure                              | Pa            | 0         | 7300      |
| `d_co2Out`| Outdoor CO2 concentration                           | ppm           | 250       | 1000      |
| `d_wind`  | Outdoor wind speed                                  | m s^{-1}      | 0         | 45        |
| `d_tSky`  | Sky temperature                                     | °C            | -50       | 30        |
| `d_tSoOut`| Outdoor soil temperature                            | °C            | -5        | 35        |
| `d_dayRadSum` | Daily radiation sum                              | MJ m^{-2} day^{-1} | 0         | 40        |
| `d_isDay` | Indicator for whether it is daytime (binary)        |               | 0         | 1         |
| `d_isDaySmooth` | Smoothed indicator for whether it is daytime (binary) |           | 0         | 1         |


### U: (Controlled Inputs)

| Parameter | Description | Unit | Min Value | Max Value |
| --- | --- | --- | --- | --- |
| `u_boil` | Boiler control value | - | 0 (closed) | 1 (full capacity) |
| `u_boilGro` | Grow pipe boiler control value | - | 0 (closed) | 1 (full capacity) |
| `u_extCo2` | External CO2 control value | - | 0 (closed) | 1 (full capacity) |
| `u_shScr` | Shadow screen control value | - | 0 (open/folded) | 1 (closed/spread out) |
| `u_shScrPer` | Perforated shadow screen control value | - | 0 (open/folded) | 1 (closed/spread out) |
| `u_thScr` | Thermal screen control value | - | 0 (open/folded) | 1 (closed/spread out) |
| `u_roof` | Roof vent control value | - | 0 (closed) | 1 (open/maximal ventilation) |
| `u_side` | Side vent control value | - | 0 (closed) | 1 (open/maximal ventilation) |
| `u_lamp` | Lamp control value | - | 0 (off) | 1 (on) |
| `u_intLamp` | Interlight control value | - | 0 (off) | 1 (on) |
| `u_blScr` | Blackout screen control value | - | 0 (open/folded) | 1 (closed/spread out) |


### A: Auxiliary States

| Parameter Name                          | Description                                                            | Units              | Min Value | Max Value |
|----------------------------------------|------------------------------------------------------------------------|--------------------|-----------|-----------|
| `a_tauShScrPar`                        | Time constant for shadow screen parameter                              | s                  |           |           |
| `a_tauShScrPerPar`                     | Time constant for shadow screen periodic parameter                     | s                  |           |           |
| `a_rhoShScrPar`                        | Reflection coefficient for shadow screen parameter                     |                    |           |           |
| `a_rhoShScrPerPar`                     | Reflection coefficient for shadow screen periodic parameter             |                    |           |           |
| `a_tauShScrShScrPerPar`                | Time constant for shadow screen and shadow screen periodic parameter    | s                  |           |           |
| `a_rhoShScrShScrPerParUp`              | Reflection coefficient for shadow screen and shadow screen periodic, upper limit |                    |           |           |
| `a_rhoShScrShScrPerParDn`              | Reflection coefficient for shadow screen and shadow screen periodic, lower limit |                    |           |           |
| `a_tauShScrNir`                        | Time constant for shadow screen NIR                                    | s                  |           |           |
| `a_tauShScrPerNir`                     | Time constant for shadow screen periodic NIR                           | s                  |           |           |
| `a_rhoShScrNir`                        | Reflection coefficient for shadow screen NIR                           |                    |           |           |
| `a_rhoShScrPerNir`                     | Reflection coefficient for shadow screen periodic NIR                  |                    |           |           |
| `a_tauShScrShScrPerNir`                | Time constant for shadow screen and shadow screen periodic NIR          | s                  |           |           |
| `a_rhoShScrShScrPerNirUp`              | Reflection coefficient for shadow screen and shadow screen periodic NIR, upper limit |                    |           |           |
| `a_rhoShScrShScrPerNirDn`              | Reflection coefficient for shadow screen and shadow screen periodic NIR, lower limit |                    |           |           |
| `a_tauShScrFir`                        | Time constant for shadow screen FIR                                    | s                  |           |           |
| `a_tauShScrPerFir`                     | Time constant for shadow screen periodic FIR                           | s                  |           |           |
| `a_rhoShScrFir`                        | Reflection coefficient for shadow screen FIR                           |                    |           |           |
| `a_rhoShScrPerFir`                     | Reflection coefficient for shadow screen periodic FIR                  |                    |           |           |
| `a_tauShScrShScrPerFir`                | Time constant for shadow screen and shadow screen periodic FIR          | s                  |           |           |
| `a_rhoShScrShScrPerFirUp`              | Reflection coefficient for shadow screen and shadow screen periodic FIR, upper limit |                    |           |           |
| `a_rhoShScrShScrPerFirDn`              | Reflection coefficient for shadow screen and shadow screen periodic FIR, lower limit |                    |           |           |
| `a_tauThScrPar`                        | Time constant for thermal screen parameter                             | s                  |           |           |
| `a_rhoThScrPar`                        | Reflection coefficient for thermal screen parameter                    |                    |           |           |
| `a_tauCovThScrPar`                     | Time constant for cover and thermal screen parameter                    | s                  |           |           |
| `a_rhoCovThScrParUp`                   | Reflection coefficient for cover and thermal screen, upper limit       |                    |           |           |
| `a_rhoCovThScrParDn`                   | Reflection coefficient for cover and thermal screen, lower limit       |                    |           |           |
| `a_tauThScrNir`                        | Time constant for thermal screen NIR                                   | s                  |           |           |
| `a_rhoThScrNir`                        | Reflection coefficient for thermal screen NIR                          |                    |           |           |
| `a_tauCovThScrNir`                     | Time constant for cover and thermal screen NIR                         | s                  |           |           |
| `a_rhoCovThScrNirUp`                   | Reflection coefficient for cover and thermal screen NIR, upper limit    |                    |           |           |
| `a_rhoCovThScrNirDn`                   | Reflection coefficient for cover and thermal screen NIR, lower limit    |                    |           |           |
| `a_tauCovParOld`                       | Time constant for old cover and PAR                                    | s                  |           |           |
| `a_rhoCovParOldUp`                     | Reflection coefficient for old cover and PAR, upper limit              |                    |           |           |
| `a_rhoCovParOldDn`                     | Reflection coefficient for old cover and PAR, lower limit              |                    |           |           |
| `a_tauCovNirOld`                       | Time constant for old cover and NIR                                    | s                  |           |           |
| `a_rhoCovNirOldUp`                     | Reflection coefficient for old cover and NIR, upper limit              |                    |           |           |
| `a_rhoCovNirOldDn`                     | Reflection coefficient for old cover and NIR, lower limit              |                    |           |           |
| `a_tauBlScrPar`                        | Time constant for blind screen parameter                               | s                  |           |           |
| `a_rhoBlScrPar`                        | Reflection coefficient for blind screen parameter                      |                    |           |           |
| `a_tauCovBlScrPar`                     | Time constant for cover and blind screen parameter                     | s                  |           |           |
| `a_rhoCovBlScrParUp`                   | Reflection coefficient for cover and blind screen, upper limit         |                    |           |           |
| `a_rhoCovBlScrParDn`                   | Reflection coefficient for cover and blind screen, lower limit         |                    |           |           |
| `a_tauBlScrNir`                        | Time constant for blind screen NIR                                    | s                  |           |           |
| `a_rhoBlScrNir`                        | Reflection coefficient for blind screen NIR                           |                    |           |           |
| `a_tauCovBlScrNir`                     | Time constant for cover and blind screen NIR                          | s                  |           |           |
| `a_rhoCovBlScrNirUp`                   | Reflection coefficient for cover and blind screen NIR, upper limit      |                    |           |           |
| `a_rhoCovBlScrNirDn`                   | Reflection coefficient for cover and blind screen NIR, lower limit      |                    |           |           |
| `a_tauCovPar`                          | Time constant for cover and PAR                                       | s                  |           |           |
| `a_rhoCovPar`                          | Reflection coefficient for cover and PAR                               |                    |           |           |
| `a_tauCovNir`                          | Time constant for cover and NIR                                       | s                  |           |           |
| `a_rhoCovNir`                          | Reflection coefficient for cover and NIR                               |                    |           |           |
| `a_tauCovFir`                          | Time constant for cover and FIR                                       | s                  |           |           |
| `a_rhoCovFir`                          | Reflection coefficient for cover and FIR                               |                    |           |           |
| `a_aCovPar`                            | Area coefficient for cover and PAR                                    |                    |           |           |
| `a_aCovNir`                            | Area coefficient for cover and NIR                                    |                    |           |           |
| `a_aCovFir`                            | Area coefficient for cover and FIR                                    |                    |           |           |
| `a_epsCovFir`                          | Emissivity of cover FIR                                                |                    |           |           |
| `a_capCov`                             | Capacity of cover                                                      |                    |           |           |
| `a_lai`                                | Leaf area index                                                        |                    |           |           |
| `a_capCan`                             | Capacity of canopy                                                     |                    |           |           |
| `a_capCovE`                            | Capacity of cover E                                                    |                    |           |           |
| `a_capCovIn`                           | Capacity of internal cover                                             |                    |           |           |
| `a_capVpAir`                           | Capacity of air vapor pressure                                         |                    |           |           |
| `a_capVpTop`                           | Capacity of top vapor pressure                                         |                    |           |           |
| `a_qLampIn`                            | Heat output from lamp input                                            |                    |           |           |
| `a_qIntLampIn`                         | Heat output from internal lamp input                                  |                    |           |           |
| `a_rParGhSun`                          | PAR radiation from the sun to the greenhouse                         |                    |           |           |
| `a_rParGhLamp`                         | PAR radiation from the lamp to the greenhouse                        |                    |           |           |
| `a_rParGhIntLamp`                      | PAR radiation from internal lamp to the greenhouse                    |                    |           |           |
| `a_rCanSun`                            | Radiation from the sun to the canopy                                  |                    |           |           |
| `a_rCanLamp`                           | Radiation from the lamp to the canopy                                 |                    |           |           |
| `a_rCanIntLamp`                        | Radiation from internal lamp to the canopy                             |                    |           |           |
| `a_rCan`                               | Radiation from the canopy                                             |                    |           |           |
| `a_rParSunCanDown`                     | PAR radiation from sun to canopy down                                 |                    |           |           |
| `a_rParLampCanDown`                    | PAR radiation from lamp to canopy down                                |                    |           |           |
| `a_fIntLampCanPar`                     | Fraction of internal lamp PAR absorbed by canopy                      |                    |           |           |
| `a_fIntLampCanNir`                     | Fraction of internal lamp NIR absorbed by canopy                      |                    |           |           |
| `a_rParIntLampCanDown`                 | PAR radiation from internal lamp to canopy down                       |                    |           |           |
| `a_rParSunFlrCanUp`                    | PAR radiation from sun to floor up                                    |                    |           |           |
| `a_rParLampFlrCanUp`                   | PAR radiation from lamp to floor up                                   |                    |           |           |
| `a_rParIntLampFlrCanUp`                | PAR radiation from internal lamp to floor up                           |                    |           |           |
| `a_rParSunCan`                         | PAR radiation from sun to canopy                                       |                    |           |           |
| `a_rParLampCan`                        | PAR radiation from lamp to canopy                                     |                    |           |           |
| `a_rParIntLampCan`                     | PAR radiation from internal lamp to canopy                            |                    |           |           |
| `a_tauHatCovNir`                       | Time constant for cover NIR                                           | s                  |           |           |
| `a_tauHatFlrNir`                       | Time constant for floor NIR                                           | s                  |           |           |
| `a_tauHatCanNir`                       | Time constant for canopy NIR                                          | s                  |           |           |
| `a_rhoHatCanNir`                       | Reflection coefficient for canopy NIR                                 |                    |           |           |
| `a_tauCovCanNir`                       | Time constant for cover canopy NIR                                    | s                  |           |           |
| `a_rhoCovCanNirUp`                     | Reflection coefficient for cover canopy NIR, upper limit              |                    |           |           |
| `a_rhoCovCanNirDn`                     | Reflection coefficient for cover canopy NIR, lower limit              |                    |           |           |
| `a_tauCovCanFlrNir`                    | Time constant for cover and floor NIR                                  | s                  |           |           |
| `a_rhoCovCanFlrNir`                    | Reflection coefficient for cover and floor NIR                        |                    |           |           |
| `a_aCanNir`                            | Area coefficient for canopy NIR                                       |                    |           |           |
| `a_aFlrNir`                            | Area coefficient for floor NIR                                        |                    |           |           |
| `a_rNirSunCan`                         | NIR radiation from sun to canopy                                      |                    |           |           |
| `a_rNirLampCan`                        | NIR radiation from lamp to canopy                                     |                    |           |           |
| `a_rNirIntLampCan`                     | NIR radiation from internal lamp to canopy                            |                    |           |           |
| `a_rNirSunFlr`                         | NIR radiation from sun to floor                                       |                    |           |           |
| `a_rNirLampFlr`                        | NIR radiation from lamp to floor                                      |                    |           |           |
| `a_rNirIntLampFlr`                     | NIR radiation from internal lamp to floor                             |                    |           |           |
| `a_rParSunFlr`                         | PAR radiation from sun to floor                                       |                    |           |           |
| `a_rParLampFlr`                        | PAR radiation from lamp to floor                                      |                    |           |           |
| `a_rParIntLampFlr`                     | PAR radiation from internal lamp to floor                             |                    |           |           |
| `a_rLampAir`                           | Radiation from lamp to air                                           |                    |           |           |
| `a_rIntLampAir`                        | Radiation from internal lamp to air                                  |                    |           |           |
| `a_rGlobSunAir`                        | Global solar radiation to air                                        |                    |           |           |
| `a_rGlobSunCovE`                       | Global solar radiation to cover external                             |                    |           |           |
| `a_tauThScrFirU`                       | Time constant for thermal screen FIR upper limit                     | s                  |           |           |
| `a_tauBlScrFirU`                       | Time constant for blind screen FIR upper limit                       | s                  |           |           |
| `a_aCan`                               | Area coefficient for canopy                                          |                    |           |           |
| `a_rCanCovIn`                          | Radiation from canopy to cover internal                              |                    |           |           |
| `a_rCanSky`                            | Radiation from canopy to sky                                         |                    |           |           |
| `a_rCanThScr`                          | Radiation from canopy to thermal screen                              |                    |           |           |
| `a_rCanFlr`                            | Radiation from canopy to floor                                       |                    |           |           |
| `a_rPipeCovIn`                         | Radiation from pipe to cover internal                                |                    |           |           |
| `a_rPipeSky`                           | Radiation from pipe to sky                                           |                    |           |           |
| `a_rPipeThScr`                         | Radiation from pipe to thermal screen                                |                    |           |           |
| `a_rPipeFlr`                           | Radiation from pipe to floor                                         |                    |           |           |
| `a_rPipeCan`                           | Radiation from pipe to canopy                                        |                    |           |           |
| `a_rFlrCovIn`                          | Radiation from floor to cover internal                              |                    |           |           |
| `a_rFlrSky`                            | Radiation from floor to sky                                          |                    |           |           |
| `a_rFlrThScr`                          | Radiation from floor to thermal screen                               |                    |           |           |
| `a_rThScrCovIn`                        | Radiation from thermal screen to cover internal                     |                    |           |           |
| `a_rThScrSky`                          | Radiation from thermal screen to sky                                 |                    |           |           |
| `a_rCovESky`                           | Radiation from cover to external sky                                 |                    |           |           |
| `a_rFirLampFlr`                        | Radiation from FIR to lamp floor                                     |                    |           |           |
| `a_rLampPipe`                          | Radiation from lamp to pipe                                          |                    |           |           |
| `a_rFirLampCan`                        | Radiation from FIR to lamp canopy                                    |                    |           |           |
| `a_rLampThScr`                         | Radiation from lamp to thermal screen                                |                    |           |           |
| `a_rLampCovIn`                         | Radiation from lamp to cover internal                               |                    |           |           |
| `a_rLampSky`                           | Radiation from lamp to sky                                           |                    |           |           |
| `a_rGroPipeCan`                        | Radiation from growth pipe to canopy                                 |                    |           |           |
| `a_rFlrBlScr`                          | Radiation from floor to blind screen                                |                    |           |           |
| `a_rPipeBlScr`                         | Radiation from pipe to blind screen                                 |                    |           |           |
| `a_rCanBlScr`                          | Radiation from canopy to blind screen                               |                    |           |           |
| `a_rBlScrThScr`                        | Radiation from blind screen to thermal screen                        |                    |           |           |
| `a_rBlScrCovIn`                        | Radiation from blind screen to cover internal                       |                    |           |           |
| `a_rBlScrSky`                          | Radiation from blind screen to sky                                  |                    |           |           |
| `a_rLampBlScr`                         | Radiation from lamp to blind screen                                 |                    |           |           |
| `a_fIntLampCanUp`                      | Fraction of internal lamp PAR absorbed by canopy, upper limit        |                    |           |           |
| `a_fIntLampCanDown`                    | Fraction of internal lamp PAR absorbed by canopy, lower limit        |                    |           |           |
| `a_rFirIntLampFlr`                     | Radiation from FIR to internal lamp floor                            |                    |           |           |
| `a_rIntLampPipe`                       | Radiation from internal lamp to pipe                                 |                    |           |           |
| `a_rFirIntLampCan`                     | Radiation from FIR to internal lamp canopy                           |                    |           |           |
| `a_rIntLampLamp`                      | Radiation from internal lamp to lamp                                |                    |           |           |
| `a_rIntLampBlScr`                     | Radiation from internal lamp to blind screen                         |                    |           |           |
| `a_rIntLampThScr`                     | Radiation from internal lamp to thermal screen                      |                    |           |           |
| `a_rIntLampCovIn`                     | Radiation from internal lamp to cover internal                      |                    |           |           |
| `a_rIntLampSky`                       | Radiation from internal lamp to sky                                 |                    |           |           |
| `a_aRoofU`                             | Area coefficient for roof                                             |                    |           |           |
| `a_aRoofUMax`                          | Maximum area coefficient for roof                                    |                    |           |           |
| `a_aRoofMin`                           | Minimum area coefficient for roof                                    |                    |           |           |
| `a_aSideU`                             | Area coefficient for side surfaces                                  |                    |           |           |
| `a_etaRoof`                            | Roof emissivity                                                        |                    |           |           |
| `a_etaRoofNoSide`                      | Roof emissivity without side surfaces                                |                    |           |           |
| `a_etaSide`                            | Side emissivity                                                       |                    |           |           |
| `a_cD`                                 | Drag coefficient                                                        |                    |           |           |
| `a_cW`                                 | Wind coefficient                                                       |                    |           |           |
| `a_fVentRoof2`                         | Fraction of ventilation through roof 2                               |                    |           |           |
| `a_fVentRoof2Max`                      | Maximum fraction of ventilation through roof 2                      |                    |           |           |
| `a_fVentRoof2Min`                      | Minimum fraction of ventilation through roof 2                      |                    |           |           |
| `a_fVentRoofSide2`                     | Fraction of ventilation through side 2                              |                    |           |           |
| `a_fVentSide2`                         | Fraction of ventilation through side 2                              |                    |           |           |
| `a_fLeakage`                           | Fraction of leakage                                                   |                    |           |           |
| `a_fVentRoof`                          | Fraction of ventilation through roof                                |                    |           |           |
| `a_fVentSide`                          | Fraction of ventilation through side                                |                    |           |           |
| `a_timeOfDay`                          | Time of day                                                            | s                  |           |           |
| `a_dayOfYear`                          | Day of the year                                                        | day                | 1         | 365       |
| `a_lampTimeOfDay`                      | Lamp time of day                                                        | s                  |           |           |
| `a_lampDayOfYear`                      | Lamp day of the year                                                    | day                | 1         | 365       |
| `a_lampNoCons`                        | Lamp non-consumption                                                   |                    |           |           |
| `a_linearLampSwitchOn`                 | Linear lamp switch on                                                  |                    |           |           |
| `a_linearLampSwitchOff`                | Linear lamp switch off                                                 |                    |           |           |
| `a_linearLampBothSwitches`             | Linear lamp both switches                                              |                    |           |           |
| `a_smoothLamp`                         | Smooth lamp                                                            |                    |           |           |
| `a_isDayInside`                        | Day inside                                                             |                    |           |           |
| `a_mechAllowed`                        | Mechanical cooling allowed                                              |                    |           |           |
| `a_hotBufAllowed`                      | Hot buffer allowed                                                      |                    |           |           |
| `a_heatSetPoint`                       | Heat set point                                                          | °C                 |           |           |
| `a_heatMax`                            | Maximum heat                                                           | °C                 |           |           |
| `a_co2SetPoint`                        | CO2 set point                                                          | ppm                |           |           |
| `a_co2InPpm`                           | CO2 concentration in ppm                                                | ppm                |           |           |
| `a_ventHeat`                           | Ventilation heat                                                        |                    |           |           |
| `a_rhIn`                               | Relative humidity inside                                                | %                  | 0         | 100       |
| `a_ventRh`                             | Ventilation relative humidity                                          | %                  | 0         | 100       |
| `a_ventCold`                           | Ventilation cold                                                        |                    |           |           |
| `a_thScrSp`                            | Thermal screen set point                                                |                    |           |           |
| `a_thScrCold`                          | Thermal screen cold                                                     |                    |           |           |
| `a_thScrHeat`                          | Thermal screen heat                                                     |                    |           |           |
| `a_thScrRh`                            | Thermal screen relative humidity                                        | %                  | 0         | 100       |
| `a_lampOn`                             | Lamp on                                                                 |                    |           |           |
| `a_intLampOn`                          | Internal lamp on                                                        |                    |           |           |
| `a_rhoTop`                             | Top reflection coefficient                                               |                    |           |           |
| `a_rhoAir`                             | Air reflection coefficient                                               |                    |           |           |
| `a_rhoAirMean`                         | Mean air reflection coefficient                                         |                    |           |           |
| `a_fThScr`                             | Thermal screen fraction                                                  |                    |           |           |
| `a_fBlScr`                             | Blind screen fraction                                                    |                    |           |           |
| `a_fScr`                               | Screen fraction                                                          |                    |           |           |
| `a_fVentForced`                        | Forced ventilation fraction                                              |                    |           |           |
| `a_hCanAir`                            | Heat transfer coefficient for canopy air                                |                    |           |           |
| `a_hAirFlr`                            | Heat transfer coefficient for air to floor                             |                    |           |           |
| `a_hAirThScr`                          | Heat transfer coefficient for air to thermal screen                     |                    |           |           |
| `a_hAirBlScr`                          | Heat transfer coefficient for air to blind screen                      |                    |           |           |
| `a_hAirOut`                            | Heat transfer coefficient for air to outside                           |                    |           |           |
| `a_hAirTop`                            | Heat transfer coefficient for air to top                               |                    |           |           |
| `a_hThScrTop`                          | Heat transfer coefficient for thermal screen to top                     |                    |           |           |
| `a_hBlScrTop`                          | Heat transfer coefficient for blind screen to top                       |                    |           |           |
| `a_hTopCovIn`                          | Heat transfer coefficient for top to cover internal                     |                    |           |           |
| `a_hTopOut`                            | Heat transfer coefficient for top to outside                            |                    |           |           |
| `a_hCovEOut`                           | Heat transfer coefficient for cover to external                        |                    |           |           |
| `a_hPipeAir`                           | Heat transfer coefficient for pipe to air                               |                    |           |           |
| `a_hFlrSo1`                            | Heat transfer coefficient for floor to soil 1                           |                    |           |           |
| `a_hSo1So2`                            | Heat transfer coefficient for soil 1 to soil 2                          |                    |           |           |
| `a_hSo2So3`                            | Heat transfer coefficient for soil 2 to soil 3                          |                    |           |           |
| `a_hSo3So4`                            | Heat transfer coefficient for soil 3 to soil 4                          |                    |           |           |
| `a_hSo4So5`                            | Heat transfer coefficient for soil 4 to soil 5                          |                    |           |           |
| `a_hSo5SoOut`                          | Heat transfer coefficient for soil 5 to outside                         |                    |           |           |
| `a_hCovInCovE`                         | Heat transfer coefficient for cover internal to external                |                    |           |           |
| `a_hLampAir`                           | Heat transfer coefficient for lamp to air                              |                    |           |           |
| `a_hGroPipeAir`                        | Heat transfer coefficient for growth pipe to air                       |                    |           |           |
| `a_hIntLampAir`                        | Heat transfer coefficient for internal lamp to air                     |                    |           |           |
| `a_sRs`                                | Radiative heat transfer coefficient                                     |                    |           |           |
| `a_cEvap3`                             | Evaporation coefficient 3                                               |                    |           |           |
| `a_cEvap4`                             | Evaporation coefficient 4                                               |                    |           |           |
| `a_rfRCan`                             | Radiative flux for canopy                                              |                    |           |           |
| `a_rfCo2`                              | Radiative flux for CO2                                                 |                    |           |           |
| `a_rfVp`                               | Radiative flux for vapor pressure                                      |                    |           |           |
| `a_rS`                                 | Solar radiation                                                          |                    |           |           |
| `a_vecCanAir`                          | Vector for canopy air                                                  |                    |           |           |
| `a_mvCanAir`                           | Mass flow rate for canopy air                                          |                    |           |           |
| `a_mvPadAir`                           | Mass flow rate for pad air                                             |                    |           |           |
| `a_mvFogAir`                           | Mass flow rate for fog air                                             |                    |           |           |
| `a_mvBlowAir`                          | Mass flow rate for blow air                                            |                    |           |           |
| `a_mvAirOutPad`                        | Mass flow rate for air out pad                                         |                    |           |           |
| `a_mvAirThScr`                         | Mass flow rate for air to thermal screen                               |                    |           |           |
| `a_mvAirBlScr`                         | Mass flow rate for air to blind screen                                 |                    |           |           |
| `a_mvTopCovIn`                         | Mass flow rate for top to cover internal                               |                    |           |           |
| `a_mvAirTop`                           | Mass flow rate for air to top                                          |                    |           |           |
| `a_mvTopOut`                           | Mass flow rate for top to outside                                      |                    |           |           |
| `a_mvAirOut`                           | Mass flow rate for air to outside                                      |                    |           |           |
| `a_lCanAir`                            | Length for canopy air                                                 |                    |           |           |
| `a_lAirThScr`                          | Length for air to thermal screen                                       |                    |           |           |
| `a_lAirBlScr`                          | Length for air to blind screen                                         |                    |           |           |
| `a_lTopCovIn`                          | Length for top to cover internal                                       |                    |           |           |
| `a_parCan`                             | PAR for canopy                                                          |                    |           |           |
| `a_j25CanMax`                          | Maximum value for canopy                                                |                    |           |           |
| `a_gamma`                              | Gamma coefficient                                                       |                    |           |           |
| `a_co2Stom`                            | CO2 stomatal conductance                                                |                    |           |           |
| `a_jPot`                               | Potential value for canopy                                              |                    |           |           |
| `a_j`                                  | General parameter j                                                     |                    |           |           |
| `a_p`                                  | General parameter p                                                     |                    |           |           |
| `a_r`                                  | General parameter r                                                     |                    |           |           |
| `a_hAirBuf`                            | Heat transfer coefficient for air buffer                               |                    |           |           |
| `a_mcAirBuf`                           | Mass coefficient for air buffer                                        |                    |           |           |
| `a_gTCan24`                            | Growth temperature for canopy 24                                       |                    |           |           |
| `a_hTCan24`                            | Heat transfer coefficient for canopy 24                                |                    |           |           |
| `a_hTCan`                              | Heat transfer coefficient for canopy                                    |                    |           |           |
| `a_hTCanSum`                           | Heat transfer coefficient for canopy sum                               |                    |           |           |
| `a_hBufOrg`                            | Heat buffer origin coefficient                                          |                    |           |           |
| `a_mcBufLeaf`                          | Mass coefficient for buffer leaf                                        |                    |           |           |
| `a_mcBufStem`                          | Mass coefficient for buffer stem                                        |                    |           |           |
| `a_mcBufFruit`                         | Mass coefficient for buffer fruit                                       |                    |           |           |
| `a_mcBufAir`                           | Mass coefficient for buffer air                                         |                    |           |           |
| `a_mcLeafAir`                          | Mass coefficient for leaf air                                           |                    |           |           |
| `a_mcStemAir`                          | Mass coefficient for stem air                                           |                    |           |           |
| `a_mcFruitAir`                         | Mass coefficient for fruit air                                          |                    |           |           |
| `a_mcOrgAir`                           | Mass coefficient for organic air                                       |                    |           |           |
| `a_mcLeafHar`                          | Mass coefficient for leaf harvest                                      |                    |           |           |
| `a_mcFruitHar`                         | Mass coefficient for fruit harvest                                     |                    |           |           |
| `a_mcAirCan`                           | Mass coefficient for air canopy                                         |                    |           |           |
| `a_mcAirTop`                           | Mass coefficient for air top                                           |                    |           |           |
| `a_mcTopOut`                           | Mass coefficient for top out                                           |                    |           |           |
| `a_mcAirOut`                           | Mass coefficient for air out                                            |                    |           |           |
| `a_hBoilPipe`                          | Heat transfer coefficient for boiling pipe                             |                    |           |           |
| `a_hBoilGroPipe`                       | Heat transfer coefficient for boiling growth pipe                      |                    |           |           |
| `a_mcExtAir`                           | Mass coefficient for external air                                      |                    |           |           |
| `a_mcBlowAir`                          | Mass coefficient for blow air                                          |                    |           |           |
| `a_mcPadAir`                           | Mass coefficient for pad air                                           |                    |           |           |
| `a_hPadAir`                            | Heat transfer coefficient for pad air                                  |                    |           |           |
| `a_hPasAir`                            | Heat transfer coefficient for passive air                              |                    |           |           |
| `a_hBlowAir`                           | Heat transfer coefficient for blow air                                 |                    |           |           |
| `a_hAirPadOut`                         | Heat transfer coefficient for air to pad out                           |                    |           |           |
| `a_hAirOutPad`                         | Heat transfer coefficient for air to out pad                           |                    |           |           |
| `a_lAirFog`                            | Length for air fog                                                     |                    |           |           |
| `a_hIndPipe`                           | Heat transfer coefficient for indoor pipe                              |                    |           |           |
| `a_hGeoPipe`                           | Heat transfer coefficient for geothermal pipe                          |                    |           |           |
| `a_hLampCool`                          | Heat transfer coefficient for lamp cooling                             |                    |           |           |
| `a_hecMechAir`                         | Mechanical heat transfer coefficient for air                           |                    |           |           |
| `a_hAirMech`                           | Mechanical heat transfer coefficient for air                           |                    |           |           |
| `a_mvAirMech`                          | Mechanical air mass flow rate                                          |                    |           |           |
| `a_lAirMech`                           | Length for mechanical air                                             |                    |           |           |
| `a_hBufHotPipe`                        | Heat transfer coefficient for hot buffer pipe                          |                    |           |           |


### P: Parameters (constants)

| Parameter | Description | Unit | Min Value | Max Value |
|-----------|-------------|------|-----------|-----------|
| p_sigma | Stefan-Boltzmann constant | W m^-2 K^-4 | 5.67e-8 | 5.67e-8 |
| p_g | Acceleration of gravity | m s^-2 | 9.81 | 9.81 |
| p_R | Molar gas constant | J kmol^-1 K^-1 | 8314 | 8314 |
| p_L | Latent heat of evaporation | J kg^-1 | 2.45e6 | 2.45e6 |
| p_gamma | Psychrometric constant | Pa K^-1 | 65.8 | 65.8 |
| p_cPAir | Specific heat capacity of air | J K^-1 kg^-1 | 1e3 | 1e3 |
| p_cPSteel | Specific heat capacity of steel | J K^-1 kg^-1 | 0.64e3 | 0.64e3 |
| p_cPWater | Specific heat capacity of water | J K^-1 kg^-1 | 4.18e3 | 4.18e3 |
| p_etaGlobNir | Ratio of NIR in global radiation | - | 0.5 | 0.5 |
| p_etaGlobPar | Ratio of PAR in global radiation | - | 0.5 | 0.5 |
| p_epsCan | FIR emission coefficient of canopy | - | 1 | 1 |
| p_epsSky | FIR emission coefficient of the sky | - | 1 | 1 |
| p_rhoCanPar | PAR reflection coefficient | - | 0.07 | 0.07 |
| p_rhoCanNir | NIR reflection coefficient of the top of the canopy | - | 0.35 | 0.35 |
| p_k1Par | PAR extinction coefficient of the canopy | - | 0.7 | 0.7 |
| p_k2Par | PAR extinction coefficient of the canopy for light reflected from the floor | - | 0.7 | 0.7 |
| p_kNir | NIR extinction coefficient of the canopy | - | 0.27 | 0.27 |
| p_kFir | FIR extinction coefficient of the canopy | - | 0.94 | 0.94 |
| p_rhoAir0 | Density of air at sea level | kg m^-3 | 1.2 | 1.2 |
| p_rhoSteel | Density of steel | kg m^-3 | 7850 | 7850 |
| p_rhoWater | Density of water | kg m^-3 | 1e3 | 1e3 |
| p_mAir | Molar mass of air | kg kmol^-1 | 28.96 | 28.96 |
| p_mWater | Molar mass of water | kg kmol^-1 | 18 | 18 |
| p_alfaLeafAir | Convective heat exchange coefficient from the canopy leaf to the greenhouse air | W m^-2 K^-1 | 5 | 5 |
| p_capLeaf | Heat capacity of canopy leaves | J K^-1 m^-2 | 1.2e3 | 1.2e3 |
| p_rB | Boundary layer resistance of the canopy for transpiration | s m^-1 | 275 | 275 |
| p_rSMin | Minimum canopy resistance for transpiration | s m^-1 | 82 | 82 |
| p_cEvap1 | Coefficient for radiation effect on stomatal resistance | W m^-2 | 4.3 | 4.3 |
| p_cEvap2 | Coefficient for radiation effect on stomatal resistance | W m^-2 | 0.54 | 0.54 |
| p_cEvap3Day | Coefficient for CO2 effect on stomatal resistance (day) | ppm^-2 | 6.1e-7 | 6.1e-7 |
| p_cEvap3Night | Coefficient for CO2 effect on stomatal resistance (night) | ppm^-2 | 1.1e-11 | 1.1e-11 |
| p_cEvap4Day | Coefficient for vapor pressure effect on stomatal resistance (day) | Pa^-2 | 4.3e-6 | 4.3e-6 |
| p_cEvap4Night | Coefficient for vapor pressure effect on stomatal resistance (night) | Pa^-2 | 5.2e-6 | 5.2e-6 |
| p_sRs | Slope of smoothed stomatal resistance model | m W^-2 | -1 | -1 |
| p_hSo1 | Thickness of soil layer 1 | m | 0.04 | 0.04 |
| p_hSo2 | Thickness of soil layer 2 | m | 0.08 | 0.08 |
| p_hSo3 | Thickness of soil layer 3 | m | 0.16 | 0.16 |
| p_hSo4 | Thickness of soil layer 4 | m | 0.32 | 0.32 |
| p_hSo5 | Thickness of soil layer 5 | m | 0.64 | 0.64 |
| p_hSoOut | Thickness of the external soil layer | m | 1.28 | 1.28 |
| p_omega | Yearly frequency to calculate soil temperature | s^-1 | 1.99e-7 | 1.99e-7 |
| p_etaGlobAir | Ratio of global radiation absorbed by the greenhouse construction | - | 0.1 | 0.1 |
| p_psi | Mean greenhouse cover slope | degrees | 25 | 25 |
| p_aFlr | Floor area of greenhouse | m^2 | 1.4e4 | 1.4e4 |
| p_aCov | Surface of the cover including side walls | m^2 | 1.8e4 | 1.8e4 |
| p_hAir | Height of the main compartment | m | 3.8 | 3.8 |
| p_hGh | Mean height of the greenhouse | m | 4.2 | 4.2 |
| p_cHecIn | Convective heat exchange between cover and outdoor air | W m^-2 K^-1 | 1.86 | 1.86 |
| p_cHecOut1 | Convective heat exchange parameter between cover and outdoor air | W m^-2 K^-1 | 2.8 | 2.8 |
| p_cHecOut2 | Convective heat exchange parameter between cover and outdoor air | J m^-3 K^-1 | 1.2 | 1.2 |
| p_cHecOut3 | Convective heat exchange parameter between cover and outdoor air | - | 1 | 1 |
| p_hElevation | Altitude of greenhouse | m | 0 | - |
| p_aRoof | Maximum roof ventilation area | - | 1.4e3 | 1.4e3 |
| p_hVent | Vertical dimension of single ventilation opening | m | 0.68 | 0.68 |
| p_etaInsScr | Porosity of the insect screen | - | 1 | 1 |
| p_aSide | Side ventilation area | - | 0 | 0 |
| p_cDgh | Ventilation discharge coefficient | - | 0.75 | 0.75 |
| p_cLeakage | Greenhouse leakage coefficient | - | 1e-4 | 1e-4 |
| p_cWgh | Ventilation global wind pressure coefficient | - | 0.09 | 0.09 |
| p_hSideRoof | Vertical distance between mid points of side wall and roof ventilation opening | m | 0 | 0 |
| p_epsRfFir | FIR emission coefficient of the roof | - | 0.85 | 0.85 |
| p_rhoRf | Density of the roof layer | kg m^-3 | 2.6e3 | 2.6e3 |
| p_rhoRfNir | NIR reflection coefficient of the roof | - | 0.13 | 0.13 |
| p_rhoRfPar | PAR reflection coefficient of the roof | - | 0.13 | 0.13 |
| p_rhoRfFir | FIR reflection coefficient of the roof | - | 0.15 | 0.15 |
| p_tauRfNir | NIR transmission coefficient of the roof | - | 0.85 | 0.85 |
| p_tauRfPar | PAR transmission coefficient of the roof | - | 0.85 | 0.85 |
| p_tauRfFir | FIR transmission coefficient of the roof | - | 0 | 0 |
| p_lambdaRf | Thermal heat conductivity of the roof | W m^-1 K^-1 | 1.05 | 1.05 |
| p_cPRf | Specific heat capacity of roof layer | J K^-1 kg^-1 | 0.84e3 | 0.84e3 |
| p_hRf | Thickness of roof layer | m | 4e-3 | 4e-3 |
| p_epsShScrPerFir | FIR emission coefficient of the whitewash | - | 0 | 0 |
| p_rhoShScrPer | Density of the whitewash | - | 0 | 0 |
| p_rhoShScrPerNir | NIR reflection coefficient of whitewash | - | 0 | 0 |
| p_rhoShScrPerPar | PAR reflection coefficient of whitewash | - | 0 | 0 |
| p_rhoShScrPerFir | FIR reflection coefficient of whitewash | - | 0 | 0 |
| p_tauShScrPerNir | NIR transmission coefficient of whitewash | - | 1 | 1 |
| p_tauShScrPerPar | PAR transmission coefficient of whitewash | - | 1 | 1 |
| p_tauShScrPerFir | FIR transmission coefficient of whitewash | - | 1 | 1 |
| p_lambdaShScrPer | Thermal heat conductivity of the whitewash | W m^-1 K^-1 | inf | inf |
| p_cPShScrPer | Specific heat capacity of the whitewash | J K^-1 kg^-1 | 0 | 0 |
| p_hShScrPer | Thickness of the whitewash | m | 0 | 0 |
| p_rhoShScrNir | NIR reflection coefficient of shadow screen | - | 0 | 0 |
| p_rhoShScrPar | PAR reflection coefficient of shadow screen | - | 0 | 0 |
| p_rhoShScrFir | FIR reflection coefficient of shadow screen | - | 0 | 0 |
| p_tauShScrNir | NIR transmission coefficient of shadow screen | - | 1 | 1 |
| p_tauShScrPar | PAR transmission coefficient of shadow screen | - | 1 | 1 |
| p_tauShScrFir | FIR transmission coefficient of shadow screen | - | 1 | 1 |
| p_etaShScrCd | Effect of shadow screen on discharge coefficient | - | 0 | 0 |
| p_etaShScrCw | Effect of shadow screen on wind pressure coefficient | - | 0 | 0 |
| p_kShScr | Shadow screen flux coefficient | m^3 m^-2 K^-2/3 s^-1 | 0 | 0 |
| p_epsThScrFir | FIR emissions coefficient of the thermal screen | - | 0.67 | 0.67 |
| p_rhoThScr | Density of thermal screen | kg m^-3 | 0.2e3 | 0.2e3 |
| p_rhoThScrNir | NIR reflection coefficient of thermal screen | - | 0.35 | 0.35 |
| p_rhoThScrPar | PAR reflection coefficient of thermal screen | - | 0.35 | 0.35 |
| p_rhoThScrFir | FIR reflection coefficient of thermal screen | - | 0.18 | 0.18 |
| p_tauThScrNir | NIR transmission coefficient of thermal screen | - | 0.6 | 0.6 |
| p_tauThScrPar | PAR transmission coefficient of thermal screen | - | 0.6 | 0.6 |
| p_tauThScrFir | FIR transmission coefficient of thermal screen | - | 0.15 | 0.15 |
| p_cPThScr | Specific heat capacity of thermal screen | J kg^-1 K^-1 | 1.8e3 | 1.8e3 |
| p_hThScr | Thickness of thermal screen | m | 0.35e-3 | 0.35e-3 |
| p_kThScr | Thermal screen flux coefficient | m^3 m^-2 K^-2/3 s^-1 | 0.05e-3 | 0.05e-3 |
| p_epsBlScrFir | FIR emissions coefficient of the blackout screen | - | 0.67 | 0.67 |
| p_rhoBlScr | Density of blackout screen | kg m^-3 | 0.2e3 | 0.2e3 |
| p_rhoBlScrNir | NIR reflection coefficient of blackout screen | - | 0.35 | 0.35 |
| p_rhoBlScrPar | PAR reflection coefficient of blackout screen | - | 0.35 | 0.35 |
| p_tauBlScrNir | NIR transmission coefficient of blackout screen | - | 0.01 | 0.01 |
| p_tauBlScrPar | PAR transmission coefficient of blackout screen | - | 0.01 | 0.01 |
| p_tauBlScrFir | FIR transmission coefficient of blackout screen | - | 0.7 | 0.7 |
| p_cPBlScr | Specific heat capacity of blackout screen | J kg^-1 K^-1 | 1.8e3 | 1.8e3 |
| p_hBlScr | Thickness of blackout screen | m | 0.35e-3 | 0.35e-3 |
| p_kBlScr | Blackout screen flux coefficient | m³ m⁻² K⁻²/³ s⁻¹ | - | - |
| p_epsFlr | FIR emission coefficient of the floor | - | 0 | 1 |
| p_rhoFlr | Density of the floor | kg m^{-3} | - | - |
| p_rhoFlrNir | NIR reflection coefficient of the floor | - | 0 | 1 |
| p_rhoFlrPar | PAR reflection coefficient of the floor | - | 0 | 1 |
| p_lambdaFlr | Thermal heat conductivity of the floor | W m^{-1} K^{-1} | - | - |
| p_cPFlr | Specific heat capacity of the floor | J kg^{-1} K^{-1} | - | - |
| p_hFlr | Thickness of floor | m | - | - |
| p_rhoCpSo | Volumetric heat capacity of the soil | J m^{-3} K^{-1} | - | - |
| p_lambdaSo | Thermal heat conductivity of the soil layers | W m^{-1} K^{-1} | - | - |
| p_epsPipe | FIR emission coefficient of the heating pipes | - | 0 | 1 |
| p_phiPipeE | External diameter of pipes | m | - | - |
| p_phiPipeI | Internal diameter of pipes | m | - | - |
| p_lPipe | Length of heating pipes per gh floor area | m m^{-2} | - | - |
| p_pBoil | Capacity of the heating system | W | - | - |
| p_phiExtCo2 | Capacity of external CO2 source | mg s^{-1} | - | - |
| p_capPipe | Heat capacity of heating pipes | J K^{-1} m^{-2} | - | - |
| p_aPipe | Surface of pipes for floor area | - | - | - |
| p_rhoAir | Density of the air | kg m^{-3} | - | - |
| p_pressure | Absolute air pressure at given elevation | Pa | - | - |
| p_capAir | Heat capacity of air in main compartment | J K^{-1} m^{-2} | - | - |
| p_capFlr | Heat capacity of the floor | J K^{-1} m^{-2} | - | - |
| p_capSo1 | Heat capacity of soil layer 1 | J K^{-1} m^{-2} | - | - |
| p_capSo2 | Heat capacity of soil layer 2 | J K^{-1} m^{-2} | - | - |
| p_capSo3 | Heat capacity of soil layer 3 | J K^{-1} m^{-2} | - | - |
| p_capSo4 | Heat capacity of soil layer 4 | J K^{-1} m^{-2} | - | - |
| p_capSo5 | Heat capacity of soil layer 5 | J K^{-1} m^{-2} | - | - |
| p_capThScr | Heat capacity of thermal screen | J K^{-1} m^{-2} | - | - |
| p_capTop | Heat capacity of air in top compartments | J K^{-1} m^{-2} | - | - |
| p_capBlScr | Heat capacity of blackout screen | J K^{-1} m^{-2} | - | - |
| p_capCo2Air | Capacity for CO2 in air | m | - | - |
| p_capCo2Top | Capacity for CO2 in top | m | - | - |
| p_fCanFlr | View factor from canopy to floor | - | - | - |
| p_globJtoUmol | Conversion factor from global radiation to PAR | umol{photons} J^{-1} | - | - |
| p_j25LeafMax | Maximal rate of electron transport at 25°C of the leaf | umol{e-} m^{-2}{leaf} s^{-1} | - | - |
| p_cGamma | Effect of canopy temperature on CO2 compensation point | umol{co2} mol^{-1}{air} K^{-1} | - | - |
| p_etaCo2AirStom | Conversion from greenhouse air CO2 concentration and stomatal CO2 concentration | umol{co2} mol^{-1}{air} | - | - |
| p_eJ | Activation energy for Jpot calculation | J mol^{-1} | - | - |
| p_t25k | Reference temperature for Jpot calculation | K | - | - |
| p_S | Entropy term for Jpot calculation | J mol^{-1} K^{-1} | - | - |
| p_H | Deactivation energy for Jpot calculation | J mol^{-1} | - | - |
| p_theta | Degree of curvature of the electron transport rate | - | - | - |
| p_alpha | Conversion factor from photons to electrons including efficiency term | umol{e-} umol^{-1}{photons} | - | - |
| p_mCh2o | Molar mass of CH2O | mg umol^{-1} | - | - |
| p_mCo2 | Molar mass of CO2 | mg umol^{-1} | - | - |
| p_parJtoUmolSun | Conversion factor of sun's PAR from J to umol{photons} | umol{photons} J^{-1} | - | - |
| p_laiMax | Leaf area index | m^{2}{leaf} m^{-2}{floor} | - | - |
| p_sla | Specific leaf area | m^{2}{leaf} mg^{-1}{leaf} | - | - |
| p_rgr | Relative growth rate | kg{dw grown} kg^{-1}{existing dw} s^{-1} | - | - |
| p_cLeafMax | Maximum leaf size | mg{leaf} m^{-2} | - | - |
| p_cFruitMax | Maximum fruit size | mg{fruit} m^{-2} | - | - |
| p_cFruitG | Fruit growth respiration coefficient | - | - | - |
| p_cLeafG | Leaf growth respiration coefficient | - | - | - |
| p_cStemG | Stem growth respiration coefficient | - | - | - |
| p_cRgr | Regression coefficient in maintenance respiration function | s^{-1} | - | - |
| p_q10m | Q10 value of temperature effect on maintenance respiration | - | - | - |
| p_cFruitM | Fruit maintenance respiration coefficient | mg mg^{-1} s^{-1} | - | - |
| p_cLeafM | Leaf maintenance respiration coefficient | mg mg^{-1} s^{-1} | - | - |
| p_cStemM | Stem maintenance respiration coefficient | mg mg^{-1} s^{-1} | - | - |
| p_rgFruit | Potential fruit growth coefficient | mg m^{-2} s^{-1} | - | - |
| p_rgLeaf | Potential leaf growth coefficient | mg m^{-2} s^{-1} | - | - |
| p_rgStem | Potential stem growth coefficient | mg m^{-2} s^{-1} | - | - |
| p_cBufMax | Maximum capacity of carbohydrate buffer | mg m^{-2} | - | - |
| p_cBufMin | Minimum capacity of carbohydrate buffer | mg m^{-2} | - | - |
| p_tCan24Max | Inhibition of carbohydrate flow because of high temperatures | °C | - | - |
| p_tCan24Min | Inhibition of carbohydrate flow because of low temperatures | °C | - | - |
| p_tCanMax | Inhibition of carbohydrate flow because of high instantaneous temperatures | °C | - | - |
| p_tCanMin | Inhibition of carbohydrate flow because of low instantaneous temperatures | °C | - | - |
| p_tEndSum | Temperature sum where crop is fully generative | °C day | - | - |
| p_rhMax | Upper bound on relative humidity | % | 0 | 100 |
| p_dayThresh | Threshold to consider switch from night to day | W m^{-2} | - | - |
| p_tSpDay | Heat is on below this point in day | °C | - | - |
| p_tSpNight | Heat is on below this point in night | °C | - | - |
| p_tHeatBand | P-band for heating | °C | - | - |
| p_tVentOff | Distance from heating setpoint where ventilation stops | °C | - | - |
| p_tScreenOn | Distance from screen setpoint where screen is on | °C | - | - |
| p_thScrSpDay | Screen is closed at day when outdoor is below this temperature | °C | - | - |
| p_thScrSpNight | Screen is closed at night when outdoor is below this temperature | °C | - | - |
| p_thScrPband | P-band for thermal screen | °C | - | - |
| p_co2SpDay | CO2 is supplied if CO2 is below this point during day | ppm | - | - |
| p_co2Band | P-band for CO2 supply | ppm | - | - |
| p_heatDeadZone | Zone between heating setpoint and ventilation setpoint | °C | - | - |
| p_ventHeatPband | P-band for ventilation due to excess heat | °C | - | - |
| p_ventColdPband | P-band for ventilation due to low indoor temperature | °C | - | - |
| p_ventRhPband | P-band for ventilation due to relative humidity | % | - | - |
| p_thScrRh | Relative humidity where thermal screen is forced to open, with respect to rhMax | % | - | - |
| p_thScrRhPband | P-band for thermal screen opening due to excess relative humidity | % | - | - |
| p_thScrDeadZone | Zone between heating setpoint and point where screen opens | °C | - | - |
| p_lampsOn | Time of day to switch on lamps | hours since midnight | 0 | 24 |
| p_lampsOff | Time of day to switch off lamps | hours since midnight | 0 | 24 |
| p_dayLampStart | Day of year when lamps start | day of year | 1 | 365 |
| p_dayLampStop | Day of year when lamps stop | day of year | 1 | 365 |
| p_lampsOffSun | Lamps are switched off if global radiation is above this value | W m^{-2} | - | - |
| p_lampRadSumLimit | Predicted daily radiation sum from the sun where lamps are not used that day | MJ m^{-2} day^{-1} | - | - |
| p_lampExtraHeat | Control for lamps due to too much heat | °C | - | - |
| p_blScrExtraRh | Control for blackout screen due to humidity | % | - | - |
| p_useBlScr | Determines whether a blackout screen is used | - | 0 | 1 |
| p_mechCoolPband | P-band for mechanical cooling | °C | - | - |
| p_mechDehumidPband | P-band for mechanical dehumidification | % | - | - |
| p_heatBufPband | P-band for heating from the buffer | °C | - | - |
| p_mechCoolDeadZone | Zone between heating setpoint and mechanical cooling setpoint | °C | - | - |
| p_epsGroPipe | Emissivity of grow pipes | - | 0 | 1 |
| p_lGroPipe | Length of grow pipes per gh floor area | m m^{-2} | - | - |
| p_phiGroPipeE | External diameter of grow pipes | m | - | - |
| p_phiGroPipeI | Internal diameter of grow pipes | m | - | - |
| p_aGroPipe | Surface area of pipes for floor area | m^{2}{pipe} m^{-2}{floor} | - | - |
| p_pBoilGro | Capacity of the grow pipe heating system | W | - | - |
| p_capGroPipe | Heat capacity of grow pipes | J K^{-1} m^{-2} | - | - |
| p_thetaLampMax | Maximum intensity of lamps | W m^{-2} | - | - |
| p_heatCorrection | Correction for temperature setpoint when lamps are on | °C | - | - |
| p_etaLampPar | Fraction of lamp input converted to PAR | - | 0 | 1 |
| p_etaLampNir | Fraction of lamp input converted to NIR | - | 0 | 1 |
| p_tauLampPar | Transmissivity of lamp layer to PAR | - | 0 | 1 |
| p_rhoLampPar | Reflectivity of lamp layer to PAR | - | 0 | 1 |
| p_tauLampNir | Transmissivity of lamp layer to NIR | - | 0 | 1 |
| p_rhoLampNir | Reflectivity of lamp layer to NIR | - | 0 | 1 |
| p_tauLampFir | Transmissivity of lamp layer to FIR | - | 0 | 1 |
| p_aLamp | Lamp area | m^{2}{lamp} m^{-2}{floor} | - | - |
| p_epsLampTop | Emissivity of top side of lamp | - | 0 | 1 |
| p_epsLampBottom | Emissivity of bottom side of lamp | - | 0 | 1 |
| p_capLamp | Heat capacity of lamp | J K^{-1} m^{-2} | - | - |
| p_cHecLampAir | Heat exchange coefficient of lamp | W m^{-2} K^{-1} | - | - |
| p_etaLampCool | Fraction of lamp input removed by cooling | - | 0 | 1 |
| p_zetaLampPar | J to umol conversion of PAR output of lamp | J{PAR} umol{PAR}^{-1} | - | - |
| p_vIntLampPos | Vertical position of the interlights within the canopy | - | 0 | 1 |
| p_fIntLampDown | Fraction of interlight light output (PAR and NIR) that goes downwards | - | 0 | 1 |
| p_capIntLamp | Heat capacity of interlight lamp | J K^{-1} m^{-2} | - | - |
| p_etaIntLampPar | Fraction of interlight lamp input converted to PAR | - | 0 | 1 |
| p_etaIntLampNir | Fraction of interlight lamp input converted to NIR | - | 0 | 1 |
| p_aIntLamp | Interlight lamp area | m^{2}{lamp} m^{-2}{floor} | - | - |
| p_epsIntLamp | Emissivity of interlight | - | 0 | 1 |
| p_thetaIntLampMax | Maximum intensity of interlights | W m^{-2} | - | - |
| p_zetaIntLampPar | J to umol conversion of PAR output of interlight | J{PAR} umol{PAR}⁻¹ | - | - |
| p_cHecIntLampAir | Heat exchange coefficient of interlights | W m⁻² K⁻¹ | - | - |
| p_tauIntLampFir | Transmissivity of FIR through the interlights | - | - | - |
| p_k1IntPar | PAR extinction coefficient of the canopy for light coming from the interlights | - | - | - |
| p_k2IntPar | PAR extinction coefficient of the canopy for light coming from the interlights through the floor | - | - | - |
| p_kIntNir | NIR extinction coefficient of the canopy for light coming from the interlights | - | - | - |
| p_kIntFir | FIR extinction coefficient of the canopy for light coming from the interlights | - | - | - |
| p_etaMgPpm | CO2 conversion factor from mg m⁻³ to ppm | ppm mg⁻¹ m³ | - | - |
| p_etaRoofThr | Ratio between roof vent area and total vent area where no chimney effects are assumed | - | - | - |
| p_rCanSp | Radiation value above the canopy when night becomes day | W m⁻² | - | - |
| p_cLeakTop | Fraction of leakage ventilation going from the top | - | - | - |
| p_minWind | Wind speed where the effect of wind on leakage begins | m s⁻¹ | - | - |