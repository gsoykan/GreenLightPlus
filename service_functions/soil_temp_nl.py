import numpy as np


def soil_temp_nl(time):
    """
    Estimate of the soil temperature in the Netherlands at a given time of year.
    Based on Figure 3 in:
    Jacobs, A. F. G., Heusinkveld, B. G. & Holtslag, A. A. M.
    Long-term record and analysis of soil temperatures and soil heat fluxes in
    a grassland area, The Netherlands. Agric. For. Meteorol. 151, 774–780 (2011).

    Parameters:
        time (float or np.ndarray): Seconds since beginning of the year [s]

    Returns:
        soilT (float or np.ndarray): Soil temperature at 1 meter depth at given time [°C]
    """
    SECS_IN_YEAR = 3600 * 24 * 365
    soilT = 10 + 5 * np.sin((2 * np.pi * (time + 0.625 * SECS_IN_YEAR) / SECS_IN_YEAR))

    return soilT
