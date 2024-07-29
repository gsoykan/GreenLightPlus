from core.greenhouse_geometry import GreenhouseGeometry

if __name__ == '__main__':
    """
    To display the generated models use the OpenStudio Application
    https://openstudio.net/downloads
    https://github.com/openstudiocoalition/OpenStudioApplication
    https://github.com/openstudiocoalition/OpenStudioApplication/releases
    """
    # Define different roof types
    roof_types = [
        "triangle",
        "half_circle",
        "flat_arch",
        "gothic_arch",
        "sawtooth",
        "sawtooth_arch",
    ]

    for roof_type in roof_types:
        print(f"Creating greenhouse with {roof_type} roof")
        # Set basic greenhouse parameters
        wall_height = 6.5  # Ridge height {m}
        wall_width = 4  # Width of each roof segment {m}
        wall_length = 1.67  # Greenhouse length {m}
        num_segments = 6  # Number of roof segments
        slope = 22  # Roof slope angle {°}
        number_length = 10  # Number of greenhouses in length direction
        number_width = 10  # Number of greenhouses in width direction
        time_step = 60  # Time step (minutes)

        # Create a GreenhouseGeometry instance
        greenhouse_model = GreenhouseGeometry(roof_type=roof_type,
                                              slope=slope,
                                              wall_height=wall_height,
                                              wall_width=wall_width,
                                              wall_length=wall_length,
                                              num_segments=num_segments,
                                              time_step=time_step,
                                              number_width=number_width,
                                              number_length=number_length,
                                              max_indoor_temp=60,
                                              min_indoor_temp=0,
                                              max_outdoor_temp=60,
                                              min_outdoor_temp=0,
                                              max_delta_temp=1,
                                              max_wind_speed=30,
                                              start_month=4,
                                              start_day=1,
                                              end_month=4,
                                              end_day=7)

        # Generate the greeenhouse model
        greenhouse_model.create_houses()
