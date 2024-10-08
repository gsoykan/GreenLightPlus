# File path: GreenLightPlus/create_green_light_model/set_gl_control_init.py
"""
Copyright Statement:

This Python version of the code is based on the open-source Matlab code released by David Katzin at Wageningen University and is subject to his original copyright.

Original Matlab code author: David Katzin
Original author's email: david.katzin@wur.nl, david.katzin1@gmail.com

David Katzin, Simon van Mourik, Frank Kempkes, and Eldert J. Van Henten. 2020. “GreenLight - An Open Source Model for Greenhouses with Supplemental Lighting: Evaluation of Heat Requirements under LED and HPS Lamps.” Biosystems Engineering 194: 61–81. https://doi.org/10.1016/j.biosystemseng.2020.03.010


New Python code author: Daidai Qiu
Author's email: qiu.daidai@outlook.com

This code is licensed under the GNU GPLv3 License. For details, see the LICENSE file.
"""
import numpy as np
import pandas as pd


def set_gl_control_init(gl, controls=None):
    """
    Set control variables for the greenhouse model.

    Args:
        gl (dict): GreenLight greenhouse model.
        controls (ndarray, optional): Control trajectories. Default is None.

    Returns:
        dict: Modified GreenLight greenhouse model.

    Note:
        - If controls is not None, the control trajectories are added to the greenhouse model.
        - The pipe temperatures are defined as inputs and added to the greenhouse model.
        - The controls not considered in the model are set to 0.
    """

    u = gl["u"]
    d = gl["d"]

    if controls is not None:

        # Ensure controls is a numpy array
        # controls = np.array(controls)
        # controls = controls.to_numpy()

        print("controls is not None")

        time = controls[:, 0].reshape(-1, 1)

        # Add control trajectories
        u['thScr'] = np.hstack(
            [time, controls[:, 1].reshape(-1, 1)])  # Note the change of index
        u['blScr'] = np.hstack([time, controls[:, 2].reshape(-1, 1)])
        u['roof'] = np.hstack([time, controls[:, 3].reshape(-1, 1)])

        # Pipe temperature as input
        d['tPipe'] = np.hstack([time, controls[:, 4].reshape(-1, 1)])  # Modify index
        d['tGroPipe'] = np.hstack([time, controls[:, 5].reshape(-1, 1)])

        # Check if the pipe is about to turn off
        result = (controls[:, 4] != 0) & (
            np.roll(controls[:, 4], shift=-1) == 0)
        d['pipeSwitchOff'] = np.hstack([time, result.reshape(-1, 1)])

        # Check if the grow pipe is about to turn off
        result = (controls[:, 5] != 0) & (
            np.roll(controls[:, 5], shift=-1) == 0)
        d['groPipeSwitchOff'] = np.hstack([time, result.reshape(-1, 1)])

        # Other controls not considered
        u['lamp'] = np.hstack([time, controls[:, 6].reshape(-1, 1)])
        u['extCo2'] = np.hstack(
            [time, controls[:, 8].reshape(-1, 1)])  # The index may need adjustment
        u['intLamp'] = np.hstack(
            [time, controls[:, 7].reshape(-1, 1)])  # The index may need adjustment
        u['boil'] = np.hstack([time, np.zeros_like(time)])
        u['shScrPer'] = np.hstack([time, np.zeros_like(time)])
        u['side'] = np.hstack([time, np.zeros_like(time)])
        u['boilGro'] = np.hstack([time, np.zeros_like(time)])
        u['shScr'] = np.hstack([time, np.zeros_like(time)])

    else:

        # set initial values of control variables
        gl["u"]["boil"] = 0
        gl["u"]["boilGro"] = 0
        gl["u"]["extCo2"] = 0
        gl["u"]["shScr"] = 0
        gl["u"]["shScrPer"] = 0
        gl["u"]["thScr"] = 1
        gl["u"]["roof"] = 0
        gl["u"]["side"] = 0
        gl["u"]["lamp"] = 0
        gl["u"]["intLamp"] = 0
        gl["u"]["blScr"] = 0

    return gl
