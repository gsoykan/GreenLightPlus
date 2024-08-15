from ray.rllib.algorithms.ppo import PPOConfig  # Import PPO algorithm configuration
from ray.tune.logger import pretty_print  # For beautifying output
from tqdm import tqdm  # For displaying progress bar
import os
import random

from core.greenhouse_env import GreenhouseEnv

# Configure PPO algorithm
config = PPOConfig()
config.rollouts(num_rollout_workers=10)  # Set 10 parallel rollout workers
config.resources(num_cpus_per_worker=1)  # Each worker uses 1 CPU

# Configure environment parameters
config.environment(
    env=GreenhouseEnv,  # Use GreenhouseEnv as the environment
    env_config={
        "first_day": 101,  # Start date of simulation (day of the year)
        "epw_path": "../data/NLD_Amsterdam.062400_IWEC.epw",  # Path to weather data file
        "isMature": False,  # Whether the crop is mature
        "season_length": 60,  # Length of growing season (days)
        "season_interval": 1 / 24 * 4,  # Simulation time interval (hours)
        "current_step": 0,  # Current step
        "target_yield": 8,  # Target yield
        "target_yield_unit_energy_input": 22,  # Target yield per unit energy input
        "init_state": {
            "p": {
                # Greenhouse structure settings
                'psi': 22,  # Mean greenhouse cover slope [degrees]
                'aFlr': 4e4,  # Floor area [m^2]
                'aCov': 4.84e4,  # Surface of the cover including side walls [m^2]
                'hAir': 6.3,  # Height of the main compartment [m]
                'hGh': 6.905,  # Mean height of the greenhouse [m]
                'aRoof': 0.1169 * 4e4,  # Maximum roof ventilation area [m^2]
                'hVent': 1.3,  # Vertical dimension of single ventilation opening [m]
                'cDgh': 0.75,  # Ventilation discharge coefficient [-]
                'lPipe': 1.25,  # Length of pipe rail system [m m^-2]
                'phiExtCo2': 7.2e4 * 4e4 / 1.4e4,  # Capacity of CO2 injection for the entire greenhouse [mg s^-1]
                'pBoil': 300 * 4e4,  # Capacity of boiler for the entire greenhouse [W]

                # Control settings
                'co2SpDay': 1000,  # CO2 setpoint during the light period [ppm]
                'tSpNight': 18.5,  # temperature set point dark period [°C]
                'tSpDay': 19.5,  # temperature set point light period [°C]
                'rhMax': 87,  # maximum relative humidity [%]
                'ventHeatPband': 4,  # P-band for ventilation due to high temperature [°C]
                'ventRhPband': 50,  # P-band for ventilation due to high RH [% humidity]
                'thScrRhPband': 10,  # P-band for screen opening due to high RH [% humidity]
                'lampsOn': 0,  # time of day to switch on lamps [h]
                'lampsOff': 18,  # time of day to switch off lamps [h]
                'lampsOffSun': 400,  # lamps off if radiation above this value [W m^-2]
                'lampRadSumLimit': 10  # Daily radiation sum limit for lamp use [MJ m^-2 day^-1]
            }
        }
    },
    render_env=False,  # Do not render the environment
)

# Configure training parameters
config.training(
    gamma=0.9,  # Discount factor
    lr=0.0001,  # Learning rate
    kl_coeff=0.3,  # KL divergence coefficient
    model={
        "fcnet_hiddens": [256, 256],  # Number of hidden units in fully connected layers
        "fcnet_activation": "relu",  # Activation function
        "use_lstm": True,  # Use LSTM
        "max_seq_len": 48,  # Maximum sequence length for LSTM
    }
)

if __name__ == '__main__':
    # Build algorithm
    algo = config.build()

    # Train the model
    for episode in tqdm(range(250)):  # Train for 250 episodes
        # Train the algorithm
        result = algo.train()

        # Print results and save checkpoint every 5 episodes
        if episode % 5 == 0:
            checkpoint_dir = algo.save().checkpoint.path
            print(f"Checkpoint saved in directory {checkpoint_dir}")
            print(f"Episode {episode}: Mean Reward = {result['episode_reward_mean']}, "
                  f"Mean Length = {result['episode_len_mean']}")

    # Evaluate the model after training
    print("\nEvaluating trained model:")
    env = GreenhouseEnv(config.env_config)  # Create a new environment instance
    state = env.reset()  # Reset the environment
    done = False
    total_reward = 0
    while not done:
        action = algo.compute_single_action(state)  # Use trained model to select action
        state, reward, done, info = env.step(action)  # Execute action
        total_reward += reward  # Accumulate reward

    print(f"Evaluation complete. Total reward: {total_reward}")

    # Clean up resources
    algo.stop()
