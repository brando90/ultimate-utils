# Run Wandb Sweep in python but use a yaml config file
Logical flow for running a sweep:
1. Define the sweep configuration in a YAML file and load it in Python as a dict.
2. Initialize the sweep in Python which create it on your project/eneity in wandb platform and get the sweep_id.
3. Finally, once the sweep_id is acquired, execute the sweep using the desired number of agents in python.

refs:
    - youtube video: https://www.youtube.com/watch?v=9zrmUIlScdY
    - https://chat.openai.com/share/fbf98147-3987-4d75-b7c5-52b67a1048a6com/watch?v=9zrmUIlScdY