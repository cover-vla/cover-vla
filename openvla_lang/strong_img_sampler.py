from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import os
import numpy as np


from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image, get_libero_dummy_action
from experiments.robot.robot_utils import set_seed_everywhere

set_seed_everywhere(7)

benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_spatial" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()
print("task_suite", task_suite)
num_tasks_in_suite = task_suite.n_tasks
resize_size = 224

num_trials_per_task = 50

task_id = 0
task = task_suite.get_task(task_id) # need task id to get the specific language 
initial_states = task_suite.get_task_init_states(task_id)
env, task_description = get_libero_env(task, 'openvla', resolution=256)
print("task_description", task_description)
env.reset()
chosen_initial_state_id = np.random.randint(0, len(initial_states))
print("chosen_initial_state_id", chosen_initial_state_id)
print('len(initial_states)', len(initial_states))
obs = env.set_init_state(initial_states[chosen_initial_state_id])
t = 0

num_steps_wait = 10
max_steps = 220

while t < max_steps + num_steps_wait:
    if t < num_steps_wait:
        obs, reward, done, info = env.step(get_libero_dummy_action('openvla'))
        t += 1
        continue
    img = get_libero_image(obs, resize_size)
    # print("img", img.shape)
    t += 1
env.close()
