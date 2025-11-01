"""
Evaluator/Client for the LIBERO environment.

The model will be served on a websocket server, and the client will connect to it.
The client will send observations to the server, and the server will return actions.
The client will also log the results of the evaluation to a file. The server will do some auxiliary logging to stdout.
"""
import collections
import os
import sys
import time
import traceback
from pathlib import Path

import draccus
import imageio
import numpy as np
from policy_server_client import image_tools
from typing_extensions import override

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))) # experiments
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))) # src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))) # project_folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "../third_party/LIBERO"))) # LIBERO

from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from src.agent.configuration_pipeline import TrainPipelineConfig
from src.experiments.envs.base_evaluator import BaseEvaluator
from src.utils.monitor import setup_logger

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

class LiberoEvaluator(BaseEvaluator):
    def __init__(self, pipeline_cfg: TrainPipelineConfig):
        '''
        Initializes the evaluator with configuration objects for evaluating a trained model.

            pipeline_cfg: TrainPipelineConfig, a dataclass containing all configurations
                required for the training pipeline, including evaluation and model settings.
        '''
        super().__init__(pipeline_cfg=pipeline_cfg)

    @override
    def evaluate(self):
        '''Run evaluation on all tasks in the task list'''

        for gradient_step in self.gradient_steps:
            self._initialze_model_client(gradient_step)
            for task_name in self.task_lists:
                self.evaluate_one(task_name)

            if self.use_wandb:
                wandb.log(self.wandb_metrics, step=int(gradient_step), commit=True)

    @override
    def evaluate_task(self, task_name):
        '''
        Evaluate a single task/suite

        Args:
            task_name: Name of the task to evaluate

        Returns:
            success_rate: The success rate achieved on this task
        '''
        start_task_time = time.time()

        # Initialize task-specific logging
        task_log_dir = self.log_dir / task_name
        video_dir = task_log_dir / "videos"
        if self.main_rank:
            os.makedirs(video_dir, exist_ok=True)

        task_logger = setup_logger(
            main_rank=self.main_rank,
            filename=task_log_dir / f"{task_name}.log" if not self.debug else None,  # log to console when debug is True
            debug=self.debug,
            name=f'{task_name}_logger'
        )

        task_logger.info(f"Task suite: {task_name}")
        self.main_logger.info(f"Task suite: {task_name}")

        # Setup environment
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_name]()
        num_tasks_in_suite = task_suite.n_tasks

        # Get max episode steps. varis from task to task
        max_steps = self._get_max_episode_steps(task_name)

        # self.main_logger.info(f"Env Adapter: {type(env_adapter).__name__}")

        total_episodes, total_successes = 0, 0
        for task_id in range(num_tasks_in_suite):
            # Get task
            task = task_suite.get_task(task_id)

            # Get default LIBERO initial states
            initial_states = task_suite.get_task_init_states(task_id)

            # Initialize LIBERO environment and task description
            env, instruction = self._get_libero_env(task, LIBERO_ENV_RESOLUTION, self.seed)

            task_episodes, task_successes = 0, 0
            for episode_idx in range(self.n_eval_episode):
                task_logger.info(
                    f"Task ID: {task_id}, Episode: {episode_idx}, Task description: {instruction}"
                )
                env.reset()
                action_plan = collections.deque()

                # Set initial states
                obs = env.set_init_state(initial_states[episode_idx])

                # Setup
                t = 0
                replay_images = []

                while t < max_steps + 10: # 10 extra steps to account for the dummy action
                    try:
                        # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                        # and we need to wait for them to fall
                        if t < 10:
                            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue

                        # Get preprocessed image. Rotate it 180 because for some reason libero gives upside down images...?
                        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])

                        img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(img, self.resize_size[0], self.resize_size[1])
                        )

                        # Save preprocessed image for replay video
                        replay_images.append(img)

                        if not action_plan:
                            # Finished executing previous action chunk -- compute new chunk
                            # Prepare observations dict
                            element = {
                                "observation.images.top": img,
                                "observation.state": np.concatenate(
                                    (
                                        obs["robot0_eef_pos"],
                                        obs["robot0_eef_quat"],
                                        obs["robot0_gripper_qpos"],
                                    )
                                ),
                                "task": str(instruction),
                            }

                            # Query model to get action
                            action_chunk = self.client.infer(element)
                            action_plan.extend(action_chunk[: self.action_step])

                        action = action_plan.popleft()

                        # Execute action in environment
                        obs, reward, done, info = env.step(action.tolist())

                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1

                    except Exception as e:
                        task_logger.error(f"Caught exception: {e}")
                        tb = traceback.extract_tb(sys.exc_info()[2])
                        for filename, lineno, func, text in tb:
                            task_logger.error(f"Exception occurred in {filename}, line {lineno}, in {func}")
                            task_logger.error(f"  Code: {text}")
                        break

                task_episodes += 1
                total_episodes += 1

                # Save a replay video of the episode
                suffix = "success" if done else "failure"
                task_segment = instruction.replace(" ", "_")
                imageio.mimwrite(
                    video_dir / f"rollout_{task_segment}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

                task_logger.info(f"Success: {done}")
                task_logger.info(f"# episodes completed so far: {total_episodes}")
                task_logger.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

            # Log final task results
            task_logger.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
            task_logger.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

        # log results
        task_eval_time = time.time() - start_task_time
        success_rate = float(total_successes) / float(total_episodes)

        self._log_summary(logger=task_logger,
                          cnt_episode=num_tasks_in_suite*self.n_eval_episode,
                          eval_time=task_eval_time,
                          success_rate=success_rate)

        self._log_summary(logger=self.main_logger,
                          cnt_episode=num_tasks_in_suite*self.n_eval_episode,
                          eval_time=task_eval_time,
                          success_rate=success_rate)

        if self.use_wandb:
            self.wandb_metrics[task_name] = success_rate


    def _get_libero_env(self, task, resolution, seed):
        """Initializes and returns the LIBERO environment, along with the task description."""
        task_description = task.language
        task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
        env = OffScreenRenderEnv(**env_args)
        env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
        return env, task_description

    def _get_max_episode_steps(self, task_name):
        """Returns the maximum number of steps for a given task."""
        if task_name == "libero_spatial":
            return 220  # longest training demo has 193 steps
        elif task_name == "libero_object":
            return 280  # longest training demo has 254 steps
        elif task_name == "libero_goal":
            return 300  # longest training demo has 270 steps
        elif task_name == "libero_10":
            return 520  # longest training demo has 505 steps
        elif task_name == "libero_90":
            return 400  # longest training demo has 373 steps
        else:
            raise ValueError(f"Unknown task name: {task_name}")

@draccus.wrap()
def main(pipeline_cfg: TrainPipelineConfig):
    """
    Main function to run the evaluator.
    This function is called by the draccus pipeline.
    """

    evaluator = LiberoEvaluator(pipeline_cfg=pipeline_cfg)
    evaluator.evaluate()

if __name__ == "__main__":
    main()
