"""
Main evaluation agent. Using torch.compile and bfloat16 by default.
"""
import collections
import os
import sys
import time
from pathlib import Path

import imageio
import numpy as np
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from typing_extensions import override

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.agent.configuration_pipeline import TrainPipelineConfig
from src.experiments.env_adapters.language_mapper import PersistentLanguageMapper
from src.experiments.envs.base_evaluator import BaseEvaluator
from src.utils.monitor import setup_logger

os.environ["WANDB__SERVICE_WAIT"] = "300"


class SimplerEvaluator(BaseEvaluator):
    def __init__(self,
                 pipeline_cfg: TrainPipelineConfig):
        '''
        Initializes the evaluator with configuration objects for evaluating a trained model.

            pipeline_cfg: TrainPipelineConfig, a dataclass containing all configurations
                required for the training pipeline, including evaluation and model settings.
        '''
        super().__init__(pipeline_cfg)

        self.language_logic_chain = pipeline_cfg.eval_cfg.language_logic_chain

        if self.language_logic_chain:
            language_mapping_candidates = {
                "carrot": ["the yellow vegetable", "the veggie", "the yellow thing that rabbit likes", "the veggie that rabbit likes"],
                "eggplant": ["the purple vegetable", "the veggie", "the thing that looks like a purple balloon"],
                "spoon": ["the silver spoon", "the thing that people use to eat soup", "the shiny thing"],
                "cube": ["the thing that looks like a box", "lego"],
            }
            self.language_mapper = PersistentLanguageMapper(mapping_candidates=language_mapping_candidates, seed=self.seed)

    @override
    def evaluate(self):
        '''Run evaluation on all tasks in the task list'''
        for gradient_step in self.gradient_steps:
            if self.no_gradient_steps:
                model_path = Path(self.eval_cfg.pretrained_model_path)
            else:
                model_path = Path(self.eval_cfg.pretrained_model_path) / f"step_{gradient_step!s}"

            self._initialze_model_client(model_path=str(model_path), gradient_step=gradient_step)

            for task_name in self.task_lists:
                if not self.debug:
                    self._update_n_eval_episode(task_name) # some tasks have different number of possible episodes
                self.evaluate_task(task_name)

            if self.use_wandb:
                wandb.log(self.wandb_metrics, step=int(gradient_step), commit=True)

    @override
    def evaluate_task(self, task_name):
        '''
        Evaluate a single task

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

        # Set up environment
        env = simpler_env.make(task_name)

        cnt_episode = 0
        metrics = {'Src Intention Correct': [],
                'Move Correct':[],
                'Wrong Obj Attempt':[],
                'Grasp Correct':[],
                'Success Rate':[]}

        # Set up episodes TODO: This seems like it's repeated in the loop. See if we can remove it
        env_reset_options = {}
        env_reset_options["obj_init_options"] = {
            "episode_id": cnt_episode,
        }

        obs, reset_info = env.reset(seed=self.seed, options=env_reset_options)
        instruction = env.get_language_instruction()
        if self.language_logic_chain:
            instruction = self._preprocess_task_instruction(instruction)

        # Set up video recording
        recording = self.n_video > 0 if self.pipeline_cfg.eval_cfg.recording else False
        if recording:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            video_default_path = video_dir / f"video_{cnt_episode}.mp4"
            video_writer = imageio.get_writer(video_default_path)

        task_logger.info(
            f"Reset info: {reset_info} Instruction: {instruction} Max episode length: {env.spec.max_episode_steps}"
        )

        # Set up receding horizon control
        action_plan = collections.deque()
        while True:

            img = np.ascontiguousarray(get_image_from_maniskill2_obs_dict(env, obs))


            if not action_plan:
                # action horizon is all executed
                # Query model to get action
                element = {
                           "observation.images.top": img,
                           "observation.state": obs,
                           "task": str(instruction)
                          }
                action_chunk = self.client.infer(element)
                action_plan.extend(action_chunk[: self.action_step])

            action = action_plan.popleft()
            obs, reward, success, truncated, info = env.step(action.copy()) # somehow simpler needs a writable np array

            # Record video frame if enabled
            if recording:
                video_writer.append_data(img)

            # Update instruction if changed. Currently no need since we only evaluate short tasks
            # new_instruction = env.get_language_instruction()
            # if new_instruction != instruction:
            #     instruction = new_instruction

            # Episode end handling
            if truncated:
                episode_stats = info.get('episode_stats', {})
                self._process_episode_stats(
                    metric=metrics,
                    episode_stats=episode_stats,
                    success=success
                )
                self.client.reset()  # Reset the client for the next episode
                if recording:
                    video_writer.close()
                    if success:
                        os.rename(
                            video_default_path,
                            video_dir / f"video_{cnt_episode}_success.mp4",
                        )

                cnt_episode += 1

                episode_stats = info.get('episode_stats', {})
                task_logger.info(f"Episode {cnt_episode} stats: {episode_stats}")
                # Exit if we've completed enough episodes
                if cnt_episode >= self.n_eval_episode:
                    break

                # Reset for next episode
                if self.language_logic_chain:
                    self.language_mapper.reset()
                action_plan.clear()

                env_reset_options["obj_init_options"] = {
                    "episode_id": cnt_episode,
                }
                obs, reset_info = env.reset(options=env_reset_options)
                instruction = env.get_language_instruction()
                if self.language_logic_chain:
                    instruction = self._preprocess_task_instruction(instruction)

                task_logger.info(
                    f"Reset info: {reset_info} Instruction: {instruction} Max episode length: {env.spec.max_episode_steps}"
                )

                recording = self.n_video > cnt_episode if self.pipeline_cfg.eval_cfg.recording else False
                if recording:
                    video_default_path = video_dir / f"video_{cnt_episode}.mp4"
                    video_writer = imageio.get_writer(video_default_path)

        # Calculate and log results
        aggregated_metrics = self._aggregate_metrics(metrics)
        task_eval_time = time.time() - start_task_time

        # log results
        self._log_summary(logger=task_logger,
                          cnt_episode=cnt_episode,
                          eval_time=task_eval_time,
                          metrics=aggregated_metrics)

        self._log_summary(logger=self.main_logger,
                          cnt_episode=cnt_episode,
                          eval_time=task_eval_time,
                          metrics=aggregated_metrics)

        if self.use_wandb:
            self.wandb_metrics[task_name] = aggregated_metrics['Success Rate']

    def _update_n_eval_episode(self, task_name):
        if "google_robot" in task_name:
            if 'coke' in task_name:
                self.n_eval_episode = 25 * 4 # 25 locations, 4 urdfs, 10 trials each
            elif 'move' in task_name:
                self.n_eval_episode = 60 * 4 # 60 locations, 4 urdfs, 10 trials each
            elif 'drawer' in task_name:
                self.n_eval_episode = 3 * 4 * 9 # 3 drawers, 4 urdfs, 9 locations/rgb_overlay_paths, 10 trials each
            elif 'apple' in task_name:
                self.n_eval_episode = 9 * 4 * 3 # 9 apple locations, 4 urdfs, 3 robot locations/rgb_overlay_paths, 10 trials each
            self.n_video = self.n_eval_episode

    def _process_episode_stats(self, metric, episode_stats, success):
        '''
        Process episode stats to extract relevant information
        '''
        # Extract relevant information from episode_stats
        metric['Success Rate'].append(success)
        metric['Move Correct'].append(episode_stats.get('moved_correct_obj', 0))
        metric['Wrong Obj Attempt'].append(episode_stats.get('moved_wrong_obj', 0))
        metric['Grasp Correct'].append(episode_stats.get('is_src_obj_grasped', 0))
        metric['Src Intention Correct'].append(episode_stats.get('source_intention', 0))

    def _aggregate_metrics(self, metrics):
        '''
        Aggregate metrics across all episodes
        '''
        aggregated_metrics = {}
        for key in metrics.keys():
            aggregated_metrics[key] = np.mean(metrics[key])
        return aggregated_metrics
