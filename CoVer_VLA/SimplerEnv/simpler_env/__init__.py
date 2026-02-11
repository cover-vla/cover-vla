import gymnasium as gym
import mani_skill2_real2sim.envs
import warnings

ENVIRONMENTS = [
    "google_robot_pick_coke_can",
    "google_robot_pick_horizontal_coke_can",
    "google_robot_pick_vertical_coke_can",
    "google_robot_pick_standing_coke_can",
    "google_robot_pick_object",
    "google_robot_move_near_v0",
    "google_robot_move_near_v1",
    "google_robot_move_near",
    "google_robot_open_drawer",
    "google_robot_open_top_drawer",
    "google_robot_open_middle_drawer",
    "google_robot_open_bottom_drawer",
    "google_robot_close_drawer",
    "google_robot_close_top_drawer",
    "google_robot_close_middle_drawer",
    "google_robot_close_bottom_drawer",
    "google_robot_place_in_closed_drawer",
    "google_robot_place_in_closed_top_drawer",
    "google_robot_place_in_closed_middle_drawer",
    "google_robot_place_in_closed_bottom_drawer",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
    # =================================== Following are the Custom Environments =========================================
    "widowx_carrot_on_plate_unseen_lighting",
    "widowx_spoon_on_towel_new_table_cloth",
    "widowx_spoon_on_towel_google",
    "widowx_redbull_on_plate",
    "widowx_tennis_ball_in_basket",
    "widowx_zucchini_on_towel",
    "widowx_tape_measure_in_basket",
    "widowx_toy_dinosaur_on_towel",
    "widowx_stapler_on_paper",
]

ENVIRONMENT_MAP = {
    "google_robot_pick_coke_can": ("GraspSingleOpenedCokeCanInScene-v0", {}),
    "google_robot_pick_horizontal_coke_can": (
        "GraspSingleOpenedCokeCanInScene-v0",
        {"lr_switch": True},
    ),
    "google_robot_pick_vertical_coke_can": (
        "GraspSingleOpenedCokeCanInScene-v0",
        {"laid_vertically": True},
    ),
    "google_robot_pick_standing_coke_can": (
        "GraspSingleOpenedCokeCanInScene-v0",
        {"upright": True},
    ),
    "google_robot_pick_object": ("GraspSingleRandomObjectInScene-v0", {}),
    "google_robot_move_near": ("MoveNearGoogleBakedTexInScene-v1", {}),
    "google_robot_move_near_v0": ("MoveNearGoogleBakedTexInScene-v0", {}),
    "google_robot_move_near_v1": ("MoveNearGoogleBakedTexInScene-v1", {}),
    "google_robot_open_drawer": ("OpenDrawerCustomInScene-v0", {}),
    "google_robot_open_top_drawer": ("OpenTopDrawerCustomInScene-v0", {}),
    "google_robot_open_middle_drawer": ("OpenMiddleDrawerCustomInScene-v0", {}),
    "google_robot_open_bottom_drawer": ("OpenBottomDrawerCustomInScene-v0", {}),
    "google_robot_close_drawer": ("CloseDrawerCustomInScene-v0", {}),
    "google_robot_close_top_drawer": ("CloseTopDrawerCustomInScene-v0", {}),
    "google_robot_close_middle_drawer": ("CloseMiddleDrawerCustomInScene-v0", {}),
    "google_robot_close_bottom_drawer": ("CloseBottomDrawerCustomInScene-v0", {}),
    "google_robot_place_in_closed_drawer": ("PlaceIntoClosedDrawerCustomInScene-v0", {}),
    "google_robot_place_in_closed_top_drawer": ("PlaceIntoClosedTopDrawerCustomInScene-v0", {}),
    "google_robot_place_in_closed_middle_drawer": ("PlaceIntoClosedMiddleDrawerCustomInScene-v0", {}),
    "google_robot_place_in_closed_bottom_drawer": ("PlaceIntoClosedBottomDrawerCustomInScene-v0", {}),
    "google_robot_place_apple_in_closed_top_drawer": (
        "PlaceIntoClosedTopDrawerCustomInScene-v0", 
        {"model_ids": "baked_apple_v2"}
    ),
    "widowx_spoon_on_towel": ("PutSpoonOnTableClothInScene-v0", {}),
    "widowx_carrot_on_plate": ("PutCarrotOnPlateInScene-v0", {}),
    "widowx_stack_cube": ("StackGreenCubeOnYellowCubeBakedTexInScene-v0", {}),
    "widowx_put_eggplant_in_basket": ("PutEggplantInBasketScene-v0", {}),
    # =================================== Following are the Custom Environments =========================================
    "widowx_carrot_on_plate_unseen_lighting": ("PutCarrotOnPlateUnseenLighting", {}),
    "widowx_spoon_on_towel_new_table_cloth": ("PutSpoonOnTableClothInSceneNewTableCloth", {}),
    "widowx_spoon_on_towel_google": ("PutSpoonOnTableClothInSceneGoogle", {}),
    "widowx_redbull_on_plate": ("PutRedbullOnPlateInScene", {}),
    "widowx_tennis_ball_in_basket": ("PutTennisBallInBasketScene", {}),
    "widowx_zucchini_on_towel": ("PutZucchiniOnTableClothInScene", {}),
    "widowx_tape_measure_in_basket": ("PutTapeMeasureInBasketScene-v0", {}),
    "widowx_toy_dinosaur_on_towel": ("PutToyDinosaurOnTowelInScene", {}),
    "widowx_stapler_on_paper": ("PutStaplerOnPaperInScene", {}),
}


def make(task_name, **kwargs):
    """Creates simulated eval environment from task name."""
    assert task_name in ENVIRONMENTS, f"Task {task_name} is not supported. Environments: \n {ENVIRONMENTS}"
    env_name, env_kwargs = ENVIRONMENT_MAP[task_name]
    
    env_kwargs["obs_mode"] = "rgbd",
    env_kwargs["prepackaged_config"] = True

    for key, value in kwargs.items():
        if key in env_kwargs:
            warnings.warn(f"default value [{env_kwargs[key]}] for Key {key} changes to value [{value}]")
        env_kwargs[key] = value

    env = gym.make(env_name, **env_kwargs)
    return env
