import json
from lang_transform import LangTransform
import time
from tqdm import tqdm
libero_task_map = {
    "libero_spatial": [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ],
    "libero_object": [
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick_up_the_cream_cheese_and_place_it_in_the_basket",
        "pick_up_the_salad_dressing_and_place_it_in_the_basket",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        "pick_up_the_ketchup_and_place_it_in_the_basket",
        "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
        "pick_up_the_butter_and_place_it_in_the_basket",
        "pick_up_the_milk_and_place_it_in_the_basket",
        "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
        "pick_up_the_orange_juice_and_place_it_in_the_basket",
    ],
    "libero_goal": [
        "open_the_middle_drawer_of_the_cabinet",
        "put_the_bowl_on_the_stove",
        "put_the_wine_bottle_on_top_of_the_cabinet",
        "open_the_top_drawer_and_put_the_bowl_inside",
        "put_the_bowl_on_top_of_the_cabinet",
        "push_the_plate_to_the_front_of_the_stove",
        "put_the_cream_cheese_in_the_bowl",
        "turn_on_the_stove",
        "put_the_bowl_on_the_plate",
        "put_the_wine_bottle_on_the_rack",
    ],
    "libero_10": [
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    ],
    "libero_90": [
        "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it",
        "KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it",
        "KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it",
        "KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it",
        "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet",
        "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it",
        "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate",
        "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate",
        "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate",
        "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate",
        "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle",
        "KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl",
        "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove",
        "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove",
        "KITCHEN_SCENE3_turn_on_the_stove",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it",
        "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet",
        "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet",
        "KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet",
        "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack",
        "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE5_put_the_black_bowl_on_the_plate",
        "KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE6_close_the_microwave",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug",
        "KITCHEN_SCENE7_open_the_microwave",
        "KITCHEN_SCENE7_put_the_white_bowl_on_the_plate",
        "KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate",
        "KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove",
        "KITCHEN_SCENE8_turn_off_the_stove",
        "KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf",
        "KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet",
        "KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf",
        "KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE9_turn_on_the_stove",
        "KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it",
        "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray",
        "LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray",
        "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate",
        "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate",
        "LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate",
        "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate",
        "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
        "STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy",
        "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
        "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
        "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
        "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
        "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
        "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
        "STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy",
        "STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy",
        "STUDY_SCENE4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf",
        "STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf",
        "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf",
        "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf",
    ],
}

import re

list_of_transforms = ['synonym_noun', 'synonym_verb', 'antonym_verb',
                      'negation', 'verb_noun_shuffle', 'out_set', 'random_shuffle']

def remove_scene_prefix(task):
    match = re.match(r'^([A-Z0-9_]+_)([a-z].+)$', task)
    if match:
        return match.group(2)
    return task

negative_set = set()
lang_transform = LangTransform()

for task_suite_name in libero_task_map.keys():
    list_of_tasks = libero_task_map[task_suite_name]
    for task in list_of_tasks:
        # print('task', '\n', task)
        cleaned_task = ' '.join(remove_scene_prefix(task).split('_'))
        # print('cleaned task', '\n', cleaned_task.strip())
        # print('-------------')
        negative_set.add(cleaned_task)

print(len(negative_set))

save_dic = {task : {transform : [] for transform in list_of_transforms} for task in negative_set}

rate_counter = 0

start_time = time.time()
for task in negative_set:
    for transform in tqdm(list_of_transforms, desc='Transforming tasks'):
        for i in range(50):
            
            transformed_task = lang_transform.transform(task, transform)
            if transform in ['synonym_noun', 'synonym_verb', 'antonym_verb', 
                             'negation', 'verb_noun_shuffle', 'out_set']:
                rate_counter += 1
                curr_time = time.time()
                if rate_counter >= 9999:
                    elapsed_time = curr_time - start_time
                    if elapsed_time < 60:
                        sleep_time = 60 - elapsed_time + 10  # Add 10 second buffer
                        time.sleep(sleep_time)
                    rate_counter = 0
                    start_time = time.time()
            
            save_dic[task][transform].append(transformed_task)

for task in save_dic.keys():
    for transform in list_of_transforms:
        list_of_tasks = save_dic[task][transform]
        assert len(list_of_tasks) == 50

with open('./negative_set.json', 'w') as f:
    json.dump(save_dic, f)

with open('./negative_set.json', 'r') as f:
    loaded_dic = json.load(f)

print(len(loaded_dic))
print(len(loaded_dic['put the middle black bowl on top of the cabinet']['random_shuffle']))

