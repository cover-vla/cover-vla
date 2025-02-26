import glob
import json

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# list_of_transforms = ['no_transform', 'synonym', 'antonym', 'negation', 'verb_noun_shuffle', 'in_set', 'out_set']
list_of_transforms = ['no_transform', 'random_shuffle', 'synonym', 'antonym', 'verb_noun_shuffle', 'negation', 'out_set']
list_of_task_suite_names = ['libero_spatial', 'libero_object']
for task_suite_name in list_of_task_suite_names:
    print(f'Task suite: {task_suite_name}')
    for transform in list_of_transforms:
        json_path = glob.glob(f"./examples/libero/rollouts/pi0/{task_suite_name}/{transform}/*/*.json")
        print(f'Transform: {transform}')
        num_rollouts = len(json_path)
        transform_success_count = 0
        for json_file in json_path:
            data = load_json(json_file)
            success = data['success']
            if success == 'success':
                transform_success_count += 1
        if num_rollouts > 0:
            success_rate = transform_success_count / num_rollouts
            print(f"Success rate for {transform}: {success_rate}")
        else:
            print(f"No rollouts found for {transform}")
        print('--------------------------------')
    print('************************************************')
    
    


