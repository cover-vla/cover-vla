import csv
import glob
import os

# --- CONFIGURATION ---
CSV_FOLDER = os.path.join("scripts/eval", "data_csv")
DELTA_MAPPING = {
    "widowx_spoon_on_towel": "widowx_spoon_on_towel",
    "widowx_carrot_on_plate": "widowx_carrot_on_plate",
    "widowx_stack_cube": "widowx_stack_cube",
    "widowx_put_eggplant_in_basket": "widowx_put_eggplant_in_basket",

    'widowx_cube_on_plate_clean': 'widowx_carrot_on_plate',
    'widowx_small_plate_on_green_cube_clean': 'widowx_cube_on_plate_clean',
    'widowx_coke_can_on_plate_clean': 'widowx_carrot_on_plate',
    'widowx_pepsi_on_plate_clean': 'widowx_carrot_on_plate',
    'widowx_carrot_on_sponge_clean': 'widowx_carrot_on_plate',
    'widowx_eggplant_on_sponge_clean': 'widowx_put_eggplant_in_basket',
    'widowx_carrot_on_keyboard_clean': 'widowx_carrot_on_plate',
    'widowx_coke_can_on_keyboard_clean': 'widowx_coke_can_on_plate_clean',
    'widowx_spoon_on_towel_distract': "widowx_spoon_on_towel",
    'widowx_carrot_on_plate_distract': "widowx_carrot_on_plate",
    'widowx_carrot_on_keyboard_distract': 'widowx_carrot_on_keyboard_clean',
    'widowx_coke_can_on_plate_distract': 'widowx_coke_can_on_plate_clean',
    'widowx_coke_can_on_keyboard_distract': 'widowx_coke_can_on_keyboard_clean',

    'widowx_carrot_on_plate_lang_common': 'widowx_carrot_on_plate',
    'widowx_carrot_on_plate_lang_action': 'widowx_carrot_on_plate',
    'widowx_carrot_on_plate_lang_neg': 'widowx_carrot_on_plate',
    'widowx_carrot_on_plate_lang_neg_action': 'widowx_carrot_on_plate_distract',
    'widowx_carrot_on_plate_lang_common_distract': 'widowx_carrot_on_plate_lang_common',
    'widowx_spoon_on_towel_lang_action': 'widowx_spoon_on_towel',
    'widowx_spoon_on_towel_lang_common': 'widowx_spoon_on_towel',
    'widowx_spoon_on_towel_lang_common_distract': 'widowx_spoon_on_towel_lang_common',
    'widowx_stack_cube_lang_action': 'widowx_stack_cube',
    'widowx_eggplant_in_basket_lang_action': 'widowx_put_eggplant_in_basket',
    'widowx_eggplant_in_basket_lang_color': 'widowx_put_eggplant_in_basket',
    'widowx_eggplant_in_basket_lang_common': 'widowx_put_eggplant_in_basket',
    'widowx_carrot_on_keyboard_lang_common': 'widowx_carrot_on_keyboard_clean',
    'widowx_coke_can_on_plate_lang_common': 'widowx_coke_can_on_plate_clean',
    'widowx_coke_can_on_plate_lang_neg': 'widowx_coke_can_on_plate_clean',
    'widowx_coke_can_on_plate_lang_common_distract': 'widowx_coke_can_on_plate_lang_common',

    "widowx_orange_juice_on_plate_clean": "widowx_carrot_on_plate",
    "widowx_orange_juice_on_plate_distract": 'widowx_orange_juice_on_plate_clean',
    "widowx_orange_juice_on_plate_lang_neg": "widowx_orange_juice_on_plate_clean",
    "widowx_orange_juice_on_plate_lang_common": "widowx_orange_juice_on_plate_clean",
    "widowx_orange_juice_on_plate_lang_common_distract": "widowx_orange_juice_on_plate_lang_common",
    "widowx_orange_juice_on_plate_lang_common_distractv2": "widowx_orange_juice_on_plate_lang_common",
    "widowx_nut_on_plate_clean": "widowx_carrot_on_plate",
    "widowx_nut_on_plate_lang_common": "widowx_nut_on_plate_clean",
    "widowx_eggplant_on_keyboard_clean": "widowx_put_eggplant_in_basket",
    "widowx_carrot_on_ramekin_clean": "widowx_carrot_on_plate",
    "widowx_carrot_on_wheel_clean": "widowx_carrot_on_plate",
    "widowx_coke_can_on_ramekin_clean": "widowx_coke_can_on_plate_clean",
    "widowx_coke_can_on_wheel_clean": "widowx_coke_can_on_plate_clean",
    "widowx_nut_on_wheel_clean": "widowx_nut_on_plate_clean",
    "widowx_cube_on_plate_lang_shape": 'widowx_cube_on_plate_clean',
    "widowx_spoon_on_towel_lang_neg": "widowx_spoon_on_towel",
    "widowx_spoon_on_towel_lang_color": "widowx_spoon_on_towel",
    "widowx_carrot_on_plate_lang_color": "widowx_carrot_on_plate",
}
# DELTA_MAPPING = {
#     'widowx_carrot_on_plate': "widowx_spoon_on_towel",
# }

def process_file(inpath):
    print(f"Processing {inpath}")
    base, _ = os.path.splitext(inpath)
    outpath = base + "_delta.csv"

    # read the two-row header + data
    with open(inpath, newline="") as f:
        reader = csv.reader(f)
        header1 = next(reader)   # task names
        header2 = next(reader)   # metric names
        data    = [row for row in reader]

    tasks   = header1
    metrics = header2

    # find for each mapping the (metric, idx1, idx2) pairs
    mappings_pos = {}
    for t1, t2 in DELTA_MAPPING.items():
        lst = []
        for i, (tsk, met) in enumerate(zip(tasks, metrics)):
            if tsk == t1:
                # look for same metric index in t2
                for j, (tsk2, met2) in enumerate(zip(tasks, metrics)):
                    if tsk2 == t2 and met2 == met:
                        lst.append((met, i, j))
                        break
        if lst:
            mappings_pos[t1] = lst

    # flatten into a single lookup: idx1 -> (key, idx2, metric_name)
    delta_map = {}
    for key, lst in mappings_pos.items():
        for met0, i1, i2 in lst:
            delta_map[i1] = (key, i2, met0)

    # build new headers with Δ and Δ (%) inserted
    new_h1, new_h2 = [], []
    for i, (tsk, met) in enumerate(zip(tasks, metrics)):
        new_h1.append(tsk)
        new_h2.append(met)
        if i in delta_map:
            key, _, met0 = delta_map[i]
            new_h1 += [key, key]
            new_h2 += ["Delta", "Delta(%)"]

    # build new data rows
    new_data = []
    for row in data:
        new_row = []
        for i, cell in enumerate(row):
            new_row.append(cell)
            if i in delta_map:
                key, j, metric_name = delta_map[i]
                try:
                    v1 = float(row[i])
                    v2 = float(row[j])
                    # print(f"Processing {key} {metric_name} {v1} {v2}")
                    d  = v1 - v2
                    p  = (d / v2 * 100) if v2 != 0 else ""
                except ValueError:
                    d, p = "", ""
                new_row += [str(d), str(p)]
        new_data.append(new_row)

    # write out the new CSV
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(new_h1)
        writer.writerow(new_h2)
        writer.writerows(new_data)

    print(f"Wrote {outpath}")

def main():
    pattern = os.path.join(CSV_FOLDER, "*.csv")
    for path in glob.glob(pattern):
        if "_delta" in path:
            print(f"Skipping {path}")
            continue
        process_file(path)

if __name__ == "__main__":
    main()
