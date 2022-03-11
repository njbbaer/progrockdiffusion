import json
import os
import shutil
import cv2

BASE_SETTINGS = {
    # High Quality
    "width": 1536,
    "height": 1024,
    "ViTL14": True,
    "steps": 250,

    # # Low Quality
    # "width": 1536,
    # "height": 1024,
    # "ViTL14": False,
    # "steps": 100,

    "skip_steps_multiple": 0.4,
}

PRIMARY_RUN_SETTINGS = {
    **BASE_SETTINGS,
    "use_secondary_model": False,
    "width": int(BASE_SETTINGS["width"] / 2),
    "height": int(BASE_SETTINGS["height"] / 2),
    "init_image": "",
    "skip_steps": 0,
    "clamp_max": 0.03,
}

SECONDARY_RUN_SETTINGS = {
    **BASE_SETTINGS,
    "use_secondary_model": True,
    "init_image": "images_out/temp/temp(0)_0.png",
    "skip_steps": int(BASE_SETTINGS["steps"] * BASE_SETTINGS["skip_steps_multiple"]),
}


def read_settings():
    with open('settings.json', 'r') as file:
        return json.load(file)


def write_settings(settings):
    with open('settings_temp.json', 'w') as file:
        file.write(json.dumps(settings))


def apply_settings(new_settings):
    write_settings({
        **read_settings(),
        **new_settings,
    })


def upscale(filepath):
    img = cv2.imread(filepath, 1)
    scaled_img = cv2.resize(img, (0, 0), fx=2, fy=2)
    cv2.imwrite(filepath, scaled_img)


def clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def do_primary_run(custom_settings={}):
    apply_settings({**PRIMARY_RUN_SETTINGS, **custom_settings})
    os.system("python prd.py -s settings_temp.json -o temp")


def do_secondary_run(custom_settings={}):
    apply_settings({**SECONDARY_RUN_SETTINGS, **custom_settings})
    print(SECONDARY_RUN_SETTINGS)
    os.system("python prd.py -s settings_temp.json")


if __name__ == '__main__':
    while True:
        clear_dir('images_out/temp')
        do_primary_run()
        upscale('images_out/temp/temp(0)_0.png')
        do_secondary_run()
