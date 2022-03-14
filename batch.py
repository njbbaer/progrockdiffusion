import json
import os
import shutil
import cv2
import glob

TEMP_DIR = 'images_out/temp'

BASE_SETTINGS = {
    "width": 1024,
    "height": 1536,
    "ViTL14": False,
    "steps": 150,
    "skip_steps_ratio": 0.5,
}

GENERATION_SETTINGS = {
    **BASE_SETTINGS,
    "use_secondary_model": False,
    "width": int(BASE_SETTINGS["width"] / 2),
    "height": int(BASE_SETTINGS["height"] / 2),
    "init_image": "",
    "skip_steps": 0,
    "clamp_max": 0.02,
}

FINISHING_SETTINGS = {
    **BASE_SETTINGS,
    "use_secondary_model": True,
    "init_image": TEMP_DIR + "/upscaled.png",
    "skip_steps": int(BASE_SETTINGS["steps"] * BASE_SETTINGS["skip_steps_ratio"]),
}


def apply_settings(new_settings):
    with open('settings.json', 'r') as file:
        current_settings = json.load(file)

    with open('settings_temp.json', 'w') as file:
        file.write(json.dumps({
            **current_settings,
            **new_settings,
        }))


def upscale(filepath):
    img = cv2.imread(filepath, 1)
    scaled_img = cv2.resize(img, (0, 0), fx=2, fy=2)
    cv2.imwrite(TEMP_DIR + '/upscaled.png', scaled_img)


def generate_image():
    apply_settings(GENERATION_SETTINGS)
    os.system("python prd.py -s settings_temp.json -o temp")


def finish_image(filepath=None):
    if not filepath:
        filepath = sorted(glob.glob(TEMP_DIR + '/temp*.png'), key=os.path.getctime)[-1]

    upscale(filepath)
    apply_settings(FINISHING_SETTINGS)
    os.system("python prd.py -s settings_temp.json")


def finish_all():
    for filepath in glob.glob(TEMP_DIR + '/temp*.png'):
        finish_image(filepath)


if __name__ == '__main__':
    for _ in range(10):
        generate_image()
        # finish_image()
