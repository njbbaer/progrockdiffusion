# progrockdiffusion
A command line version of Disco Diffusion

# Setup
Make sure you're in the directory you plan to run it from.
```
git clone https://github.com/crowsonkb/guided-diffusion
git clone https://github.com/openai/CLIP.git
git clone https://github.com/assafshocher/ResizeRight.git
pip install -e ./CLIP
pip install -e ./guided-diffusion
pip install lpips datetime timm
apt install imagemagick
git clone https://github.com/CompVis/latent-diffusion.git
git clone https://github.com/CompVis/taming-transformers
pip install -e ./taming-transformers
pip install ipywidgets omegaconf>=2.0.0 pytorch-lightning>=1.0.8 torch-fidelity einops wandb
```

# Use

```
usage: python3 dd.py [-h] [-s SETTINGS] [-o OUTPUT] [-p PROMPT]

Generate images from text prompts.
By default, the supplied settings.json file will be used.
You can edit that, and/or use the options below to fine tune:

Optional arguments:
  -h, --help            show this help message and exit
  -s SETTINGS, --settings SETTINGS
                        A settings JSON file to use, best to put in quotes
  -o OUTPUT, --output OUTPUT
                        What output directory to use within images_out
  -p PROMPT, --prompt PROMPT
                        Override the prompt

Usage examples:

To use the Default output directory and settings from settings.json:
 python3 dd.py

To use your own settings.json file (note that putting it in quotes can help parse errors):
 python3 dd.py -s "some_directory/mysettings.json"

To quickly just override the output directory name and the prompt:
 python3 dd.py -p "A cool image of the author of this program" -o Coolguy
```
# Notes

- Currently Superres Sampling doesn't work, it will crash.
- When using multiple prompts only the last prompt is used.

# TODO

- The SLIP models are currently failing due to a variable not being defined.
- Provide a section in this readme that goes over all the settings in settings.json and what they do, since we don't have the colab notebook to show those hints
