# progrockdiffusion
A command line version of Disco Diffusion

# Setup
Make sure you're in the directory you plan to run it from.
```
git clone https://github.com/crowsonkb/guided-diffusion
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

For now, edit dd.py and find the parameters you want to change.

The main ones (and at what line of code) are:
```
batch_name on 1701
steps on 1702
width_height on 1703
sharpen_preset on line 2004 NEEDS TO BE "OFF" (for now)
text_prompts on line 2050 or so
n_batches on line 2066 or so
```

Then simply run it:
```
python3 dd.py
```
# Notes

- Currently Superres Sampling doesn't work, it will crash.

# TODO

- Get all the main parameters from either a config file or command line (command line seems like too much)
- The code already saves a settings.txt file in the output directory with each run, based on the internal parameters, so perhaps there's a way to read that back in and use its values? At the very least, a similar file in the root directory.
- The SLIP models are currently failing due to a variable not being defined.
