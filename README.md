# progrockdiffusion
A command line version of Disco Diffusion

# Setup
Make sure you're in the direction you plan to run it from.
```
git clone https://github.com/crowsonkb/guided-diffusion
git clone https://github.com/assafshocher/ResizeRight.git
pip install -e ./CLIP
pip install -e ./guided-diffusion
pip install lpips datetime timm
apt install imagemagick
```

##SuperRes setup (currently crashing if enabled in dd.py)
```
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
text_prompts on line 2050 or so
n_batches on line 2066 or so
```
# Notes

- Currently Superres Sampling doesn't work, it will crash.
- Since you can't see your progress (short of opening progress.png manually), you might as well set "display_rate" on line 2063 to the same number as steps on 1702

# TODO

- Get all the main parameters from either a config file or command line (command line seems like too much)
- The code already saves a settings.txt file based on the internal parameters, so perhaps there's a way to read that back in and use its values?
