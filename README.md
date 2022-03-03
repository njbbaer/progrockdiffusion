# progrockdiffusion
A command line version of [Disco Diffusion](https://github.com/alembics/disco-diffusion).

# Hardware prerequisites
An Nvidia GPU capable of running CUDA-based softare. 8gb is probably the minimum amount of GPU memory.

This author has an RTX 3080 with 10gb and it runs fairly well, but some advanced features are not possible with "only" 10gb.

# Software prerequisties
Ubuntu 20.04 (A docker environment, VM, or Windows Subsystem for Linux should work provided it can access your GPU).
Note that Windows Subsystem for Linux (WSL) has only been successful on Windows 11 using WSL2, due to Nvidia driver integration.

CUDA 11.4 (installation instructions can be found here: https://developer.nvidia.com/cuda-11-4-1-download-archive). Note that this seems to be working out of the box on WSL2 for Windows 11.

You can test that your environment is working properly by running:

```
nvidia-smi
```

The output should indicate a driver version, CUDA version, and so on. If you get an error, stop here and troubleshoot how to get Nvidia drivers, CUDA, and/or a connection to your GPU with the environment you're using.

# First time setup

## Update Ubuntu 20.04 packages
```
sudo apt update
sudo apt upgrade -y
```

## Download Anaconda (python env manager) installer
```
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
```

## Install Anaconda
```
bash Anaconda3-2021.11-Linux-x86_64.sh
respond 'yes' to accept license terms and provide install dir when prompted
respond 'yes' to run conda initialization
```

## Reboot for changes to take effect
```
sudo reboot
```

## Create prog rock diffusion env
```
conda create --name progrockdiffusion python=3.7
conda activate progrockdiffusion
```

## Clone the prog rock diffusion repo
```
git clone https://[your-github-username]@github.com/lowfuel/progrockdiffusion.git
(you'll need to generate an access token on github and then use it as your password here)
cd progrockdiffusion
```
Note: the "cd" command above is important, as the next steps will add additional libraries and data to ProgRockDiffusion

## Install the required libraries and tools
```
git clone https://github.com/crowsonkb/guided-diffusion
git clone https://github.com/openai/CLIP.git
git clone https://github.com/assafshocher/ResizeRight.git
git clone https://github.com/facebookresearch/SLIP.git
git clone https://github.com/CompVis/latent-diffusion.git
git clone https://github.com/CompVis/taming-transformers
pip install -e ./CLIP
pip install -e ./guided-diffusion
pip install -e ./taming-transformers
pip install lpips datetime timm
pip install https://download.pytorch.org/whl/cu111/torch-1.10.0%2Bcu111-cp37-cp37m-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu111/torchaudio-0.10.0%2Bcu111-cp37-cp37m-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu111/torchvision-0.11.1%2Bcu111-cp37-cp37m-linux_x86_64.whl
pip install ipywidgets omegaconf pytorch_lightning einops
pip install matplotlib pandas
conda install opencv
sudo apt install imagemagick
```

NOTE: On your first run it might appear to hang. Let it go for a good while, though, as it might just be downloading models.
Somtimes there is no feedback during the download process (why? Who knows)

# Use

```
usage: python3 prd.py [-h] [-s SETTINGS] [-o OUTPUT] [-p PROMPT]

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
  -i, --ignoreseed
                        Use a random seed instead of what is in your settings file

Usage examples:

To use the Default output directory and settings from settings.json:
 python3 prd.py

To use your own settings.json file (note that putting it in quotes can help parse errors):
 python3 prd.py -s "some_directory/mysettings.json"

To quickly just override the output directory name and the prompt:
 python3 prd.py -p "A cool image of the author of this program" -o Coolguy

Multiple prompts with weight values are supported:
 python3 prd.py -p "A cool image of the author of this program" -p "Pale Blue Sky:.5"

You can ignore the seed coming from a settings file by adding -i, resulting in a new random seed
```
Simply edit the settings.json file provided, or copy it and make several that include your favorite settings, if you wish to tweak the defaults.

# Notes

- Currently Superres Sampling doesn't work, it will crash.
- Animations are untested but likely not working

# TODO

- The SLIP models are currently failing due to a variable not being defined.
- Provide a section in this readme that goes over all the settings in settings.json and what they do, since we don't have the colab notebook to show those hints
