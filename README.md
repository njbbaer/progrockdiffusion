# progrockdiffusion
A command line version of Disco Diffusion

# Hardware prerequisites
An Nvidia GPU capable of running CUDA-based softare. 8gb is probably the minimum amount of GPU memory.

This author has an RTX 3080 with 10gb and it runs fairly well, but some advanced features are not possible with "only" 10gb.

# Software prerequisties
Ubuntu 20.04 (A docker environment, VM, or Windows Subsystem for Linux will work provided it can access your GPU).

CUDA 11.4

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
git clone git@github.com:lowfuel/progrockdiffusion.git
cd progrockdiffusion
```

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
apt install imagemagick
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
