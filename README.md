# progrockdiffusion
A command line version of [Disco Diffusion](https://github.com/alembics/disco-diffusion).

# Hardware prerequisites
An Nvidia GPU capable of running CUDA-based softare. 8gb is probably the minimum amount of GPU memory.

This author has an RTX 3080 with 10gb and it runs fairly well, but some advanced features are not possible with "only" 10gb.

You'll also need between 20 and 40gb of free disk space, depending on which models you enable.

# Software prerequisties
## Linux
Ubuntu 20.04 or similar (A docker environment, VM, or Windows Subsystem for Linux should work provided it can access your GPU).

CUDA 11.4+ (installation instructions can be found here: https://developer.nvidia.com/cuda-11-4-1-download-archive).

## Windows
Windows 10 or 11 with NVIDIA drivers installed (other versions may work but are untested)

## Test NVIDIA drivers
You can test that your environment is working properly by running:

```
nvidia-smi
```

The output should indicate a driver version, CUDA version, and so on. If you get an error, stop here and troubleshoot how to get Nvidia drivers, CUDA, and/or a connection to your GPU with the environment you're using.

# First time setup

## **Linux** Update Ubuntu 20.04 packages
```
sudo apt update
sudo apt upgrade -y
```

## Download and install Anaconda
### **Linux**
```
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
```

Install Anaconda
```
bash Anaconda3-2021.11-Linux-x86_64.sh
respond 'yes' to accept license terms and provide install dir when prompted
respond 'yes' to run conda initialization
```

### **Windows**
```
Download from here and install: https://www.anaconda.com/products/individual
```

## **Linux**
Logout and back in for the changes to take effect
## **Windows**
From the start menu, open a "Anaconda Powershell Prompt" (*Powershell* is important)

## Create prog rock diffusion env
```
conda create --name progrockdiffusion python=3.7
conda activate progrockdiffusion
```

Now change to whatever base directory you want ProgRockDiffusion to go into.
## Clone the prog rock diffusion repo
```
git clone https://github.com/lowfuel/progrockdiffusion.git
cd progrockdiffusion
```
**Note: the "cd" command above is important, as the next steps will add additional libraries and data to ProgRockDiffusion**

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
```
## Install PyTorch
### **Linux**
```
pip install https://download.pytorch.org/whl/cu111/torch-1.10.0%2Bcu111-cp37-cp37m-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu111/torchaudio-0.10.0%2Bcu111-cp37-cp37m-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu111/torchvision-0.11.1%2Bcu111-cp37-cp37m-linux_x86_64.whl
```
## **Windows**
```
pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
## Install remaining libraries and tools
```
pip install ipywidgets omegaconf pytorch_lightning einops
pip install matplotlib pandas
conda install opencv
```
**Linux** Depending on your Linux platform, you may get an error about libGL.so.1
If you do, try installing these dependencies:
```
sudo apt-get install ffmpeg libsm6 libxext6 -y
```
**Linux** Finally:
```
sudo apt install imagemagick
```

# Use

NOTE: On your first run it might appear to hang. Let it go for a good while, though, as it might just be downloading models.
Somtimes there is no feedback during the download process (why? Who knows)


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
- Native Windows support
