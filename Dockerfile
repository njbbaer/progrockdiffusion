FROM nvidia/cuda:11.6.0-runtime-ubuntu20.04

RUN echo "This dockerfile is GPU only, and needs you to nominate the GPUs to" && \
    echo "run.  The common form is 'docker build . -t dockerdiffusion' and" && \
    echo "'docker run dockerdiffusion --gpus all'" && \
    echo "" && \
    echo "Updating the base" && \
    apt update && \
    apt update && \
    apt upgrade -y

RUN echo "Setting up Anaconda" && \
    apt install wget -y && \
    wget -nv https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh && \
    bash Anaconda3-2021.11-Linux-x86_64.sh -b

RUN echo "Switching into a conda env for progrockdiffusion" && \
    conda create --name progrockdiffusion python=3.7 && \
    conda activate progrockdiffusion

RUN echo "Pulling the various repos" && \
    git clone https://github.com/lowfuel/progrockdiffusion.git && \
    cd progrockdiffusion && \
    git clone https://github.com/crowsonkb/guided-diffusion && \
    git clone https://github.com/openai/CLIP.git && \
    git clone https://github.com/assafshocher/ResizeRight.git && \
    git clone https://github.com/facebookresearch/SLIP.git && \
    git clone https://github.com/CompVis/latent-diffusion.git && \
    git clone https://github.com/CompVis/taming-transformers

RUN pip install -e ./CLIP && \
    pip install -e ./guided-diffusion && \
    pip install -e ./taming-transformers && \
    pip install lpips datetime timm && \
    pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html && \
    pip install torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html && \
    pip install torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html && \
# #    sudo apt-get install ffmpeg libsm6 libxext6 -y && \
    apt install imagemagick && \
    pip install ipywidgets omegaconf pytorch_lightning einops && \
    pip install matplotlib pandas && \
    conda install opencv

RUN nvidia-smi

# CMD /bin/bash
