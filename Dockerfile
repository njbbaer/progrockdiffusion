FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

RUN echo "" && \
    echo "-================================================================-" && \
    echo "-================================================================-" && \
    echo "" && \
    echo "This dockerfile is GPU only, and needs you to nominate the GPUs to" && \
    echo "run.  The common form is 'docker build . -t progrockdiffusion' and" && \
    echo "'docker run progrockdiffusion --gpus all'" && \
    echo "" && \
    echo "-================================================================-" && \
    echo "-================================================================-" && \
    echo "" && \
    echo "Updating the base" && \
    apt update && \
    apt update && \
    apt upgrade -y && \
    apt install git -y && \
    apt install nano -y && \
    apt install curl -y && \
    apt install wget -y

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata

RUN apt-get -y update && \
    apt-get install -y software-properties-common && \
    apt-get -y update && \
    add-apt-repository universe

RUN apt-get -y install python3

RUN apt-get -y install python3-pip

RUN echo "Pulling the various repos" && \
    git clone https://github.com/lowfuel/progrockdiffusion.git && \
    cd progrockdiffusion && \
    git clone https://github.com/crowsonkb/guided-diffusion && \
    git clone https://github.com/openai/CLIP.git && \
    git clone https://github.com/assafshocher/ResizeRight.git && \
    git clone https://github.com/facebookresearch/SLIP.git && \
    git clone https://github.com/CompVis/latent-diffusion.git && \
    git clone https://github.com/CompVis/taming-transformers

RUN cd progrockdiffusion && \
    pip -V && \
    pip install opencv-python && \
    pip install -e ./CLIP && \
    pip install -e ./guided-diffusion && \
    pip install -e ./taming-transformers && \
    pip install lpips datetime timm

RUN pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html --no-cache && \
    pip install torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html && \
    pip install torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html && \
# #    sudo apt-get install ffmpeg libsm6 libxext6 -y && \
    apt install imagemagick -y && \
    pip install ipywidgets omegaconf pytorch_lightning einops && \
    pip install matplotlib pandas

# RUN nvidia-smi

# CMD /bin/bash
