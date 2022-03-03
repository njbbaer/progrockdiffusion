apt update
apt install -y git imagemagick ffmpeg libsm6 libxext6

git clone https://github.com/crowsonkb/guided-diffusion
git clone https://github.com/openai/CLIP.git
git clone https://github.com/assafshocher/ResizeRight.git
git clone https://github.com/facebookresearch/SLIP.git
git clone https://github.com/CompVis/latent-diffusion.git
git clone https://github.com/CompVis/taming-transformers

pip install -e ./CLIP
pip install -e ./guided-diffusion
pip install -e ./taming-transformers

pip install lpips datetime timm ipywidgets omegaconf pytorch_lightning einops matplotlib pandas opencv-python