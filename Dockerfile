# Stage 1: Base image with common dependencies
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 



# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    # add-apt-repository -y ppa:git-core/ppa && apt-get update -y \
    python3.10 \
    python3-pip \
    wget \
    aria2 \
    git \
    git-lfs \
    unzip \
    ffmpeg

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install ComfyUI dependencies
RUN pip3 install --upgrade --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --upgrade -r requirements.txt

# Install runpod
RUN pip3 install runpod requests

RUN pip3 install opencv-python imageio imageio-ffmpeg ffmpeg-python av \
xformers==0.0.25 torchsde==0.2.6 einops==0.8.0 diffusers==0.28.0 transformers==4.44.2 accelerate==0.33.0 insightface==0.7.3 onnxruntime==1.18.0 onnxruntime-gpu==1.18.0 color-matcher==0.5.0 pilgram==1.2.1 \
ultralytics==8.2.27 segment-anything==1.0 piexif==1.1.3 qrcode==7.4.2 requirements-parser==0.9.0 rembg==2.0.57 rich==13.7.1 rich-argparse==1.5.1 matplotlib==3.8.4 pillow spandrel==0.3.4 \
scikit-image==0.24.0 opencv-python-headless==4.10.0.84 GitPython==3.1.43 scipy==1.14.0 numpy==1.26.4 cachetools==5.4.0 librosa==0.10.2.post1 importlib-metadata==8.0.0 PyYAML==6.0.1 filelock==3.15.4 \
mediapipe==0.10.14 svglib==1.5.1 fvcore==0.1.5.post20221221 yapf==0.40.2 omegaconf==2.3.0 ftfy==6.2.0 addict==2.4.0 yacs==0.1.8 albumentations==1.4.11 scikit-learn==1.5.1 fairscale==0.4.13 bitsandbytes \ 
git+https://github.com/WASasquatch/img2texture git+https://github.com/WASasquatch/cstr git+https://github.com/WASasquatch/ffmpy joblib==1.4.2 numba==0.60.0 timm==1.0.7 tqdm==4.66.4

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Add the start and the handler
ADD src/start.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh

# Stage 2: Download models
FROM base as downloader

ARG HUGGINGFACE_ACCESS_TOKEN
ARG MODEL_TYPE

# Change working directory to ComfyUI
WORKDIR /comfyui

RUN git clone https://github.com/ltdrdata/ComfyUI-Manager custom_nodes/ComfyUI-Manager
RUN git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts custom_nodes/ComfyUI-Custom-Scripts
RUN git clone https://github.com/cubiq/ComfyUI_essentials custom_nodes/ComfyUI_essentials
RUN git clone https://github.com/rgthree/rgthree-comfy custom_nodes/rgthree-comfy
RUN git clone https://github.com/Derfuu/Derfuu_ComfyUI_ModdedNodes custom_nodes/Derfuu_ComfyUI_ModdedNodes
RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack custom_nodes/ComfyUI-Impact-Pack
RUN git clone https://github.com/ltdrdata/ComfyUI-Inspire-Pack custom_nodes/ComfyUI-Inspire-Pack
RUN git clone https://github.com/aidenli/ComfyUI_NYJY custom_nodes/ComfyUI_NYJY
RUN git clone https://github.com/kijai/ComfyUI-KJNodes custom_nodes/ComfyUI-KJNodes
RUN git clone https://github.com/Fannovel16/comfyui_controlnet_aux custom_nodes/comfyui_controlnet_aux
RUN git clone https://github.com/shiimizu/ComfyUI-TiledDiffusion custom_nodes/ComfyUI-TiledDiffusion
RUN git clone https://github.com/WASasquatch/was-node-suite-comfyui custom_nodes/was-node-suite-comfyui
RUN git clone https://github.com/Extraltodeus/ComfyUI-AutomaticCFG custom_nodes/ComfyUI-AutomaticCFG

RUN git clone https://huggingface.co/google/siglip-so400m-patch14-384 models/clip/siglip-so400m-patch14-384
RUN git clone https://huggingface.co/unsloth/Meta-Llama-3.1-8B-bnb-4bit models/llm/Meta-Llama-3.1-8B-bnb-4bit
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/ultralytics/resolve/main/PitEyeDetailer-v2-seg.pt -d models/ultralytics/segm -o PitEyeDetailer-v2-seg.pt
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/ultralytics/resolve/main/4xRealWebPhoto_v4_dat2.safetensors -d models/upscale_models -o 4xRealWebPhoto_v4_dat2.safetensors
RUN  aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/ultralytics/resolve/main/dreamshaperXL_lightningDPMSDE.safetensors -d models/checkpoints -o dreamshaperXL_lightningDPMSDE.safetensors
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/ultralytics/resolve/main/xinsir-controlnet-tile-sdxl-1.0.safetensors -d models/controlnet -o xinsir-controlnet-tile-sdxl-1.0.safetensors
RUN  aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/flux1-dev.sft -d models/unet -o flux1-dev.sft 
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/ae.sft -d models/vae -o ae.sft
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/clip_l.safetensors -d models/clip -o clip_l.safetensors
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/t5xxl_fp16.safetensors -d models/clip -o t5xxl_fp16.safetensors
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/unsloth/Meta-Llama-3.1-8B-bnb-4bit/resolve/main/model.safetensors -d models/clip -o t5xxl_fp16.safetensors

# Stage 3: Final image
FROM base as final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models
COPY --from=downloader /comfyui/custom_nodes /comfyui/custom_nodes

# Start the container
CMD /start.sh