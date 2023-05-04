FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
RUN apt-get update && apt-get install git -y
RUN pip install nb_black monai lovely-numpy tqdm pytorch-lightning einops notebook jupyterlab pandas ipywidgets
WORKDIR /host
