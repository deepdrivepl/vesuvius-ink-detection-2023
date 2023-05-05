set -e
docker build -t vesuvius .
# docker run --gpus all --hostname $(hostname) --ipc=host --rm -v $(pwd):/host -v /mnt/12TB/Data/Kaggle/Vesuvius2023:/data -p 18003:18003 -it vesuvius jupyter notebook --allow-root --ip 0.0.0.0 --port 18003
docker run --gpus all --hostname $(hostname) --ipc=host --rm -v $(pwd):/host -v /mnt/ssd4/Data/Kaggle/Vesuvius2023:/data -it vesuvius bash