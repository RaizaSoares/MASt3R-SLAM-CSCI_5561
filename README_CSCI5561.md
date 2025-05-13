before cloning the fork, please follow the installation steps to build the conda environment following the official MASt-3R-SLAM guide- https://github.com/rmurai0610/MASt3R-SLAM

Run Mast3R-SLAM under the project root using the following command with your video replacing the dataset parameter:
python3 main.py --dataset data/video3.mp4 --config config/base.yaml

Note that sample videos are provided on GoogleDrive: https://drive.google.com/file/d/16nF2Z3-pakol-BPpZYK55D28wNMtDtVx/view?usp=drive_link
Please unzip the file and put the data folder into the repo root before using the command above.

Development and testing was conducted using Ubuntu 22.04.5 (On WSL), and CUDA 12.6