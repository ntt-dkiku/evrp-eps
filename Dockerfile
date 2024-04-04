FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN echo "Building docker image"

RUN apt-get -y update && \
    apt-get install -y \
    curl \
    build-essential \
    git \
    vim \
    tmux 

# jupyter-notebook & lab
RUN python3 -m pip install jupyter==1.0.0
RUN python3 -m pip install jupyterlab==4.0.2

# OR-tools 
RUN python3 -m pip install ortools==9.6.2534

# other convenient packages
RUN python3 -m pip install numpy
RUN python3 -m pip install scipy
RUN python3 -m pip install pandas
RUN python3 -m pip install matplotlib
RUN python3 -m pip install tqdm
RUN python3 -m pip install scikit-learn
