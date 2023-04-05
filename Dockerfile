FROM gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/prp:latest

LABEL maintainer="Simon Poon <spoon@ucsd.edu>"

USER root

RUN apt-get -q update && \ 
    apt-get -y install openssh-client
    apt-get install -yq --no-install-recommends \
    gdal-bin libgdal-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

RUN pip install --quiet --no-cache-dir \
    uproot \
    awkward \
    uproot \
    tqdm \
    setGPU \
    mplhep \
    autopep8 \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    torch-geometric \
    git+https://github.com/google/qkeras#egg=qkeras \
    git+https://github.com/jmduarte/hls4ml@l1metml#egg=hls4ml[profiling]

