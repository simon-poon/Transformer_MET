FROM gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/prp:latest

LABEL maintainer="Simon Poon <spoon@ucsd.edu>"

USER root

RUN apt-get update \
    && apt-get -yq --no-install-recommends install openssh-client vim emacs \
    && sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*

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
