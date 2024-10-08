# author Fahad Akbar , fahadakbar@gmail.com
# Choose base image
FROM jupyter/base-notebook:latest

# name your environment and choose python 3.x version
ARG conda_env=pycaret
ARG py_ver=3.11

# add additional libraries you want mamba to install by listing them below
RUN mamba create --quiet --yes -p "${CONDA_DIR}/envs/${conda_env}" python=${py_ver} ipython ipykernel && \
    mamba clean --all -f -y 


# create Python 3.x environment and link it to jupyter
RUN "${CONDA_DIR}/envs/${conda_env}/bin/python" -m ipykernel install --user --name="${conda_env}" && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# install pycaret light version
RUN "${CONDA_DIR}/envs/${conda_env}/bin/pip" install pycaret>=3.3.2


# prepend conda environment to path
ENV PATH "${CONDA_DIR}/envs/${conda_env}/bin:${PATH}"

# make this environment to be the default one
ENV CONDA_DEFAULT_ENV ${conda_env}
