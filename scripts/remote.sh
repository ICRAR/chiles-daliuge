#!/usr/bin/env bash

START_DIR=${PWD}

# Setup environment

DEFAULT_INSTALL="/software/projects/pawsey0411/daliuge"
read -p "Create directories in default directory ($DEFAULT_INSTALL) (y/n): " RESPONSE
if [ "${RESPONSE}" == 'y' ]; then
  INSTALL_DIR=${DEFAULT_INSTALL}
else
  read -p "Please enter alternate directory: " INSTALL_DIR
fi

mkdir -p ${INSTALL_DIR}
cd ${INSTALL_DIR}

# Make virtual environment
python -m venv dlg_env
source dlg_env/bin/activate

# Install dependencies
cd ${START_DIR}
make install

deactivate

# Setup scratch if not already setup

DEFAULT_SCRATCH="/scratch/pawsey0411/chiles-daliuge"
mkdir ${DEFAULT_SCRATCH}


