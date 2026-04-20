# prerequisite

cryosparc (version<=4.7.1)
relion
chimerax

# Install anaconda (skip this step if you already have conda)

# Download the code of cryoagent
git clone https://gitee.com/fei_sun_lab/cryoagent

# download the cryoAlign2

curl -L https://zenodo.org/records/19552663/files/cryoalign_env.tar.gz?download=1 -o ~/cryoalign_env.tar.gz

# Install the necessary envs and code

bash ./install.sh

(it takes around 20 mins, installation of the cryoAlign2 need sudo permission, that is the property of docker)

# Please change the info in the following json files

## In master_config.json

### The cryosparc config need to be set
"cryosparc": {
    "host": "your_host_address (localhost)",
    "base_port": "your_port_name (39000)",
    "username": "your_user_name ()",
    "password": "your_password ()",
    "license_id": "${LICENSE_ID}"
  },

### The Relion config also need to be set

"relion": {
    "relion_exe": "your relion execuate file (/usr/local/bin/relion)",
    "continue_job": true,
    "relion_dir": "your_relion_directory",
}
