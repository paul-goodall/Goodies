
#!/usr/bin/env bash

my_wd=`pwd`
my_project_name=${PWD##*/}          # to assign to a variable
my_project_name=${my_project_name:-/}        # to correct for the case where PWD=/
my_host=`hostname`
my_host=${my_host/.local/}
my_envs_dir="${my_wd}/python_envs"
my_envs_git_dir="${my_envs_dir}/.git"
my_host_env="${my_envs_dir}/${my_host}"
my_python_installs="${my_envs_dir}/python_installs.sh"
echo $my_wd
echo $my_project_name
echo $my_host
echo $my_envs_dir
echo $my_host_env
echo $my_python_installs

# Get things up-to-date
python3 -m pip install --upgrade pip
python3 -m pip install setuptools wheel virtualenv --upgrade

# Create the env if it doesn't already exist:
if [ ! -d "$my_host_env" ];
then
  python3 -m venv $my_host_env
fi

# activate the venv:
source ${my_host_env}/bin/activate

# install ipython INSIDE the venv
python3 -m pip install --upgrade pip
python3 -m pip install ipython
python3 -m pip install ipykernel

# install the venv as a Jupyter kernel
ipython kernel install --user --name=${my_project_name}

bash $my_python_installs
