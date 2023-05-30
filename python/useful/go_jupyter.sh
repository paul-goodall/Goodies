#!/usr/bin/env bash

my_wd=`pwd`
my_project_name=${PWD##*/}              # to assign to a variable
my_project_name=${my_project_name:-/}   # to correct for the case where PWD=/
my_host=`hostname`
my_host=${my_host/.local/}
my_envs_dir="${my_wd}/python_envs"
my_envs_git_dir="${my_envs_dir}/.git"
my_host_env="${my_envs_dir}/${my_host}"
my_create_python_envs="${my_envs_dir}/create_python_env.sh"
my_python_installs="${my_envs_dir}/python_installs.sh"
echo $my_wd
echo $my_project_name
echo $my_host
echo $my_envs_dir
echo $my_host_env
echo $my_python_installs

# activate the venv:
source ${my_host_env}/bin/activate
jupyter notebook & disown 
