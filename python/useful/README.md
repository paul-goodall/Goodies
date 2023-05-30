# python_envs

This script is just a neat hack to enable env-specific python kernels for Jupyter across multiple machines of various architectures.

I found this useful because of wanting to use a Jupyter notebook on my Desktop (Intel Mac) and also use it and make edits on the fly with my Laptop (Apple M1).

The script does the following:

* clones this repo into your working directory
* creates a specific python env for your current project, named as your HOSTNAME
* creates a specific python kernel for that environment, named as your PROJECT
* installs a set of python modules into the local environment, according to `python_installs.sh`
* You can then run Jupyter using `go_jupyter.sh` on any of your machines (provided that you have initialised that machine once with `python_envs` for you project)
* Each machine will look for the PROJECT kernel on your environment (being none the wiser that it's actually using the local kernel)

Disclaimer - only tested on Macs (Intel and M1) so far.

## Getting going:

Run the following code from the directory of your latest python project.
e.g:

`cd /path/to/my_new_project`

`wget -O - https://raw.githubusercontent.com/paul-goodall/python_envs/main/go_python_env.sh | bash`
