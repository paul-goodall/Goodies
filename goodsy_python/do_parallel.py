#!/usr/bin/env Rscript
library(optparse)
 
option_list = list(
  make_option(c("-s", "--script"), type="character", default=NULL, 
              help="script to run in parallel", metavar="character")
); 
 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);


# --------------------------------------------------
library(parallel)

# create the variables to share with the cluster:
num_cores   <- detectCores()
script_path <- opt$script

cat('Detected ', num_cores, ' cores.\n')
cl <- makeCluster(num_cores, outfile = "debug.txt")
# --------------------------------------------------
single_command <- function(procnum, num_cores, script_path){

  exec_cmd    <- '/opt/conda/bin/python'

  com <- paste(exec_cmd, script_path, '-n', num_cores, '-p', procnum)
  print(com)
  system(com)

}
clusterExport(cl, c('num_cores', 'script_path', 'single_command'))
# ------------------------------------------------
parallel_wrapper <- function(x) {
  single_command(x, num_cores, script_path)
}
# ------------------------------------------------
x <- 1:num_cores
clusterMap(cl, parallel_wrapper, x)
# --------------------------------------------------
stopCluster(cl)
# ------------------------------------------------
