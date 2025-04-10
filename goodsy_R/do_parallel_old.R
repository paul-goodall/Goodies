library(parallel)
num_cores <- detectCores()

cl<-makeCluster(num_cores, outfile = "debug.txt")
# --------------------------------------------------
do_stuff_parallel <- function(x){

  exec_cmd    <- '/path/to/executor/eg/python'
  script_path <- '/path/to/some/script/eg/slave_file.py'
  data_path   <- '/path/to/some/datafile/eg/distribution.csv'

  com <- paste(exec_cmd, script_path, data_path, x)
  system(com)

}
# ------------------------------------------------
x <- 1:num_cores
clusterMap(cl, do_stuff_parallel, x)
# --------------------------------------------------
stopCluster(cl)
# ------------------------------------------------
