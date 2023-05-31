# ==============================================================================
# Goodsy functions to make R a bit less awful.
# This is not how R functions were intended to be used, but I find it saves time
# and all the hassle relating to the ever-changing R-packaging setup nonsense.
# Requirements:
# 1. Every function must be attached to the g_ object, as a one-stop-shop
# 2. Every function must have a 'help' component
# 3. Args to functions must be safeguarded against null values
# ==============================================================================
# Load packages:
library("jsonlite")
library('tidyverse')
library('rvest')
library('data.table')
library('xml2')
library('httr')
library('stringi')
# ==============================================================================
g_ <- list()
# ------------------------------------------------------------------------------
g_$gsub <- function(.data=NULL, ss1=NULL, ss2=NULL){
  # ...............
  help_message <- 'Must specify arg: .data
  --------------------
  This function just makes gsub pipe-friendly.
  Usage:
  some_string_variable %>% g_$gsub(str1,str2)
  --------------------
  '
  if(is.null(.data)) stop(help_message, call. = F)
  # ...............
  # safeguards:
  if(is.null(ss1)) stop('Must specify arg: ss1')
  if(is.null(ss2)) stop('Must specify arg: ss2')
  # ...............
  gsub(ss1,ss2,.data)
}
# ------------------------------------------------------------------------------
g_$str_del <- function(.data=NULL, ss1=NULL){
  # ...............
  help_message <- 'Must specify arg: .data
  --------------------
  This function deletes the specified substring.
  Usage:
  some_string_variable %>% g_$str_del(str1)
  
  This is equivalent to:
  some_string_variable %>% g_$gsub(str1,"")
  --------------------
  '
  if(is.null(.data)) stop(help_message, call. = F)
  # ...............
  # safeguards:
  if(is.null(ss1)) stop('Must specify arg: ss1')
  # ...............
  gsub(ss1,'',.data)
}
# ------------------------------------------------------------------------------
g_$read_html <- function(.data=NULL, outfile=NULL, ip=NULL, port=NULL){
  # ...............
  help_message <- 'Must specify arg: .data
  --------------------
  This function reads a html page and saves it as a local file.
  In the instance where the local file exists, it will read this file rather
  than calling out to the webpage again, for speed reasons.
  If the outfile is left null, then no outfile is created.
  Can take an optional ip and port if needed.
  Returns NULL if the url is not found.
  Usage:
  webpage <- g_$read_html(.data=url, [my_html_file], [ip], [port])
  --------------------
  '
  if(is.null(.data)) stop(help_message, call. = F)
  # ...............
  # safeguards:
  mode <- 1
  if((!is.null(ip))*(!is.null(port)) == 1) {
    mode <- 2
  }
  # ...............
  if(is.null(outfile)){
    webpage <- tryCatch({
      if(mode == 1) read_html(url)
      if(mode == 2) content( GET( url, use_proxy(ip, as.numeric(port)) ) )
    }, warning = function(w){NULL}, error = function(e){NULL})
  } else {
    if(file.exists(outfile)){
      print('Getting webpage locally.')
      webpage <- read_html(outfile)
    } else {
      print('Getting webpage remotely.')
      webpage <- tryCatch({
        if(mode == 1) read_html(url)
        if(mode == 2) content( GET( url, use_proxy(ip, as.numeric(port)) ) )
      }, warning = function(w){NULL}, error = function(e){NULL})
      if(class(webpage)[1] == "xml_document") write_xml(webpage, file=outfile)
    }
  }
  return (webpage)
}
# ------------------------------------------------------------------------------
g_$left <- function(.data=NULL, numchars=NULL){
  # ...............
  help_message <- 'Must specify arg: .data
  --------------------
  This function takes the first numchars from the beginning of the string.
  Usage:
  some_string_variable %>% g_$left(numchars)
  Example:
  > "australia" %>% g_$left(6)
  [1] "austra"
  --------------------
  '
  if(is.null(.data)) stop(help_message, call. = F)
  # ...............
  # safeguards:
  if(is.null(.data)) stop('Must specify arg: .data')
  if(is.null(numchars)) stop('Must specify arg: numchars')
  # ...............
  if(numchars < 0){
    numchars <- nchar(.data) + numchars
  }
  my_str <- substr(.data, 1, numchars)
  return (my_str)
}
# ------------------------------------------------------------------------------
g_$right <- function(.data=NULL, numchars=NULL){
  # ...............
  help_message <- 'Must specify arg: .data
  --------------------
  This function takes the last numchars from the end of the string.
  Usage:
  some_string_variable %>% g_$right(numchars)
  Example:
  > "australia" %>% g_$right(6)
  [1] "tralia"
  --------------------
  '
  if(is.null(.data)) stop(help_message, call. = F)
  # ...............
  # safeguards:
  if(is.null(numchars)) stop('Must specify arg: numchars')
  # ...............
  if(numchars < 0){
    numchars <- nchar(.data) + numchars
  }
  my_str <- stri_reverse(.data)
  my_str <- substr(my_str, 1, numchars)
  my_str <- stri_reverse(my_str)
  return (my_str)
}
# ------------------------------------------------------------------------------
g_$pad0 <- function(.data=NULL, num0=NULL){
  # ...............
  help_message <- 'Must specify arg: .data
  --------------------
  This function pads an integer with leading zeroes and returns a string.
  Usage:
  some_integer %>% g_$pad0(num0)
  Example:
  > 21 %>% g_$pad0(6)
  [1] "000021"
  --------------------
  '
  if(is.null(.data)) stop(help_message, call. = F)
  # ...............
  # safeguards:
  if(is.null(num0)) stop('Must specify arg: num0')
  # ...............
  my_str <- sprintf(paste0("%0",num0,"d"), .data)
  return (my_str)
}
# ------------------------------------------------------------------------------
g_$phantom_get_url <- function(.data=NULL, my_outfile=NULL){
  # ...............
  help_message <- 'Must specify arg: .data
  --------------------
  This function uses phantomjs to evaluate the JS on a webpage and return the full file.
  Usage:
  some_URL %>% g_$phantom_get_url(my_outfile="my_file.html")
  --------------------
  '
  if(is.null(.data)) stop(help_message, call. = F)
  # ...............
  # safeguards:
  if(is.null(my_outfile)) stop('Must specify arg: my_outfile')
  # ...............
  if(!file.exists(my_outfile)){
    com <- paste0('./phantomjs/bin/phantomjs js_eval_webpage.js ', .data, ' ', my_outfile)
    cat(com,'\n')
    system(com)
  }
  webpage <- read_html(my_outfile)
  return (webpage)
}
# ------------------------------------------------------------------------------
g_$fast_combine_csv_set <- function(.data=NULL, my_outfile=NULL){
  # ...............
  help_message <- 'Must specify arg: .data
  --------------------
  This function concatenates a bunch of files into a master file.
  Usage:
  some_glob_pattern %>% g_$fast_combine_csv_set(my_outfile="my_file.csv[.gz]")
  --------------------
  '
  if(is.null(.data)) stop(help_message, call. = F)
  # ...............
  # safeguards:
  if(is.null(my_outfile)) stop('Must specify arg: my_outfile')
  # ...............
  my_csv_glob <- .data
  my_tmpfile <- gsub(".gz","",my_outfile)
  my_files <- Sys.glob(my_csv_glob)
  first_file <- my_files[1]
  if((tools::file_ext(first_file) == "csv")){
    com <- "head -n 1 first_file > my_tmpfile; tail -n +2 -q my_csv_glob >> my_tmpfile;"
    com <- gsub("my_tmpfile", my_tmpfile, com)
    com <- gsub("my_csv_glob", my_csv_glob, com)
    com <- gsub("first_file", first_file, com)
    cat(com,"\n")
    system(com)
  }
  if((tools::file_ext(first_file) == "gz")){
    com <- "zcat first_file |  head -n 1  > my_tmpfile;"
    com <- gsub("my_tmpfile", my_tmpfile, com)
    com <- gsub("first_file", first_file, com)
    cat(com,"\n")
    system(com)
    for(my_file in my_files){
      com <- "zcat my_file |  tail -n +2 -q  >> my_tmpfile;"
      com <- gsub("my_file", my_file, com)
      com <- gsub("my_tmpfile", my_tmpfile, com)
      cat(com,"\n")
      system(com)
    }
  }
  if((tools::file_ext(my_outfile) == "gz")){
    com <- "gzip my_tmpfile;"
    com <- gsub("my_tmpfile", my_tmpfile, com)
    cat(com,"\n")
    system(com)
  }
  my_df <- fread(my_outfile)
  return (my_df)
}

# ------------------------------------------------------------------------------
g_$countdesc <- function(.data=NULL, ...){
  # ...............
  help_message <- 'Must specify arg: .data
  --------------------
  This function uses phantomjs to evaluate the JS on a webpage and return the full file.
  Usage:
  some_df %>% g_$countdesc(grouping_var1,grouping_var2,...)
  --------------------
  '
  if(is.null(.data)) stop(help_message, call. = F)
  # ...............
  return (.data %>% group_by(...) %>% summarise('n'=n()) %>% arrange(desc(n)))
}
# ------------------------------------------------------------------------------

# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================