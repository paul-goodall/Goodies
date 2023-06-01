# Goodies

## Python Usage:

```python
# ==============
# Python magic:
import os
import sys
repos_dir   = '/Path/To/Your/Repos/'
goodies_dir = repos_dir + 'Goodies'
magic_dir   = goodies_dir + '/goodsy_python'

if not os.path.exists(goodies_dir):
    com='git clone https://github.com/paul-goodall/Goodies.git ' + goodies_dir
    print(com)
    os.system(com)

sys.path.append(goodies_dir)
from goodsy_python import magic as gm

pd  = gm.pd
np  = gm.np
cv2 = gm.cv2
plt = gm.plt

msg  = 'Congrats!  '
msg += 'You now have Python Magic at your fingertips. '
msg += 'Usage: gm.magic_function '
print(msg)
# ==============
```

## R Usage:

```R
# ==============
repos_dir   <- '/Path/To/Your/Repos/'
goodies_dir <- paste0(repos_dir, 'Goodies')
magic_dir   <- paste0(goodies_dir, '/goodsy_R')

if(!file.exists(goodies_dir)){
  com <- paste0('git clone https://github.com/paul-goodall/Goodies.git ', goodies_dir)
  cat(com)
  system(com)
}

magic_R <- paste0(magic_dir, '/magic.R')
source(magic_R)

msg <- 'Congrats!  '
msg <- paste0(msg, 'You now have R Magic at your fingertips. ')
msg <- paste0(msg, 'Usage: gm$magic_function ')
cat(msg)
# ==============
```
