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

print('Congrats!  You now have Python Magic at your fingertips.  Call as gm.magic_function etc.')
# ==============
```
