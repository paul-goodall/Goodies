# Goodsy Magic Python Library - A Reference

Note - all required modules for this library have been installed on the `py10kernel` kernel.  If you use a different kernel you may have some installing to do.


```python
import sys
sys.path.append('/Volumes/Abyss/Dropbox/my_DataScience/Repos/Goodies/')
from goodsy_python import magic as gm
```


```python

```


```python

```


```python

```

## Dates


```python


gm.date_info_df('2025-02-04')

import pandas as pd

aa = pd.DataFrame({'Dt':['2025-02-04','2025-02-05']}, index=[0,1])
aa['Dt2'] = pd.to_datetime(aa.Dt, format="%Y-%m-%d")
gm.create_useful_dates(aa['Dt2'])
```


```python

```
