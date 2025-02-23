# Goodsy Magic Python Library - A Reference

Note - all required modules for this library have been installed on the `py10kernel` kernel.  If you use a different kernel you may have some installing to do.


```python
import sys
import subprocess

goodies_dir = '/Path/To/Goodies/Repo/'
goodies_dir = subprocess.check_output(['pwd']).decode().replace('goodsy_python\n','')

sys.path.append(goodies_dir)

from goodsy_python import magic as gm
```


```python

```

## Dates

### date_info_df

This function can be called on a single date-like string or on a pandas series:


```python
gm.date_info_df('2025-02-04')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>WeekDay</th>
      <th>Year</th>
      <th>Month</th>
      <th>DayOfMonth</th>
      <th>DOM</th>
      <th>DayOfYear</th>
      <th>DOY</th>
      <th>DaysInYear</th>
      <th>NumericDate</th>
      <th>DayName_1stJan</th>
      <th>WeekOfYear</th>
      <th>WOY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Tuesday</td>
      <td>2025</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>35</td>
      <td>35</td>
      <td>365</td>
      <td>2025.094521</td>
      <td>Wednesday</td>
      <td>6</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
import pandas as pd

df = pd.DataFrame({'Dt':['2021-02-03','2022-03-04','2023-04-05','2024-05-06']}, index=[0,1,2,3])
print(df)

gm.date_info_df(df['Dt'])
```

               Dt
    0  2021-02-03
    1  2022-03-04
    2  2023-04-05
    3  2024-05-06





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>WeekDay</th>
      <th>Year</th>
      <th>Month</th>
      <th>DayOfMonth</th>
      <th>DOM</th>
      <th>DayOfYear</th>
      <th>DOY</th>
      <th>DaysInYear</th>
      <th>NumericDate</th>
      <th>DayName_1stJan</th>
      <th>WeekOfYear</th>
      <th>WOY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wednesday</td>
      <td>2021</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>34</td>
      <td>34</td>
      <td>365</td>
      <td>2021.091781</td>
      <td>Friday</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Friday</td>
      <td>2022</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>63</td>
      <td>63</td>
      <td>365</td>
      <td>2022.171233</td>
      <td>Saturday</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Wednesday</td>
      <td>2023</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>95</td>
      <td>95</td>
      <td>365</td>
      <td>2023.258904</td>
      <td>Sunday</td>
      <td>15</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Monday</td>
      <td>2024</td>
      <td>5</td>
      <td>6</td>
      <td>6</td>
      <td>127</td>
      <td>127</td>
      <td>366</td>
      <td>2024.345628</td>
      <td>Monday</td>
      <td>19</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
