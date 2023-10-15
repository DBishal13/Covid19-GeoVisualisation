# Analysing Covid-19 Geospatial data with Python


```python
from datetime import datetime 
import pandas as pd
import numpy as np
import geopandas as gpd
import contextily as ctxdflfölfd
import matplotlib.pyplot as plt
```

## Reading Data


```python
#url = "https://proxy.hxlstandard.org/data/e2bb4b/download/jrc-covid-19-regions-hxl.csv"

#df = pd.read_csv(url)
#df.head()
```


```python
#df.dtypes
```


```python
url = 'https://proxy.hxlstandard.org/data/e2bb4b/download/jrc-covid-19-regions-hxl.csv'
dtypes = {  # bool will return value error because we hav NaN values
    'Date': object,
    'iso3': object,
    'CountryName': object,
    'Region': object,
    'lat': float,
    'lon': float,
    'CumulativePositive': float,
    'CumulativeDeceased': float,
    'CumulativeRecovered': float,
    'CurrentlyPositive': float,
    'Hospitalized': float,
    'IntensiveCare': float,
    'EUcountry': np.bool,
    'EUCPMcountry': np.bool,
    'NUTS': object,
    }

df = pd.read_csv(url, skiprows=range(1, 2), dtype=dtypes)
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>iso3</th>
      <th>CountryName</th>
      <th>Region</th>
      <th>lat</th>
      <th>lon</th>
      <th>CumulativePositive</th>
      <th>CumulativeDeceased</th>
      <th>CumulativeRecovered</th>
      <th>CurrentlyPositive</th>
      <th>Hospitalized</th>
      <th>IntensiveCare</th>
      <th>EUcountry</th>
      <th>EUCPMcountry</th>
      <th>NUTS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-22</td>
      <td>GRC</td>
      <td>Greece</td>
      <td>NOT SPECIFIED</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>True</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-22</td>
      <td>ISL</td>
      <td>Iceland</td>
      <td>NOT SPECIFIED</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-22</td>
      <td>LIE</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>47.164696</td>
      <td>9.555000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>LI</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-22</td>
      <td>MCO</td>
      <td>Monaco</td>
      <td>Monaco</td>
      <td>43.738348</td>
      <td>7.424451</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>MC</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-22</td>
      <td>MKD</td>
      <td>North Macedonia</td>
      <td>North Macedonia</td>
      <td>41.611000</td>
      <td>21.751417</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>MK</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>iso3</th>
      <th>CountryName</th>
      <th>Region</th>
      <th>lat</th>
      <th>lon</th>
      <th>CumulativePositive</th>
      <th>CumulativeDeceased</th>
      <th>CumulativeRecovered</th>
      <th>CurrentlyPositive</th>
      <th>Hospitalized</th>
      <th>IntensiveCare</th>
      <th>EUcountry</th>
      <th>EUCPMcountry</th>
      <th>NUTS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>133450</th>
      <td>2021-01-29</td>
      <td>DEU</td>
      <td>Germany</td>
      <td>Saarland</td>
      <td>49.384368</td>
      <td>6.953135</td>
      <td>25232.0</td>
      <td>720.0</td>
      <td>NaN</td>
      <td>24512.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>True</td>
      <td>DEC</td>
    </tr>
    <tr>
      <th>133451</th>
      <td>2021-01-29</td>
      <td>DEU</td>
      <td>Germany</td>
      <td>Sachsen</td>
      <td>51.052334</td>
      <td>13.348561</td>
      <td>178330.0</td>
      <td>6153.0</td>
      <td>NaN</td>
      <td>172177.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>True</td>
      <td>DED</td>
    </tr>
    <tr>
      <th>133452</th>
      <td>2021-01-29</td>
      <td>DEU</td>
      <td>Germany</td>
      <td>Sachsen-Anhalt</td>
      <td>52.013193</td>
      <td>11.700691</td>
      <td>50945.0</td>
      <td>1653.0</td>
      <td>NaN</td>
      <td>49292.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>True</td>
      <td>DEE</td>
    </tr>
    <tr>
      <th>133453</th>
      <td>2021-01-29</td>
      <td>DEU</td>
      <td>Germany</td>
      <td>Schleswig Holstein</td>
      <td>54.029500</td>
      <td>9.705555</td>
      <td>35426.0</td>
      <td>834.0</td>
      <td>NaN</td>
      <td>34592.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>True</td>
      <td>DEF</td>
    </tr>
    <tr>
      <th>133454</th>
      <td>2021-01-29</td>
      <td>DEU</td>
      <td>Germany</td>
      <td>Thüringen</td>
      <td>50.903878</td>
      <td>11.024880</td>
      <td>64392.0</td>
      <td>2122.0</td>
      <td>NaN</td>
      <td>62270.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>True</td>
      <td>DEG</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (133455, 15)




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lat</th>
      <th>lon</th>
      <th>CumulativePositive</th>
      <th>CumulativeDeceased</th>
      <th>CumulativeRecovered</th>
      <th>CurrentlyPositive</th>
      <th>Hospitalized</th>
      <th>IntensiveCare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>124740.000000</td>
      <td>124740.000000</td>
      <td>1.323170e+05</td>
      <td>106394.000000</td>
      <td>8.986600e+04</td>
      <td>1.331730e+05</td>
      <td>60714.000000</td>
      <td>57952.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>49.185773</td>
      <td>21.552218</td>
      <td>1.820859e+04</td>
      <td>689.480450</td>
      <td>1.058690e+04</td>
      <td>1.037220e+04</td>
      <td>814.748361</td>
      <td>87.857071</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.323308</td>
      <td>26.150401</td>
      <td>9.616718e+04</td>
      <td>2888.820894</td>
      <td>7.231191e+04</td>
      <td>8.361337e+04</td>
      <td>3614.047833</td>
      <td>358.314878</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-20.890660</td>
      <td>-61.272382</td>
      <td>-1.246000e+03</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>-1.898900e+06</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>44.244926</td>
      <td>8.116991</td>
      <td>1.890000e+02</td>
      <td>1.000000</td>
      <td>0.000000e+00</td>
      <td>1.190000e+02</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>47.417407</td>
      <td>17.174810</td>
      <td>1.865000e+03</td>
      <td>42.000000</td>
      <td>1.160000e+02</td>
      <td>9.600000e+02</td>
      <td>3.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>54.928337</td>
      <td>26.889951</td>
      <td>9.217000e+03</td>
      <td>314.000000</td>
      <td>4.286000e+03</td>
      <td>4.237000e+03</td>
      <td>187.000000</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>94.793040</td>
      <td>166.788413</td>
      <td>3.106859e+06</td>
      <td>90698.000000</td>
      <td>2.340216e+06</td>
      <td>3.106802e+06</td>
      <td>57102.000000</td>
      <td>4750.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 133455 entries, 0 to 133454
    Data columns (total 15 columns):
     #   Column               Non-Null Count   Dtype  
    ---  ------               --------------   -----  
     0   Date                 133455 non-null  object 
     1   iso3                 133455 non-null  object 
     2   CountryName          133455 non-null  object 
     3   Region               133455 non-null  object 
     4   lat                  124740 non-null  float64
     5   lon                  124740 non-null  float64
     6   CumulativePositive   132317 non-null  float64
     7   CumulativeDeceased   106394 non-null  float64
     8   CumulativeRecovered  89866 non-null   float64
     9   CurrentlyPositive    133173 non-null  float64
     10  Hospitalized         60714 non-null   float64
     11  IntensiveCare        57952 non-null   float64
     12  EUcountry            133455 non-null  bool   
     13  EUCPMcountry         133455 non-null  bool   
     14  NUTS                 126379 non-null  object 
    dtypes: bool(2), float64(8), object(5)
    memory usage: 13.5+ MB
    


```python
df["Date"].max(), df["Date"].min()
```




    ('2021-01-29', '2020-01-22')



## Geodataframe


```python
df.dropna(axis=0, subset=["lat","lon"], inplace=True)
```


```python
crs="EPSG:4326"
```


```python
gdf = gpd.GeoDataFrame(df, crs=crs, geometry=gpd.points_from_xy(df.lon, df.lat))
gdf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>iso3</th>
      <th>CountryName</th>
      <th>Region</th>
      <th>lat</th>
      <th>lon</th>
      <th>CumulativePositive</th>
      <th>CumulativeDeceased</th>
      <th>CumulativeRecovered</th>
      <th>CurrentlyPositive</th>
      <th>Hospitalized</th>
      <th>IntensiveCare</th>
      <th>EUcountry</th>
      <th>EUCPMcountry</th>
      <th>NUTS</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>2020-01-22</td>
      <td>LIE</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>47.164696</td>
      <td>9.555000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>LI</td>
      <td>POINT (9.55500 47.16470)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-22</td>
      <td>MCO</td>
      <td>Monaco</td>
      <td>Monaco</td>
      <td>43.738348</td>
      <td>7.424451</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>MC</td>
      <td>POINT (7.42445 43.73835)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-22</td>
      <td>MKD</td>
      <td>North Macedonia</td>
      <td>North Macedonia</td>
      <td>41.611000</td>
      <td>21.751417</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>MK</td>
      <td>POINT (21.75142 41.61100)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-01-22</td>
      <td>SMR</td>
      <td>San Marino</td>
      <td>San Marino</td>
      <td>43.942973</td>
      <td>12.460035</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>SM</td>
      <td>POINT (12.46003 43.94297)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2020-01-22</td>
      <td>SRB</td>
      <td>Serbia</td>
      <td>Serbia</td>
      <td>44.206802</td>
      <td>20.911009</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>POINT (20.91101 44.20680)</td>
    </tr>
  </tbody>
</table>
</div>




```python
gdf.plot(figsize=(12,10));
```


    
![png](output_15_0.png)
    



```python
gdf[gdf["Date"] == '2020-03-30'].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>iso3</th>
      <th>CountryName</th>
      <th>Region</th>
      <th>lat</th>
      <th>lon</th>
      <th>CumulativePositive</th>
      <th>CumulativeDeceased</th>
      <th>CumulativeRecovered</th>
      <th>CurrentlyPositive</th>
      <th>Hospitalized</th>
      <th>IntensiveCare</th>
      <th>EUcountry</th>
      <th>EUCPMcountry</th>
      <th>NUTS</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6726</th>
      <td>2020-03-30</td>
      <td>ALB</td>
      <td>Albania</td>
      <td>Berat</td>
      <td>40.628500</td>
      <td>20.090775</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>AL031</td>
      <td>POINT (20.09078 40.62850)</td>
    </tr>
    <tr>
      <th>6727</th>
      <td>2020-03-30</td>
      <td>ALB</td>
      <td>Albania</td>
      <td>Durrës</td>
      <td>41.518365</td>
      <td>19.651766</td>
      <td>19.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>AL012</td>
      <td>POINT (19.65177 41.51837)</td>
    </tr>
    <tr>
      <th>6728</th>
      <td>2020-03-30</td>
      <td>ALB</td>
      <td>Albania</td>
      <td>Elbasan</td>
      <td>41.040028</td>
      <td>20.186454</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>AL021</td>
      <td>POINT (20.18645 41.04003)</td>
    </tr>
    <tr>
      <th>6729</th>
      <td>2020-03-30</td>
      <td>ALB</td>
      <td>Albania</td>
      <td>Fier</td>
      <td>40.774633</td>
      <td>19.620528</td>
      <td>27.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>AL032</td>
      <td>POINT (19.62053 40.77463)</td>
    </tr>
    <tr>
      <th>6730</th>
      <td>2020-03-30</td>
      <td>ALB</td>
      <td>Albania</td>
      <td>Korçë</td>
      <td>40.628830</td>
      <td>20.666004</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>AL034</td>
      <td>POINT (20.66600 40.62883)</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(14, 12))
gdf[gdf["Date"] == '2020-03-30'].plot(ax=ax, color="red", alpha=.4)
plt.title("Deceased Map -  2020-03-30", fontsize=30, fontname="Palatino Linotype", color="grey")

plt.show()
```


    
![png](output_17_0.png)
    


## Plotting Maps 


```python
fig, ax = plt.subplots(figsize=(14, 12))
gdf[gdf["Date"] == '2020-03-30'].to_crs(epsg=3857).plot(ax=ax, color="red", edgecolor="white")
ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite) 
plt.title("Deceased Map -  2020-03-30", fontsize=30, fontname="Palatino Linotype", color="grey")
ax.axis("off")
plt.show()
```


    
![png](output_19_0.png)
    



```python
fig, ax = plt.subplots(figsize=(14, 12))
gdf[gdf["EUCPMcountry"] == True].to_crs(epsg=3857).plot(ax=ax, color="red", edgecolor="white")
ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite) 
plt.title(" Covid-19 -  2021-01-01 ", fontsize=30, fontname="Palatino Linotype", color="grey")
ax.axis("off")
plt.show()
```


    
![png](output_20_0.png)
    



```python
fig, ax = plt.subplots(figsize=(14, 12))
gdf[gdf["Date"] == '2020-03-30'].to_crs(epsg=3857).plot(ax=ax, color="red", alpha=.4,  markersize="CumulativeDeceased")
ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite) 
plt.title("Deceased Bubble Map -  2020-03-30", fontsize=30, fontname="Palatino Linotype", color="grey")
ax.axis("off")
plt.show()
```


    
![png](output_21_0.png)
    



```python
fig, ax = plt.subplots(figsize=(14, 12))
gdf[gdf["Date"] == '2021-01-10'].to_crs(epsg=3857).plot(ax=ax, color="red", alpha=.4,  markersize="CumulativeDeceased")
ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite) 
plt.title(" Covid-19 -  2021-01-01 ", fontsize=30, fontname="Palatino Linotype", color="grey")
ax.axis("off")
plt.show()
```


    
![png](output_22_0.png)
    



```python
gdf["Normalized_mean_death"]=((gdf["CumulativeDeceased"]-gdf["CumulativeDeceased"].min())/(gdf["CumulativeDeceased"].max()-gdf["CumulativeDeceased"].min()))*100
```


```python
gdf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>iso3</th>
      <th>CountryName</th>
      <th>Region</th>
      <th>lat</th>
      <th>lon</th>
      <th>CumulativePositive</th>
      <th>CumulativeDeceased</th>
      <th>CumulativeRecovered</th>
      <th>CurrentlyPositive</th>
      <th>Hospitalized</th>
      <th>IntensiveCare</th>
      <th>EUcountry</th>
      <th>EUCPMcountry</th>
      <th>NUTS</th>
      <th>geometry</th>
      <th>Normalized_mean_death</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>2020-01-22</td>
      <td>LIE</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>47.164696</td>
      <td>9.555000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>LI</td>
      <td>POINT (9.55500 47.16470)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-22</td>
      <td>MCO</td>
      <td>Monaco</td>
      <td>Monaco</td>
      <td>43.738348</td>
      <td>7.424451</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>MC</td>
      <td>POINT (7.42445 43.73835)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-22</td>
      <td>MKD</td>
      <td>North Macedonia</td>
      <td>North Macedonia</td>
      <td>41.611000</td>
      <td>21.751417</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>MK</td>
      <td>POINT (21.75142 41.61100)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-01-22</td>
      <td>SMR</td>
      <td>San Marino</td>
      <td>San Marino</td>
      <td>43.942973</td>
      <td>12.460035</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>SM</td>
      <td>POINT (12.46003 43.94297)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2020-01-22</td>
      <td>SRB</td>
      <td>Serbia</td>
      <td>Serbia</td>
      <td>44.206802</td>
      <td>20.911009</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>POINT (20.91101 44.20680)</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(14, 12))
gdf[gdf["Date"] == '2021-01-10'].to_crs(epsg=3857).plot(ax=ax, color="red", alpha=.7,  markersize="Normalized_mean_death")
ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite) 
plt.title(" Covid-19 -  2021-01-01 ", fontsize=30, fontname="Palatino Linotype", color="grey")
#ax.axis("off")
plt.show()
```


    
![png](output_25_0.png)
    


## Join & Merge


```python
eu_lv2 = gpd.read_file("NUTS_RG_01M_2021_4326_LEVL_2.geojson")
eu_lv2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>NUTS_ID</th>
      <th>LEVL_CODE</th>
      <th>CNTR_CODE</th>
      <th>NAME_LATN</th>
      <th>NUTS_NAME</th>
      <th>MOUNT_TYPE</th>
      <th>URBN_TYPE</th>
      <th>COAST_TYPE</th>
      <th>FID</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FRB0</td>
      <td>FRB0</td>
      <td>2</td>
      <td>FR</td>
      <td>Centre — Val de Loire</td>
      <td>Centre — Val de Loire</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>FRB0</td>
      <td>POLYGON ((1.50153 48.94105, 1.51118 48.93461, ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CZ04</td>
      <td>CZ04</td>
      <td>2</td>
      <td>CZ</td>
      <td>Severozápad</td>
      <td>Severozápad</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>CZ04</td>
      <td>POLYGON ((14.49122 51.04353, 14.49945 51.04610...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CZ07</td>
      <td>CZ07</td>
      <td>2</td>
      <td>CZ</td>
      <td>Střední Morava</td>
      <td>Střední Morava</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>CZ07</td>
      <td>POLYGON ((16.90792 50.44945, 16.92475 50.43939...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DK02</td>
      <td>DK02</td>
      <td>2</td>
      <td>DK</td>
      <td>Sjælland</td>
      <td>Sjælland</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>DK02</td>
      <td>MULTIPOLYGON (((11.77939 55.65903, 11.78305 55...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ES12</td>
      <td>ES12</td>
      <td>2</td>
      <td>ES</td>
      <td>Principado de Asturias</td>
      <td>Principado de Asturias</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ES12</td>
      <td>MULTIPOLYGON (((-4.51230 43.39320, -4.51142 43...</td>
    </tr>
  </tbody>
</table>
</div>




```python
eu_lv2.crs
```




    <Geographic 2D CRS: EPSG:4326>
    Name: WGS 84
    Axis Info [ellipsoidal]:
    - Lat[north]: Geodetic latitude (degree)
    - Lon[east]: Geodetic longitude (degree)
    Area of Use:
    - name: World
    - bounds: (-180.0, -90.0, 180.0, 90.0)
    Datum: World Geodetic System 1984
    - Ellipsoid: WGS 84
    - Prime Meridian: Greenwich




```python
eu_lv2.plot("NUTS_NAME", figsize=(12,10));
```


    
![png](output_29_0.png)
    



```python
eu_lv2.plot("CNTR_CODE", figsize=(12,10));
```


    
![png](output_30_0.png)
    



```python
np.setdiff1d(list(gdf.NUTS.unique()), list(eu_lv2.NUTS_ID.unique()))
```




    array(['AD', 'AL012', 'AL013', 'AL014', 'AL015', 'AL021', 'AL022',
           'AL031', 'AL032', 'AL034', 'AL035', 'BA', 'BE1', 'BE2', 'BE3',
           'BG311', 'BG312', 'BG313', 'BG314', 'BG315', 'BG321', 'BG322',
           'BG323', 'BG325', 'BG331', 'BG332', 'BG333', 'BG341', 'BG342',
           'BG344', 'BG411', 'BG413', 'BG414', 'BG415', 'BG421', 'BG422',
           'BG423', 'BG424', 'BG425', 'BY', 'CH011', 'CH012', 'CH013',
           'CH021', 'CH022', 'CH023', 'CH024', 'CH025', 'CH031', 'CH032',
           'CH033', 'CH040', 'CH051', 'CH052', 'CH053', 'CH054', 'CH055',
           'CH056', 'CH057', 'CH061', 'CH062', 'CH063', 'CH064', 'CH065',
           'CH066', 'CY', 'DE1', 'DE2', 'DE3', 'DE4', 'DE5', 'DE6', 'DE7',
           'DE8', 'DE9', 'DEA', 'DEB', 'DEC', 'DED', 'DEE', 'DEF', 'DEG',
           'EE0', 'FI193', 'FI194', 'FI195', 'FI196', 'FI197', 'FI1B1',
           'FI1C1', 'FI1C2', 'FI1C3', 'FI1C4', 'FI1C5', 'FI1D1', 'FI1D2',
           'FI1D3', 'FI1D5', 'FI1D7', 'FI1D8', 'FI1D9', 'FI1X1', 'FI1X2',
           'FI200', 'FR1', 'FRC', 'FRD', 'FRE', 'FRF', 'FRI', 'FRJ', 'FRK',
           'FRL', 'FRO', 'GRL', 'HR031', 'HR032', 'HR033', 'HR034', 'HR035',
           'HR036', 'HR037', 'HR041', 'HR042', 'HR043', 'HR044', 'HR045',
           'HR046', 'HR047', 'HR048', 'HR049', 'HR04A', 'HR04B', 'HR04C',
           'HR04D', 'HR04E', 'IE', 'LI', 'LT', 'LU', 'LV0', 'MC', 'MD', 'ME',
           'MK', 'MT001', 'NO011', 'NO012', 'NO033', 'NO04', 'NO043', 'NO05',
           'NO053', 'NO071', 'NO072', 'RO111', 'RO112', 'RO113', 'RO114',
           'RO115', 'RO116', 'RO121', 'RO122', 'RO123', 'RO124', 'RO125',
           'RO126', 'RO211', 'RO212', 'RO213', 'RO214', 'RO215', 'RO216',
           'RO221', 'RO222', 'RO223', 'RO224', 'RO225', 'RO226', 'RO311',
           'RO312', 'RO313', 'RO314', 'RO315', 'RO316', 'RO317', 'RO321',
           'RO322', 'RO411', 'RO412', 'RO413', 'RO414', 'RO415', 'RO421',
           'RO422', 'RO423', 'RO424', 'RU01', 'RU02', 'RU03', 'RU04', 'RU05',
           'RU06', 'RU07', 'RU08', 'RU09', 'RU10', 'RU11', 'RU12', 'RU13',
           'RU14', 'RU15', 'RU16', 'RU17', 'RU18', 'RU19', 'RU20', 'RU21',
           'RU22', 'RU23', 'RU24', 'RU25', 'RU26', 'RU27', 'RU28', 'RU29',
           'RU30', 'RU31', 'RU32', 'RU33', 'RU34', 'RU35', 'RU36', 'RU37',
           'RU38', 'RU39', 'RU40', 'RU41', 'RU42', 'RU43', 'RU44', 'RU45',
           'RU46', 'RU47', 'RU48', 'RU49', 'RU50', 'RU51', 'RU52', 'RU53',
           'RU54', 'RU55', 'RU56', 'RU57', 'RU58', 'RU59', 'RU60', 'RU61',
           'RU62', 'RU63', 'RU64', 'RU65', 'RU66', 'RU67', 'RU68', 'RU69',
           'RU70', 'RU71', 'RU72', 'RU73', 'RU74', 'RU75', 'RU76', 'RU77',
           'RU78', 'RU79', 'RU80', 'SE121', 'SE122', 'SE123', 'SE124',
           'SE125', 'SE211', 'SE212', 'SE213', 'SE214', 'SE221', 'SE224',
           'SE231', 'SE232', 'SE311', 'SE312', 'SE313', 'SE321', 'SE322',
           'SE331', 'SE332', 'SI031', 'SI032', 'SI033', 'SI034', 'SI035',
           'SI036', 'SI037', 'SI038', 'SI041', 'SI042', 'SI043', 'SI044',
           'SK010', 'SK021', 'SK022', 'SK023', 'SK031', 'SK032', 'SK041',
           'SK042', 'SM', 'TU', 'UKC', 'UKD', 'UKF', 'UKH', 'UKI', 'UKJ',
           'UKK', 'UKL', 'UKM', 'UKM50', 'UKM64', 'UKM65', 'UKM66', 'UKM71',
           'UKM72', 'UKM75', 'UKM76', 'UKM82', 'UKM91', 'UKM92', 'UKM93',
           'UKN', 'XK', 'nan'], dtype='<U5')




```python
fig, ax = plt.subplots(figsize=(14, 12))

eu_lv2.plot("NUTS_NAME", ax=ax)
gdf.plot(ax=ax, color="red", edgecolor="white", markersize=10)
plt.title("Level 2 Boundaries and Points", fontsize=30, fontname="Palatino Linotype", color="grey")
ax.axis("off")
plt.show()
```


    
![png](output_32_0.png)
    



```python
fig, ax = plt.subplots(figsize=(14, 12))

eu_lv2.cx[-20:60, 30:70].plot("NUTS_NAME", ax=ax)
gdf.cx[-20:60, 30:70].plot(ax=ax, color="black", edgecolor="white")
plt.title("Level 2 Boundaries Zoomed and Points", fontsize=30, fontname="Palatino Linotype", color="grey")
ax.axis("off")
plt.show()
```


    
![png](output_33_0.png)
    



```python
sjoined = gpd.sjoin(gdf, eu_lv2, op="within")
```


```python
sjoined.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>iso3</th>
      <th>CountryName</th>
      <th>Region</th>
      <th>lat</th>
      <th>lon</th>
      <th>CumulativePositive</th>
      <th>CumulativeDeceased</th>
      <th>CumulativeRecovered</th>
      <th>CurrentlyPositive</th>
      <th>...</th>
      <th>id</th>
      <th>NUTS_ID</th>
      <th>LEVL_CODE</th>
      <th>CNTR_CODE</th>
      <th>NAME_LATN</th>
      <th>NUTS_NAME</th>
      <th>MOUNT_TYPE</th>
      <th>URBN_TYPE</th>
      <th>COAST_TYPE</th>
      <th>FID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>2020-01-22</td>
      <td>LIE</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>47.164696</td>
      <td>9.555</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>LI00</td>
      <td>LI00</td>
      <td>2</td>
      <td>LI</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>LI00</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2020-01-23</td>
      <td>LIE</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>47.164696</td>
      <td>9.555</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>LI00</td>
      <td>LI00</td>
      <td>2</td>
      <td>LI</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>LI00</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2020-01-24</td>
      <td>LIE</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>47.164696</td>
      <td>9.555</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>LI00</td>
      <td>LI00</td>
      <td>2</td>
      <td>LI</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>LI00</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2020-01-25</td>
      <td>LIE</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>47.164696</td>
      <td>9.555</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>LI00</td>
      <td>LI00</td>
      <td>2</td>
      <td>LI</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>LI00</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2020-01-26</td>
      <td>LIE</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>47.164696</td>
      <td>9.555</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>LI00</td>
      <td>LI00</td>
      <td>2</td>
      <td>LI</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>LI00</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
sjoined.head().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2</th>
      <th>9</th>
      <th>18</th>
      <th>28</th>
      <th>38</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Date</th>
      <td>2020-01-22</td>
      <td>2020-01-23</td>
      <td>2020-01-24</td>
      <td>2020-01-25</td>
      <td>2020-01-26</td>
    </tr>
    <tr>
      <th>iso3</th>
      <td>LIE</td>
      <td>LIE</td>
      <td>LIE</td>
      <td>LIE</td>
      <td>LIE</td>
    </tr>
    <tr>
      <th>CountryName</th>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
    </tr>
    <tr>
      <th>Region</th>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
    </tr>
    <tr>
      <th>lat</th>
      <td>47.164696</td>
      <td>47.164696</td>
      <td>47.164696</td>
      <td>47.164696</td>
      <td>47.164696</td>
    </tr>
    <tr>
      <th>lon</th>
      <td>9.555000</td>
      <td>9.555000</td>
      <td>9.555000</td>
      <td>9.555000</td>
      <td>9.555000</td>
    </tr>
    <tr>
      <th>CumulativePositive</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>CumulativeDeceased</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>CumulativeRecovered</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>CurrentlyPositive</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Hospitalized</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>IntensiveCare</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>EUcountry</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>EUCPMcountry</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>NUTS</th>
      <td>LI</td>
      <td>LI</td>
      <td>LI</td>
      <td>LI</td>
      <td>LI</td>
    </tr>
    <tr>
      <th>geometry</th>
      <td>POINT (9.555 47.164696)</td>
      <td>POINT (9.555 47.164696)</td>
      <td>POINT (9.555 47.164696)</td>
      <td>POINT (9.555 47.164696)</td>
      <td>POINT (9.555 47.164696)</td>
    </tr>
    <tr>
      <th>Normalized_mean_death</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>index_right</th>
      <td>162</td>
      <td>162</td>
      <td>162</td>
      <td>162</td>
      <td>162</td>
    </tr>
    <tr>
      <th>id</th>
      <td>LI00</td>
      <td>LI00</td>
      <td>LI00</td>
      <td>LI00</td>
      <td>LI00</td>
    </tr>
    <tr>
      <th>NUTS_ID</th>
      <td>LI00</td>
      <td>LI00</td>
      <td>LI00</td>
      <td>LI00</td>
      <td>LI00</td>
    </tr>
    <tr>
      <th>LEVL_CODE</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>CNTR_CODE</th>
      <td>LI</td>
      <td>LI</td>
      <td>LI</td>
      <td>LI</td>
      <td>LI</td>
    </tr>
    <tr>
      <th>NAME_LATN</th>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
    </tr>
    <tr>
      <th>NUTS_NAME</th>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
    </tr>
    <tr>
      <th>MOUNT_TYPE</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>URBN_TYPE</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>COAST_TYPE</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>FID</th>
      <td>LI00</td>
      <td>LI00</td>
      <td>LI00</td>
      <td>LI00</td>
      <td>LI00</td>
    </tr>
  </tbody>
</table>
</div>




```python
sjoined_to_merge = sjoined[["Date", "CumulativePositive", "CumulativeDeceased", "CumulativeRecovered", "NUTS_ID"]]
```


```python
sjoined_to_merge[sjoined["NUTS_NAME"] == "Abruzzo"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>CumulativePositive</th>
      <th>CumulativeDeceased</th>
      <th>CumulativeRecovered</th>
      <th>NUTS_ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>409</th>
      <td>2020-02-27</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>ITF1</td>
    </tr>
    <tr>
      <th>454</th>
      <td>2020-02-28</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>ITF1</td>
    </tr>
    <tr>
      <th>506</th>
      <td>2020-02-29</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>ITF1</td>
    </tr>
    <tr>
      <th>580</th>
      <td>2020-03-01</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>ITF1</td>
    </tr>
    <tr>
      <th>672</th>
      <td>2020-03-02</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>ITF1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>131465</th>
      <td>2021-01-24</td>
      <td>40889.0</td>
      <td>1393.0</td>
      <td>28842.0</td>
      <td>ITF1</td>
    </tr>
    <tr>
      <th>131856</th>
      <td>2021-01-25</td>
      <td>40955.0</td>
      <td>1418.0</td>
      <td>29341.0</td>
      <td>ITF1</td>
    </tr>
    <tr>
      <th>132303</th>
      <td>2021-01-26</td>
      <td>41107.0</td>
      <td>1435.0</td>
      <td>29625.0</td>
      <td>ITF1</td>
    </tr>
    <tr>
      <th>132748</th>
      <td>2021-01-27</td>
      <td>41450.0</td>
      <td>1441.0</td>
      <td>29933.0</td>
      <td>ITF1</td>
    </tr>
    <tr>
      <th>133157</th>
      <td>2021-01-28</td>
      <td>41718.0</td>
      <td>1446.0</td>
      <td>30206.0</td>
      <td>ITF1</td>
    </tr>
  </tbody>
</table>
<p>337 rows × 5 columns</p>
</div>




```python
sjoined_to_merge[sjoined_to_merge["Date"] == "2021-01-26"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>CumulativePositive</th>
      <th>CumulativeDeceased</th>
      <th>CumulativeRecovered</th>
      <th>NUTS_ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>132325</th>
      <td>2021-01-26</td>
      <td>2455.0</td>
      <td>52.0</td>
      <td>2322.0</td>
      <td>LI00</td>
    </tr>
    <tr>
      <th>132555</th>
      <td>2021-01-26</td>
      <td>2455.0</td>
      <td>52.0</td>
      <td>2322.0</td>
      <td>LI00</td>
    </tr>
    <tr>
      <th>132344</th>
      <td>2021-01-26</td>
      <td>91161.0</td>
      <td>2812.0</td>
      <td>79621.0</td>
      <td>MK00</td>
    </tr>
    <tr>
      <th>132492</th>
      <td>2021-01-26</td>
      <td>387206.0</td>
      <td>3924.0</td>
      <td>0.0</td>
      <td>RS21</td>
    </tr>
    <tr>
      <th>132272</th>
      <td>2021-01-26</td>
      <td>0.0</td>
      <td>13153.0</td>
      <td>58777.0</td>
      <td>FR10</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>132385</th>
      <td>2021-01-26</td>
      <td>5614.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>RO41</td>
    </tr>
    <tr>
      <th>132392</th>
      <td>2021-01-26</td>
      <td>5494.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>RO41</td>
    </tr>
    <tr>
      <th>132396</th>
      <td>2021-01-26</td>
      <td>10421.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>RO41</td>
    </tr>
    <tr>
      <th>132405</th>
      <td>2021-01-26</td>
      <td>10899.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>RO41</td>
    </tr>
    <tr>
      <th>132331</th>
      <td>2021-01-26</td>
      <td>59262.0</td>
      <td>777.0</td>
      <td>NaN</td>
      <td>ME00</td>
    </tr>
  </tbody>
</table>
<p>330 rows × 5 columns</p>
</div>




```python
merged_gdf = pd.merge(eu_lv2,sjoined_to_merge[sjoined_to_merge["Date"] == "2021-01-26"], on="NUTS_ID", how="inner")
#eu_lv2.merge(sjoined_to_merge[sjoined_to_merge["Date"] == "2021-01-26"], on="NUTS_ID", how="left")
merged_gdf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>NUTS_ID</th>
      <th>LEVL_CODE</th>
      <th>CNTR_CODE</th>
      <th>NAME_LATN</th>
      <th>NUTS_NAME</th>
      <th>MOUNT_TYPE</th>
      <th>URBN_TYPE</th>
      <th>COAST_TYPE</th>
      <th>FID</th>
      <th>geometry</th>
      <th>Date</th>
      <th>CumulativePositive</th>
      <th>CumulativeDeceased</th>
      <th>CumulativeRecovered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FRB0</td>
      <td>FRB0</td>
      <td>2</td>
      <td>FR</td>
      <td>Centre — Val de Loire</td>
      <td>Centre — Val de Loire</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>FRB0</td>
      <td>POLYGON ((1.50153 48.94105, 1.51118 48.93461, ...</td>
      <td>2021-01-26</td>
      <td>0.0</td>
      <td>1603.0</td>
      <td>6351.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DK02</td>
      <td>DK02</td>
      <td>2</td>
      <td>DK</td>
      <td>Sjælland</td>
      <td>Sjælland</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>DK02</td>
      <td>MULTIPOLYGON (((11.77939 55.65903, 11.78305 55...</td>
      <td>2021-01-26</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ES12</td>
      <td>ES12</td>
      <td>2</td>
      <td>ES</td>
      <td>Principado de Asturias</td>
      <td>Principado de Asturias</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ES12</td>
      <td>MULTIPOLYGON (((-4.51230 43.39320, -4.51142 43...</td>
      <td>2021-01-26</td>
      <td>34214.0</td>
      <td>1475.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AL01</td>
      <td>AL01</td>
      <td>2</td>
      <td>AL</td>
      <td>Veri</td>
      <td>Veri</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>AL01</td>
      <td>POLYGON ((19.83100 42.46645, 19.83568 42.47103...</td>
      <td>2021-01-26</td>
      <td>6036.0</td>
      <td>138.0</td>
      <td>4399.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AL01</td>
      <td>AL01</td>
      <td>2</td>
      <td>AL</td>
      <td>Veri</td>
      <td>Veri</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>AL01</td>
      <td>POLYGON ((19.83100 42.46645, 19.83568 42.47103...</td>
      <td>2021-01-26</td>
      <td>1678.0</td>
      <td>30.0</td>
      <td>1228.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged_gdf.shape, eu_lv2.shape
```




    ((330, 15), (334, 11))




```python
eu_lv2.plot()
```




    <AxesSubplot:>




    
![png](output_42_1.png)
    



```python
merged_gdf.plot("CumulativePositive", legend=True, figsize=(20,18));
```


    
![png](output_43_0.png)
    



```python
merged_gdf.cx[-20:60, 30:70].plot("CumulativeDeceased", legend=True, figsize=(20,18), legend_kwds={'format':"%.0f"});
```


    
![png](output_44_0.png)
    


## Choropleth Map


```python
merged_gdf["CumulativePositive"].min(), merged_gdf["CumulativeDeceased"].max()
```




    (0.0, 26789.0)




```python
population = pd.read_csv("population_data.csv")
population.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>codes</th>
      <th>labels</th>
      <th>2019</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BE</td>
      <td>Belgium</td>
      <td>11455519</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BE1</td>
      <td>Région de Bruxelles-Capitale/Brussels Hoofdste...</td>
      <td>1215290</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BE10</td>
      <td>Région de Bruxelles-Capitale/Brussels Hoofdste...</td>
      <td>1215290</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BE2</td>
      <td>Vlaams Gewest</td>
      <td>6596233</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BE21</td>
      <td>Prov. Antwerpen</td>
      <td>1860470</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged_gdf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>NUTS_ID</th>
      <th>LEVL_CODE</th>
      <th>CNTR_CODE</th>
      <th>NAME_LATN</th>
      <th>NUTS_NAME</th>
      <th>MOUNT_TYPE</th>
      <th>URBN_TYPE</th>
      <th>COAST_TYPE</th>
      <th>FID</th>
      <th>geometry</th>
      <th>Date</th>
      <th>CumulativePositive</th>
      <th>CumulativeDeceased</th>
      <th>CumulativeRecovered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FRB0</td>
      <td>FRB0</td>
      <td>2</td>
      <td>FR</td>
      <td>Centre — Val de Loire</td>
      <td>Centre — Val de Loire</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>FRB0</td>
      <td>POLYGON ((1.50153 48.94105, 1.51118 48.93461, ...</td>
      <td>2021-01-26</td>
      <td>0.0</td>
      <td>1603.0</td>
      <td>6351.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DK02</td>
      <td>DK02</td>
      <td>2</td>
      <td>DK</td>
      <td>Sjælland</td>
      <td>Sjælland</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>DK02</td>
      <td>MULTIPOLYGON (((11.77939 55.65903, 11.78305 55...</td>
      <td>2021-01-26</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ES12</td>
      <td>ES12</td>
      <td>2</td>
      <td>ES</td>
      <td>Principado de Asturias</td>
      <td>Principado de Asturias</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ES12</td>
      <td>MULTIPOLYGON (((-4.51230 43.39320, -4.51142 43...</td>
      <td>2021-01-26</td>
      <td>34214.0</td>
      <td>1475.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AL01</td>
      <td>AL01</td>
      <td>2</td>
      <td>AL</td>
      <td>Veri</td>
      <td>Veri</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>AL01</td>
      <td>POLYGON ((19.83100 42.46645, 19.83568 42.47103...</td>
      <td>2021-01-26</td>
      <td>6036.0</td>
      <td>138.0</td>
      <td>4399.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AL01</td>
      <td>AL01</td>
      <td>2</td>
      <td>AL</td>
      <td>Veri</td>
      <td>Veri</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>AL01</td>
      <td>POLYGON ((19.83100 42.46645, 19.83568 42.47103...</td>
      <td>2021-01-26</td>
      <td>1678.0</td>
      <td>30.0</td>
      <td>1228.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged_population = merged_gdf.merge(population, left_on="NUTS_ID", right_on="codes", how="left")
```


```python
merged_population.shape
```




    (330, 18)




```python
merged_population['CumulativeDeceased'] = merged_population['CumulativeDeceased'].fillna(0)
```


```python
merged_population["normalized_deceased"] = (merged_population["CumulativeDeceased"] / merged_population["2019"].astype("float")) * 100
```


```python
merged_population.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>NUTS_ID</th>
      <th>LEVL_CODE</th>
      <th>CNTR_CODE</th>
      <th>NAME_LATN</th>
      <th>NUTS_NAME</th>
      <th>MOUNT_TYPE</th>
      <th>URBN_TYPE</th>
      <th>COAST_TYPE</th>
      <th>FID</th>
      <th>geometry</th>
      <th>Date</th>
      <th>CumulativePositive</th>
      <th>CumulativeDeceased</th>
      <th>CumulativeRecovered</th>
      <th>codes</th>
      <th>labels</th>
      <th>2019</th>
      <th>normalized_deceased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FRB0</td>
      <td>FRB0</td>
      <td>2</td>
      <td>FR</td>
      <td>Centre — Val de Loire</td>
      <td>Centre — Val de Loire</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>FRB0</td>
      <td>POLYGON ((1.50153 48.94105, 1.51118 48.93461, ...</td>
      <td>2021-01-26</td>
      <td>0.0</td>
      <td>1603.0</td>
      <td>6351.0</td>
      <td>FRB0</td>
      <td>Centre - Val de Loire</td>
      <td>2565258.0</td>
      <td>0.062489</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DK02</td>
      <td>DK02</td>
      <td>2</td>
      <td>DK</td>
      <td>Sjælland</td>
      <td>Sjælland</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>DK02</td>
      <td>MULTIPOLYGON (((11.77939 55.65903, 11.78305 55...</td>
      <td>2021-01-26</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>DK02</td>
      <td>Sjælland</td>
      <td>836738.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ES12</td>
      <td>ES12</td>
      <td>2</td>
      <td>ES</td>
      <td>Principado de Asturias</td>
      <td>Principado de Asturias</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ES12</td>
      <td>MULTIPOLYGON (((-4.51230 43.39320, -4.51142 43...</td>
      <td>2021-01-26</td>
      <td>34214.0</td>
      <td>1475.0</td>
      <td>NaN</td>
      <td>ES12</td>
      <td>Principado de Asturias</td>
      <td>1022205.0</td>
      <td>0.144296</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AL01</td>
      <td>AL01</td>
      <td>2</td>
      <td>AL</td>
      <td>Veri</td>
      <td>Veri</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>AL01</td>
      <td>POLYGON ((19.83100 42.46645, 19.83568 42.47103...</td>
      <td>2021-01-26</td>
      <td>6036.0</td>
      <td>138.0</td>
      <td>4399.0</td>
      <td>AL01</td>
      <td>Veri</td>
      <td>813758.0</td>
      <td>0.016958</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AL01</td>
      <td>AL01</td>
      <td>2</td>
      <td>AL</td>
      <td>Veri</td>
      <td>Veri</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>AL01</td>
      <td>POLYGON ((19.83100 42.46645, 19.83568 42.47103...</td>
      <td>2021-01-26</td>
      <td>1678.0</td>
      <td>30.0</td>
      <td>1228.0</td>
      <td>AL01</td>
      <td>Veri</td>
      <td>813758.0</td>
      <td>0.003687</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged_population.cx[-20:60, 30:70].plot("CumulativeDeceased", legend=True, figsize=(20,18), legend_kwds={'format':"%.0f"});
```


    
![png](output_54_0.png)
    



```python
merged_population.cx[-20:60, 30:70].plot("normalized_deceased", legend=True, figsize=(20,18));
```


    
![png](output_55_0.png)
    


## Interactive 


```python
import pandas as pd
import holoviews as hv

from bokeh.sampledata import stocks
from holoviews.operation.timeseries import rolling, rolling_outlier_std

hv.extension('bokeh')
```







<div class="logo-block">
<img src='data:image/png;base64,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
roTBs2RqAAAAAElFTkSuQmCC'
     style='height:25px; border-radius:12px; display: inline-block; float: left; vertical-align: middle'></img>


  <img src='data:image/png;base64,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'
       style='height:15px; border-radius:12px; display: inline-block; float: left'></img>





</div>




```python
sjoined.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>iso3</th>
      <th>CountryName</th>
      <th>Region</th>
      <th>lat</th>
      <th>lon</th>
      <th>CumulativePositive</th>
      <th>CumulativeDeceased</th>
      <th>CumulativeRecovered</th>
      <th>CurrentlyPositive</th>
      <th>...</th>
      <th>id</th>
      <th>NUTS_ID</th>
      <th>LEVL_CODE</th>
      <th>CNTR_CODE</th>
      <th>NAME_LATN</th>
      <th>NUTS_NAME</th>
      <th>MOUNT_TYPE</th>
      <th>URBN_TYPE</th>
      <th>COAST_TYPE</th>
      <th>FID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>2020-01-22</td>
      <td>LIE</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>47.164696</td>
      <td>9.555</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>LI00</td>
      <td>LI00</td>
      <td>2</td>
      <td>LI</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>LI00</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2020-01-23</td>
      <td>LIE</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>47.164696</td>
      <td>9.555</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>LI00</td>
      <td>LI00</td>
      <td>2</td>
      <td>LI</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>LI00</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2020-01-24</td>
      <td>LIE</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>47.164696</td>
      <td>9.555</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>LI00</td>
      <td>LI00</td>
      <td>2</td>
      <td>LI</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>LI00</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2020-01-25</td>
      <td>LIE</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>47.164696</td>
      <td>9.555</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>LI00</td>
      <td>LI00</td>
      <td>2</td>
      <td>LI</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>LI00</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2020-01-26</td>
      <td>LIE</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>47.164696</td>
      <td>9.555</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>LI00</td>
      <td>LI00</td>
      <td>2</td>
      <td>LI</td>
      <td>Liechtenstein</td>
      <td>Liechtenstein</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>LI00</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
sample_dates = list(sjoined.Date.sample(10))
```


```python
def get_date(symbol, **kwargs):
    gdf = sjoined
    return gdf.hvplot(geo=True, color="red",  tiles="ESRI")
```


```python
dmap = hv.DynamicMap(get_date, kdims='Symbol').redim.values(Symbol=sample_dates)
```


```python
dmap
```

    WARNING:param.get_date: Callable raised "AttributeError("'GeoDataFrame' object has no attribute 'hvplot'")".
    Invoked as get_date(symbol='2020-12-05')
    WARNING:param.dynamic_operation: Callable raised "AttributeError("'GeoDataFrame' object has no attribute 'hvplot'")".
    Invoked as dynamic_operation('2020-12-05')
    


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    ~\anaconda3\envs\geospatial\lib\site-packages\IPython\core\formatters.py in __call__(self, obj, include, exclude)
        968 
        969             if method is not None:
    --> 970                 return method(include=include, exclude=exclude)
        971             return None
        972         else:
    

    ~\anaconda3\envs\geospatial\lib\site-packages\holoviews\core\dimension.py in _repr_mimebundle_(self, include, exclude)
       1314         combined and returned.
       1315         """
    -> 1316         return Store.render(self)
       1317 
       1318 
    

    ~\anaconda3\envs\geospatial\lib\site-packages\holoviews\core\options.py in render(cls, obj)
       1403         data, metadata = {}, {}
       1404         for hook in hooks:
    -> 1405             ret = hook(obj)
       1406             if ret is None:
       1407                 continue
    

    ~\anaconda3\envs\geospatial\lib\site-packages\holoviews\ipython\display_hooks.py in pprint_display(obj)
        280     if not ip.display_formatter.formatters['text/plain'].pprint:
        281         return None
    --> 282     return display(obj, raw_output=True)
        283 
        284 
    

    ~\anaconda3\envs\geospatial\lib\site-packages\holoviews\ipython\display_hooks.py in display(obj, raw_output, **kwargs)
        256     elif isinstance(obj, (HoloMap, DynamicMap)):
        257         with option_state(obj):
    --> 258             output = map_display(obj)
        259     elif isinstance(obj, Plot):
        260         output = render(obj)
    

    ~\anaconda3\envs\geospatial\lib\site-packages\holoviews\ipython\display_hooks.py in wrapped(element)
        144         try:
        145             max_frames = OutputSettings.options['max_frames']
    --> 146             mimebundle = fn(element, max_frames=max_frames)
        147             if mimebundle is None:
        148                 return {}, {}
    

    ~\anaconda3\envs\geospatial\lib\site-packages\holoviews\ipython\display_hooks.py in map_display(vmap, max_frames)
        204         return None
        205 
    --> 206     return render(vmap)
        207 
        208 
    

    ~\anaconda3\envs\geospatial\lib\site-packages\holoviews\ipython\display_hooks.py in render(obj, **kwargs)
         66         renderer = renderer.instance(fig='png')
         67 
    ---> 68     return renderer.components(obj, **kwargs)
         69 
         70 
    

    ~\anaconda3\envs\geospatial\lib\site-packages\holoviews\plotting\renderer.py in components(self, obj, fmt, comm, **kwargs)
        408                 doc = Document()
        409                 with config.set(embed=embed):
    --> 410                     model = plot.layout._render_model(doc, comm)
        411                 if embed:
        412                     return render_model(model, comm)
    

    ~\anaconda3\envs\geospatial\lib\site-packages\panel\viewable.py in _render_model(self, doc, comm)
        422         if comm is None:
        423             comm = state._comm_manager.get_server_comm()
    --> 424         model = self.get_root(doc, comm)
        425 
        426         if config.embed:
    

    ~\anaconda3\envs\geospatial\lib\site-packages\panel\viewable.py in get_root(self, doc, comm, preprocess)
        480         """
        481         doc = init_doc(doc)
    --> 482         root = self._get_model(doc, comm=comm)
        483         if preprocess:
        484             self._preprocess(root)
    

    ~\anaconda3\envs\geospatial\lib\site-packages\panel\layout\base.py in _get_model(self, doc, root, parent, comm)
        110         if root is None:
        111             root = model
    --> 112         objects = self._get_objects(model, [], doc, root, comm)
        113         props = dict(self._init_properties(), objects=objects)
        114         model.update(**self._process_param_change(props))
    

    ~\anaconda3\envs\geospatial\lib\site-packages\panel\layout\base.py in _get_objects(self, model, old_objects, doc, root, comm)
        100             else:
        101                 try:
    --> 102                     child = pane._get_model(doc, root, model, comm)
        103                 except RerenderError:
        104                     return self._get_objects(model, current_objects[:i], doc, root, comm)
    

    ~\anaconda3\envs\geospatial\lib\site-packages\panel\pane\holoviews.py in _get_model(self, doc, root, parent, comm)
        239             plot = self.object
        240         else:
    --> 241             plot = self._render(doc, comm, root)
        242 
        243         plot.pane = self
    

    ~\anaconda3\envs\geospatial\lib\site-packages\panel\pane\holoviews.py in _render(self, doc, comm, root)
        304                 kwargs['comm'] = comm
        305 
    --> 306         return renderer.get_plot(self.object, **kwargs)
        307 
        308     def _cleanup(self, root):
    

    ~\anaconda3\envs\geospatial\lib\site-packages\holoviews\plotting\bokeh\renderer.py in get_plot(self_or_cls, obj, doc, renderer, **kwargs)
         71         combining the bokeh model with another plot.
         72         """
    ---> 73         plot = super(BokehRenderer, self_or_cls).get_plot(obj, doc, renderer, **kwargs)
         74         if plot.document is None:
         75             plot.document = Document() if self_or_cls.notebook_context else curdoc()
    

    ~\anaconda3\envs\geospatial\lib\site-packages\holoviews\plotting\renderer.py in get_plot(self_or_cls, obj, doc, renderer, comm, **kwargs)
        218 
        219         # Initialize DynamicMaps with first data item
    --> 220         initialize_dynamic(obj)
        221 
        222         if not renderer:
    

    ~\anaconda3\envs\geospatial\lib\site-packages\holoviews\plotting\util.py in initialize_dynamic(obj)
        250             continue
        251         if not len(dmap):
    --> 252             dmap[dmap._initial_key()]
        253 
        254 
    

    ~\anaconda3\envs\geospatial\lib\site-packages\holoviews\core\spaces.py in __getitem__(self, key)
       1329         # Not a cross product and nothing cached so compute element.
       1330         if cache is not None: return cache
    -> 1331         val = self._execute_callback(*tuple_key)
       1332         if data_slice:
       1333             val = self._dataslice(val, data_slice)
    

    ~\anaconda3\envs\geospatial\lib\site-packages\holoviews\core\spaces.py in _execute_callback(self, *args)
       1098 
       1099         with dynamicmap_memoization(self.callback, self.streams):
    -> 1100             retval = self.callback(*args, **kwargs)
       1101         return self._style(retval)
       1102 
    

    ~\anaconda3\envs\geospatial\lib\site-packages\holoviews\core\spaces.py in __call__(self, *args, **kwargs)
        712 
        713         try:
    --> 714             ret = self.callable(*args, **kwargs)
        715         except KeyError:
        716             # KeyError is caught separately because it is used to signal
    

    ~\anaconda3\envs\geospatial\lib\site-packages\holoviews\util\__init__.py in dynamic_operation(*key, **kwargs)
       1016 
       1017         def dynamic_operation(*key, **kwargs):
    -> 1018             key, obj = resolve(key, kwargs)
       1019             return apply(obj, *key, **kwargs)
       1020 
    

    ~\anaconda3\envs\geospatial\lib\site-packages\holoviews\util\__init__.py in resolve(key, kwargs)
       1005             elif isinstance(map_obj, DynamicMap) and map_obj._posarg_keys and not key:
       1006                 key = tuple(kwargs[k] for k in map_obj._posarg_keys)
    -> 1007             return key, map_obj[key]
       1008 
       1009         def apply(element, *key, **kwargs):
    

    ~\anaconda3\envs\geospatial\lib\site-packages\holoviews\core\spaces.py in __getitem__(self, key)
       1329         # Not a cross product and nothing cached so compute element.
       1330         if cache is not None: return cache
    -> 1331         val = self._execute_callback(*tuple_key)
       1332         if data_slice:
       1333             val = self._dataslice(val, data_slice)
    

    ~\anaconda3\envs\geospatial\lib\site-packages\holoviews\core\spaces.py in _execute_callback(self, *args)
       1098 
       1099         with dynamicmap_memoization(self.callback, self.streams):
    -> 1100             retval = self.callback(*args, **kwargs)
       1101         return self._style(retval)
       1102 
    

    ~\anaconda3\envs\geospatial\lib\site-packages\holoviews\core\spaces.py in __call__(self, *args, **kwargs)
        712 
        713         try:
    --> 714             ret = self.callable(*args, **kwargs)
        715         except KeyError:
        716             # KeyError is caught separately because it is used to signal
    

    <ipython-input-183-193d29ff9176> in get_date(symbol, **kwargs)
          1 def get_date(symbol, **kwargs):
          2     gdf = sjoined
    ----> 3     return gdf.hvplot(geo=True, color="red",  tiles="ESRI")
    

    ~\anaconda3\envs\geospatial\lib\site-packages\pandas\core\generic.py in __getattr__(self, name)
       5458             if self._info_axis._can_hold_identifiers_and_holds_name(name):
       5459                 return self[name]
    -> 5460             return object.__getattribute__(self, name)
       5461 
       5462     def __setattr__(self, name: str, value) -> None:
    

    AttributeError: 'GeoDataFrame' object has no attribute 'hvplot'





    :DynamicMap   [Symbol]




```python
def load_symbol(symbol, **kwargs):
    df = pd.DataFrame(getattr(stocks, symbol))
    df['date'] = df.date.astype('datetime64[ns]')
    return hv.Curve(df, ('date', 'Date'), ('adj_close', 'Adjusted Close'))

stock_symbols = ['AAPL', 'FB', 'GOOG', 'IBM', 'MSFT']
dmap = hv.DynamicMap(load_symbol, kdims='Symbol').redim.values(Symbol=stock_symbols)
```


```python
dmap
```






<div id='1001'>





  <div class="bk-root" id="9468ebc8-ef73-47a5-b1c8-72c1bb9b84e7" data-root-id="1001"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
    var docs_json = {"1672a992-936e-4d82-94dd-377e4357ea5a":{"roots":{"references":[{"attributes":{},"id":"1035","type":"Selection"},{"attributes":{"children":[{"id":"1092"},{"id":"1093"},{"id":"1095"}],"margin":[0,0,0,0],"name":"Column01478"},"id":"1091","type":"Column"},{"attributes":{"months":[0,1,2,3,4,5,6,7,8,9,10,11]},"id":"1065","type":"MonthsTicker"},{"attributes":{"margin":[20,20,20,20],"min_width":250,"options":["AAPL","FB","GOOG","IBM","MSFT"],"title":"Symbol","value":"AAPL","width":250},"id":"1094","type":"Select"},{"attributes":{"axis_label":"Adjusted Close","bounds":"auto","formatter":{"id":"1044"},"major_label_orientation":"horizontal","ticker":{"id":"1019"}},"id":"1018","type":"LinearAxis"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"1027","type":"BoxAnnotation"},{"attributes":{"base":60,"mantissas":[1,2,5,10,15,20,30],"max_interval":1800000.0,"min_interval":1000.0,"num_minor_ticks":0},"id":"1059","type":"AdaptiveTicker"},{"attributes":{},"id":"1044","type":"BasicTickFormatter"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"1022"},{"id":"1023"},{"id":"1024"},{"id":"1025"},{"id":"1026"}]},"id":"1028","type":"Toolbar"},{"attributes":{"source":{"id":"1034"}},"id":"1041","type":"CDSView"},{"attributes":{"months":[0,2,4,6,8,10]},"id":"1066","type":"MonthsTicker"},{"attributes":{"line_color":"#30a2da","line_width":2,"x":{"field":"date"},"y":{"field":"adj_close"}},"id":"1037","type":"Line"},{"attributes":{"margin":[5,5,5,5],"name":"VSpacer01476","sizing_mode":"stretch_height"},"id":"1092","type":"Spacer"},{"attributes":{"days":[1,15]},"id":"1064","type":"DaysTicker"},{"attributes":{"data":{"adj_close":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[3270]},"date":{"__ndarray__":"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AADAGGpfckIAAIB+vF9yQgAAQOQOYHJCAAAASmFgckIAAMCvs2ByQgAAAOGqYXJCAADARv1hckIAAICsT2JyQgAAQBKiYnJCAAAAePRickIAAAAPPmRyQgAAwHSQZHJCAACA2uJkckIAAEBANWVyQgAAgHEsZnJCAABA135mckIAAAA90WZyQgAAwKIjZ3JCAACACHZnckIAAMA5bWhyQgAAgJ+/aHJCAABABRJpckIAAABrZGlyQgAAwNC2aXJCAAAAAq5qckIAAMBnAGtyQgAAgM1Sa3JCAABAM6VrckIAAACZ92tyQgAAADBBbXJCAADAlZNtckIAAID75W1yQgAAQGE4bnJCAACAki9vckIAAED4gW9yQgAAAF7Ub3JCAADAwyZwckIAAIApeXByQgAAwFpwcXJCAACAwMJxckIAAEAmFXJyQgAAAIxncnJCAADA8blyckIAAAAjsXNyQgAAwIgDdHJCAACA7lV0ckIAAEBUqHRyQgAAALr6dHJCAABA6/F1ckIAAABRRHZyQgAAwLaWdnJCAACAHOl2ckIAAECCO3dyQgAAgLMyeHJCAABAGYV4ckIAAAB/13hyQgAAwOQpeXJCAACASnx5ckIAAMB7c3pyQgAAgOHFenJCAABARxh7ckIAAACtantyQgAAAES0fHJCAADAqQZ9ckIAAIAPWX1yQgAAQHWrfXJCAAAA2/19ckIAAEAM9X5yQgAAAHJHf3JCAADA15l/ckIAAIA97H9yQgAAQKM+gHJCAACA1DWBckIAAEA6iIFyQgAAAKDagXJCAADABS2CckIAAIBrf4JyQgAAwJx2g3JCAACAAsmDckIAAEBoG4RyQgAAAM5thHJCAADAM8CEckIAAABlt4VyQgAAwMoJhnJCAACAMFyGckIAAECWroZyQgAAAPwAh3JCAABALfiHckIAAACTSohyQgAAwPiciHJCAACAXu+IckIAAEDEQYlyQgAAgPU4inJCAABAW4uKckIAAADB3YpyQgAAwCYwi3JCAACAjIKLckIAAMC9eYxyQgAAgCPMjHJCAABAiR6NckIAAADvcI1yQgAAwFTDjXJCAADA6wyPckIAAIBRX49yQgAAQLexj3JCAAAAHQSQckIAAEBO+5ByQgAAALRNkXJCAADAGaCRckIAAIB/8pFyQgAAQOVEknJCAACAFjyTckIAAEB8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","dtype":"float64","order":"little","shape":[3270]}},"selected":{"id":"1035"},"selection_policy":{"id":"1056"}},"id":"1034","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"1034"},"glyph":{"id":"1037"},"hover_glyph":null,"muted_glyph":{"id":"1039"},"nonselection_glyph":{"id":"1038"},"selection_glyph":null,"view":{"id":"1041"}},"id":"1040","type":"GlyphRenderer"},{"attributes":{"base":24,"mantissas":[1,2,4,6,8,12],"max_interval":43200000.0,"min_interval":3600000.0,"num_minor_ticks":0},"id":"1060","type":"AdaptiveTicker"},{"attributes":{"below":[{"id":"1014"}],"center":[{"id":"1017"},{"id":"1021"}],"left":[{"id":"1018"}],"margin":[5,5,5,5],"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"plot_height":300,"plot_width":300,"renderers":[{"id":"1040"}],"sizing_mode":"fixed","title":{"id":"1006"},"toolbar":{"id":"1028"},"x_range":{"id":"1003"},"x_scale":{"id":"1010"},"y_range":{"id":"1004"},"y_scale":{"id":"1012"}},"id":"1005","subtype":"Figure","type":"Plot"},{"attributes":{"axis":{"id":"1014"},"grid_line_color":null,"ticker":null},"id":"1017","type":"Grid"},{"attributes":{"num_minor_ticks":5,"tickers":[{"id":"1058"},{"id":"1059"},{"id":"1060"},{"id":"1061"},{"id":"1062"},{"id":"1063"},{"id":"1064"},{"id":"1065"},{"id":"1066"},{"id":"1067"},{"id":"1068"},{"id":"1069"}]},"id":"1015","type":"DatetimeTicker"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer01480","sizing_mode":"stretch_width"},"id":"1090","type":"Spacer"},{"attributes":{},"id":"1042","type":"DatetimeTickFormatter"},{"attributes":{"end":753.698,"reset_end":753.698,"reset_start":-61.558,"start":-61.558,"tags":[[["adj_close","Adjusted Close",null]]]},"id":"1004","type":"Range1d"},{"attributes":{"children":[{"id":"1094"}],"css_classes":["panel-widget-box"],"margin":[5,5,5,5],"name":"WidgetBox01471"},"id":"1093","type":"Column"},{"attributes":{},"id":"1024","type":"WheelZoomTool"},{"attributes":{},"id":"1010","type":"LinearScale"},{"attributes":{},"id":"1022","type":"SaveTool"},{"attributes":{"end":1362096000000.0,"reset_end":1362096000000.0,"reset_start":951868800000.0,"start":951868800000.0,"tags":[[["date","Date",null]]]},"id":"1003","type":"Range1d"},{"attributes":{"months":[0,6]},"id":"1068","type":"MonthsTicker"},{"attributes":{"line_alpha":0.1,"line_color":"#30a2da","line_width":2,"x":{"field":"date"},"y":{"field":"adj_close"}},"id":"1038","type":"Line"},{"attributes":{"days":[1,4,7,10,13,16,19,22,25,28]},"id":"1062","type":"DaysTicker"},{"attributes":{"days":[1,8,15,22]},"id":"1063","type":"DaysTicker"},{"attributes":{"margin":[5,5,5,5],"name":"VSpacer01477","sizing_mode":"stretch_height"},"id":"1095","type":"Spacer"},{"attributes":{"days":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]},"id":"1061","type":"DaysTicker"},{"attributes":{"axis_label":"Date","bounds":"auto","formatter":{"id":"1042"},"major_label_orientation":"horizontal","ticker":{"id":"1015"}},"id":"1014","type":"DatetimeAxis"},{"attributes":{"axis":{"id":"1018"},"dimension":1,"grid_line_color":null,"ticker":null},"id":"1021","type":"Grid"},{"attributes":{"text":"Symbol: AAPL","text_color":{"value":"black"},"text_font_size":{"value":"12pt"}},"id":"1006","type":"Title"},{"attributes":{},"id":"1026","type":"ResetTool"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer01479","sizing_mode":"stretch_width"},"id":"1002","type":"Spacer"},{"attributes":{},"id":"1069","type":"YearsTicker"},{"attributes":{"children":[{"id":"1002"},{"id":"1005"},{"id":"1090"},{"id":"1091"}],"margin":[0,0,0,0],"name":"Row01470"},"id":"1001","type":"Row"},{"attributes":{"client_comm_id":"f8f9bc28372b418ba5f14007e626171e","comm_id":"c066d57612164935b70a22d818e38dc1","plot_id":"1001"},"id":"1156","type":"panel.models.comm_manager.CommManager"},{"attributes":{},"id":"1012","type":"LinearScale"},{"attributes":{"overlay":{"id":"1027"}},"id":"1025","type":"BoxZoomTool"},{"attributes":{"line_alpha":0.2,"line_color":"#30a2da","line_width":2,"x":{"field":"date"},"y":{"field":"adj_close"}},"id":"1039","type":"Line"},{"attributes":{"mantissas":[1,2,5],"max_interval":500.0,"num_minor_ticks":0},"id":"1058","type":"AdaptiveTicker"},{"attributes":{},"id":"1023","type":"PanTool"},{"attributes":{},"id":"1019","type":"BasicTicker"},{"attributes":{},"id":"1056","type":"UnionRenderers"},{"attributes":{"months":[0,4,8]},"id":"1067","type":"MonthsTicker"}],"root_ids":["1001","1156"]},"title":"Bokeh Application","version":"2.2.3"}};
    var render_items = [{"docid":"1672a992-936e-4d82-94dd-377e4357ea5a","root_ids":["1001"],"roots":{"1001":"9468ebc8-ef73-47a5-b1c8-72c1bb9b84e7"}}];
    root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
  if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 100) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 10, root)
  }
})(window);</script>




```python

```
