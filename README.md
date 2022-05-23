# Machine learning opensource guide.

## Basic chapter
### Importing modules
```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
# Set default font size
plt.rcParams['font.size'] = 24

from IPython.core.pylabtools import figsize
plt.rcParams["figure.figsize"] = (15,7)
plt.rcParams['figure.dpi'] = 60
```
### Parsing file on api 
df - dateframe, pd - pandas.
You can use any external apis
```
df=pd.read_csv('https://raw.githubusercontent.com/*.csv')
```
### Return the first n rows.
This function returns the first n rows for the object based on position. It is useful for quickly testing if your object has the right type of data in it.
```
df.head(3)
```
### Types (Приведение типов)
```
df['date']=pd.to_datetime(df['date'])
df.info()
#Вывод пустых значений
pd.set_option('display.max_rows',None)
df.isnull().sum()
pd.set_option('display.max_rows',10)
df.shape #
```
Все данные при парсинге из репозитория были включены. Размерность набора данных составляет df.shape[0] строк и df.shape[1] столбцов

Дополним набор данных новой информацией, чтобы в случае необходимости точность модели при обучении была больше. Также новые данные могут пригодиться для того, чтобы включить их в аналих данных, из которого можно вынести некоторые зависимости, если они присутствует.

В качестве новых данных возьмём среднее количество смертей и заражённых на регион.

