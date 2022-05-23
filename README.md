# Machine learning opensource guide.

## Basic chapter
Thanks for support [@pandas](https://pandas.pydata.org/getting_started.html), [@seaborn](https://github.com/eugenerum/ToolDataHelper/tree/main/Examples), [@matplotlib](https://matplotlib.org/)
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
df.shape
```
Все данные при парсинге из репозитория были включены. Размерность набора данных составляет df.shape[0] строк и df.shape[1] столбцов

Дополним набор данных новой информацией, чтобы в случае необходимости точность модели при обучении была больше. Также новые данные могут пригодиться для того, чтобы включить их в аналих данных, из которого можно вынести некоторые зависимости, если они присутствует.

В качестве новых данных возьмём среднее количество протестированных и полностью вакцинированных на континент, чтобы оценить готовность каждого из континентов к борьбе против короновирусной инфекции.
### Заполнение пустых полей

```
df[['location', 'total_tests_per_thousand', 'people_fully_vaccinated_per_hundred']]=df[['location', 'total_tests_per_thousand', 'people_fully_vaccinated_per_hundred']].fillna(0)
```
### Формирование дополнительных атрибутов
```
# Формирование дополнительных атрибутов
external_attrs=df[['continent', 'total_tests_per_thousand', 'people_fully_vaccinated_per_hundred']].groupby(by="continent").mean().rename(
    columns={'total_tests_per_thousand':'mean_total_tests_per_thousans', 'people_fully_vaccinated_per_hundred':'mean_fully_vaccinated_per_hundered'})

```
```
df=df.merge(external_attrs, on='continent')
```
Генерация новых данных произведена

## 1.2. Data predprocessing and selection most valuable attributes
```
df=df.fillna(0)
```
Определение наиболее значимых атрибутов
Чтобы найти наиболее значимые атрибуты, построим корреляцию Пирсона на тепловой карте
### Фомирование корреляции Пирсона
```
corr=df.corr()
plt.figure(figsize=(200, 100))

matrix = np.triu(corr)
heatmap = sns.heatmap(corr, annot=True, mask=matrix, fmt='.1g', cmap='coolwarm')
heatmap.set_title('Correlation', fontdict={'fontsize':22}, pad=25)
```
Как видим выше, достаточно много признаков имеют высокий коэффициент корреляции, но наиболее значимыми атрибутами являются: `total_casem, new_case, new_cases_smoothed, total_deaths, new_deaths и new_deaths_smoothed`

Пустые значения
Пустые значения были предобработанны ранее, после предобработки их больше не осталось

pd.set_option('display.max_rows',None)
df.isnull().sum()
pd.set_option('display.max_rows',10)

Плотность распределения данных

Сформируем графики плотностей для каждого признака

```
# Распределение категориальной переменной
plt.figure(figsize=(10, 5))
sns.kdeplot(df['tests_units'].value_counts())
plt.title('Distribution tests_units')
plt.xlabel('Значения')
plt.ylabel('Распределение')
plt.show()
```
```
# Распределение категориальной переменной
plt.figure(figsize=(10, 5))
sns.kdeplot(df['iso_code'].value_counts())
plt.title('Distribution iso_code')
plt.xlabel('Значения')
plt.ylabel('Распределение')
plt.show()
```
```
# Распределение категориальной переменной
plt.figure(figsize=(10, 5))
sns.kdeplot(df['continent'].value_counts())
plt.title('Distribution continent')
plt.xlabel('Значения')
plt.ylabel('Распределение')
plt.show()
```
```
# Распределение категориальной переменной
plt.figure(figsize=(10, 5))
sns.kdeplot(df['location'].value_counts())
plt.title('Distribution location')
plt.xlabel('Значение')
plt.ylabel('Распределение')
plt.show()
```
```
#Функция вывода распределения каждого атрибута, являющегося численным признаком
def plot(column):
    plt.figure(figsize=(10, 5))
    sns.kdeplot(df[column])
    plt.title('Distribution '+column)
    plt.xlabel('Значения')
    plt.ylabel('Распределение')
    plt.show()
```
```
#Применение функции
for column in df[:100].select_dtypes(exclude=['object']).columns:
    plot(column)
```
