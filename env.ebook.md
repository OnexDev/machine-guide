import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# Set default font size
plt.rcParams['font.size'] = 24

from IPython.core.pylabtools import figsize
plt.rcParams["figure.figsize"] = (15,7)
plt.rcParams['figure.dpi'] = 60

import seaborn as sns
sns.set(font_scale = 2) 

import warnings
warnings.filterwarnings("ignore")
## 2.1 Разбиение набора данных
df=pd.read_csv('result_data.csv')
Разобъём набор данных таким образом, как это рекомендовано согласно документации `Sklearn`. А именно `30 на 70`. Как представленно в описании, такая выборка является оптимальной, поскольку абсолютное большинство данных должно находится при обучении модели, чтобы получить наиболее оптимизированную модель со стороны её точности

### Стратификация
При разделении стратифицируем данные, чтобы получить одинаковую в процентом соотношении выборку, чтобы не было перевеса на какой-то один класс и такая ситуация не повлияла на некорректное обучение модели. 

**Стратифицированое разбиение данных будет производиться по атрибуту региона, чтобы каждое значение этого столбца попало и в тестовую выборку, и в обучающую**
df
df['date']=pd.to_datetime(df['date'])
df['day']=df['date'].apply(lambda x: x.day)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df=df[df['Rt']<5].reset_index(drop=True)
### Определение переменной опасности
df1=df[df['Rt']<=0.7]
df1['Danger']=0
df2=df[(df['Rt']>0.7) & (df['Rt']<=0.95)]
df2['Danger']=1
df3=df[df['Rt']>0.95]
df3['Danger']=2
df=pd.concat([df1, df2, df3]).reset_index(drop=True)
X=df[['new_cases', 'new_deaths', 'Rt']]
y=df['Danger']
## Разбиение
#Получение выборок
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
## 2.2 Визуализация зависимостей данных
Рассмотрим несколько способов визуализации, это heatmap, reg plot, points distribution и corr plot. 

Данные способы визуализации были выбраны, потому что они визуально понятно показывают необходимую информацию.


#Данные визуализации сводной таблицы
pt = pd.pivot_table(df, index='continent', columns='Danger', aggfunc='size', fill_value=0)
#Приведение данных к единому процентному соотношению
pt = pt.apply(lambda x: round(x / x.sum() * 100,1), axis=1)
#Воспроизведение визуализации 
plt.figure(figsize=(12, 8), dpi=72)
ax = sns.heatmap(pt, annot=True, linewidths=0.1, cmap="copper", fmt='g');
#Наименование осей и названия вазиулизации
ax.set_title('Влияние фигурирования континента на целевую переменную', fontsize = 15, y=1.05)
ax.set_xlabel('Уровень угрозы', fontsize = 13)
ax.set_ylabel('Регион', fontsize = 13)
plt.show()
Данный график демонстрирует влияние региона на уровень угрозы(`Danger`). Выше можно увидеть, что наиболее высокий уровень угрозы с повышенным коэффициентом встречается равномернно во всех регионах кроме островного(`Oceania`). Минимальный уровень угрозы демонстрируется как раз в регионе `Oceania`. 0 можно не брать в рассчёт, т.к. данная переменная указывает на неизвостность региона
#Данные визуализации сводной таблицы
pt = pd.pivot_table(df, index='tests_units', columns='Danger', aggfunc='size', fill_value=0)
#Приведение данных к единому процентному соотношению
pt = pt.apply(lambda x: round(x / x.sum() * 100,1), axis=1)
#Воспроизведение визуализации 
plt.figure(figsize=(12, 8), dpi=72)
ax = sns.heatmap(pt, annot=True, linewidths=0.1, cmap="copper", fmt='g');
#Наименование осей и названия вазиулизации
ax.set_title('Влияние тестирования пациентов на целевую переменную', fontsize = 15, y=1.05)
ax.set_xlabel('Категория тестирования', fontsize = 13)
ax.set_ylabel('Уровень угрозы', fontsize = 13)
plt.show()
Данный график демонстрирует тестирования пациентов на уровень угрозы(`Danger`). Выше можно увидеть, что наиболее высокие показатели по колличеству тестирования содержится в уровне опсности 2. Минимальные показатели тестирования содержатся среди минимального уровня опасности 
ax = sns.regplot(x=df['new_cases'], y=df['Danger'], color="g")
ax = sns.regplot(x=df['new_deaths'], y=df['Danger'], color="r")
ax = sns.regplot(x=df['Rt'], y=df['Danger'], color="b")
Три графика выше демонстрируют линейную зависимость переменной Danger от целевой переменных X. Как можно увидеть, все эти три графика имеют почти идиальную линию идущую от низа вверх. Это означает, что в этих данных существует явная зависимость переменной Danger от переменных выше
plt.figure(figsize=(17, 6))
sns.scatterplot(data=df, x='date', y='day', alpha=0.005, s=13, hue='Danger');
corrDf=df[['new_cases', 'new_deaths', 'new_tests', 'population', 'Rt', 'Danger']]

corrMatrix = corrDf.corr()
sns.heatmap(corrMatrix, annot=True, vmin=-1, vmax=1, cmap='coolwarm')
plt.show()
Согласно корреляции Пирсона выше, можно увидеть, что наибольшее влияние на целевую переменную Danger имеет только Rt - индекс распространения инфекции. 
## 2.3 Классифиткация 

Рассмотрим три модели классификации
### KNeighborsClassifier
Классификация на основе соседей - это тип обучения на основе экземпляров или необобщающего обучения: он не пытается построить общую внутреннюю модель, а просто сохраняет экземпляры обучающих данных. Классификация вычисляется простым большинством голосов ближайших соседей каждой точки: точке запроса назначается класс данных, который имеет наибольшее количество представителей среди ближайших соседей точки.

### RandomForestClassifier
Случайный лес — это метаоценка, которая соответствует ряду классификаторов дерева решений для различных подвыборок набора данных и использует усреднение для повышения точности прогнозирования и контроля переобучения. Размер подвыборки управляется параметром max_samples, если bootstrap=True (по умолчанию), в противном случае для построения каждого дерева используется весь набор данных
### GaussianNB
Наи́вный ба́йесовский классифика́тор — простой вероятностный классификатор, основанный на применении теоремы Байеса со строгими (наивными) предположениями о независимости. В зависимости от точной природы вероятностной модели, наивные байесовские классификаторы могут обучаться очень эффективно

## Матрикики
Рассмотрим две метрикики для оценивания модели классификации

### accuracy f1-score
Это гармоническое среднее значений точности и полноты. Возьмём её, потому что она дает лучшую оценку неправильно классифицированных случаев

### macro avg f1-score

macro avg f1-score пожалуй, самый простой из многочисленных методов усреднения. Макроусредненная оценка F1 (или макрооценка F1) вычисляется путем взятия среднего арифметического (также известного как невзвешенное среднее) всех оценок F1 для каждого класса. Этот метод будет взят, поскольку он обрабатывает все классы одинаково, независимо от их значений поддержки
## 2.4 Обучение
#Импорт моделей
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
#Обучение
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
preds=neigh.predict(X_test)
print(classification_report(preds, y_test))
#Обучение
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_preds=rfc.predict(X_test)
print(classification_report(rfc_preds, y_test))
#Обучение
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_preds=gnb.predict(X_test)
print(classification_report(gnb_preds, y_test))
### Вывод
Наиболее оптимальной моделью будет `KNeighborsClassifier` c accuracy f1-score = `0.78` и macro avg f1-score = `0.74`, поскольку по сравнению с другими он показал наилучший результат. `RandomForestClassifier` не будет взят, поскольку у него явное переобучение
## 3.4 Feature Engineering

Преобразуем набор данных путём генерации новых данных с целью повышения точности классификатора и использование StandardScaler
#Генерация данных
df['day']=df['date'].apply(lambda x: x.day)
from sklearn.preprocessing import StandardScaler
#Преобразование с помощью StandardScaler
scaler = StandardScaler()
X=df[['new_cases', 'new_deaths', 'Rt', 'day']]
y=df['Danger']

#Получение выборок
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

#Обучение
rfc = KNeighborsClassifier(n_neighbors=3)
rfc.fit(X_train, y_train)
rfc_preds=rfc.predict(X_test)
print(classification_report(rfc_preds, y_test))
## Выводы по Feature Engineering 
Из результатов выше видно, что преобразование данных для Feature Engineering не привёло к улучшению модели
## Отчёт
* 2.1 Разбиение набора данных - набор данныхз разбит на обучаюшую и тестовую выборки
* 2.2 Визуализация зависимостей данных - визуализация данных несколькими способами
* 2.3 Классификация - выбраны 3 алгоритма классификации
* 2.4 Обучение - произведена классификация по уровню опасности
* 2.5 Feature Engineering - проведено дополнение набора данных дополнительными данных и обучение модели
# Сохранение данных
df.to_csv('result_data.csv', encoding='utf-8-sig', index=False)
