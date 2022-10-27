'''Кластеризация данных модели'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import matplotlib.cm as cm


sc = StandardScaler()

new = pd.read_csv('/path/to/csv')

# Отбросим выбросы по текущим данным. 
# В случае обработки большего массива данных, необходимо смотреть выбросы данных, соответственно, 
# большего массива данных.
new.drop(new[(new['smart_5_raw'] > 2500)].index, inplace=True)
new.drop(new[(new['smart_197_raw'] > 100)].index, inplace=True)
new.drop(new[(new['smart_199_raw'] > 50000)].index, inplace=True)

# Показатели состояния поверхности жесктого диска
surface = new[['smart_5_raw', 'smart_197_raw']]
# Показатели состояния механической части жесткого диска
mech = new[['smart_3_raw', 'smart_7_raw']]
# Показатели ошибок програмной части (микрокода) жесткого диска
hardware = new[['smart_195_raw', 'smart_199_raw']]

'''Определение методом локтя оптиального количества кластеров по разным показателям'''
scaled_x = sc.fit_transform(surface)
kx_in = []
for i in range(2, 10):
  k_x = KMeans(n_clusters=i)
  k_x.fit(scaled_x)
  kx_in.append(k_x.inertia_)
plt.plot(range(2,10), kx_in)
plt.xlabel("Количество кластеров")
plt.ylabel("SSE")
plt.show()

scaled_y = sc.fit_transform(mech)
ky_in = []
for i in range(2, 10):
  k_y = KMeans(n_clusters=i)
  k_y.fit(scaled_y)
  ky_in.append(k_y.inertia_)
plt.plot(range(2,10), ky_in)
plt.xlabel("Количество кластеров")
plt.ylabel("SSE")
plt.show()

scaled_z = sc.fit_transform(hardware)
kz_in = []
for i in range(2, 10):
  k_z = KMeans(n_clusters=i)
  k_z.fit(scaled_z)
  kz_in.append(k_z.inertia_)
plt.plot(range(2,10), kz_in)
plt.xlabel("Количество кластеров")
plt.ylabel("SSE")
plt.show()

# Визуализация кластеризации показателей
k = KMeans(n_clusters=3)
k.fit(scaled_x)

# добавим кластерные оценки в surface
surface['cluster'] = k.labels_

# визуализируем кластерную модель
plt.scatter(surface['smart_5_raw'], surface['smart_197_raw'], c=k.labels_)
plt.xlabel("smart_5_raw")
plt.ylabel("smart_197_raw")
plt.show()

g = KMeans(n_clusters=3)
g.fit(scaled_y)

# добавим кластерные оценки в mech
mech['cluster'] = g.labels_

# визуализируем кластерную модель
plt.scatter(mech['smart_3_raw'], mech['smart_7_raw'], c=g.labels_)
plt.xlabel("smart_3_raw")
plt.ylabel("smart_7_raw")
plt.show()

q = KMeans(n_clusters=3)
q.fit(scaled_z)

# добавим кластерные оценки в surface
hardware['cluster'] = q.labels_

# визуализируем кластерную модель
plt.scatter(hardware['smart_195_raw'], hardware['smart_199_raw'], c=q.labels_)
plt.xlabel("smart_195_raw")
plt.ylabel("smart_199_raw")
plt.show()

# Сохранение кластеризованных данных
from joblib import dump
dump(surface, '/path/to/surface.joblib')
dump(mech, '/path/to/mech.joblib')
dump(hardware, '/path/to/hardware.joblib')