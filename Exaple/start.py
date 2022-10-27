'''Пример работы кластеризованных данных с реальными значениями SMART'''
import pandas as pd
from joblib import load
from sklearn.neighbors import KNeighborsClassifier

surface = load('/path/to/surface.joblib')
mech = load('/path/to/mech.joblib')
hardware = load('/path/to/hardware.joblib')

knx = KNeighborsClassifier()
kny = KNeighborsClassifier()
knz = KNeighborsClassifier()

X = surface[['smart_5_raw', 'smart_197_raw']]
Y = mech[['smart_3_raw', 'smart_7_raw']]
Z = hardware[['smart_195_raw', 'smart_199_raw']]
knx = knx.fit(X, surface['cluster'])
kny = kny.fit(Y, mech['cluster'])
knz = knz.fit(Z, hardware['cluster'])
surf = pd.DataFrame(columns=['smart_5_raw', 'smart_197_raw'])
mec = pd.DataFrame(columns=['smart_3_raw', 'smart_7_raw'])
hardw = pd.DataFrame(columns=['smart_195_raw', 'smart_199_raw'])

surf.loc['smart_5_raw'] = int(input('Введите данные SMART 5 секции raw: '))
surf.loc['smart_197_raw'] = int(input('Введите данные SMART 197 секции raw: '))
mec.loc['smart_3_raw'] = int(input('Введите данные SMART 3 секции raw: '))
mec.loc['smart_7_raw'] = int(input('Введите данные SMART 7 секции raw: '))
hardw.loc['smart_195_raw'] = int(input('Введите данные SMART 195 секции raw: '))
hardw.loc['smart_199_raw'] = int(input('Введите данные SMART 199 секции raw: '))

predsurf = knx.predict(surf)
predmec = knx.predict(mec)
predhardw = knx.predict(hardw)
print('Рекомендации по жесткому диску:')
if predsurf.any() > 0:
  print('Сделайте резервную копию важных данных. Проверьте поверхность жесткого диска!')
else:
  print('Поверхность жесткого диска в порядке.')
if predmec.any() > 0:
  print('Сделайте резервную копию важных данных. Возможен выход из строя механики жесткого диска!')
else:
  print('Механика жесткого диска в порядке.')
if predhardw.any() > 0:
  print('Сделайте резервную копию важных данных.', 'Проверьте подключение жесктого диска, по возможности замените кабель.', 'Проверьте обновление микропрограммы жесткого диска.', sep='\n')
else:
  print('Програмное обеспеение (микропрограмма) работает стабильно')
print('')