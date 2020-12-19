import tensorflow as tf
import numpy as np
from tensorflow import keras
import timeit

ile_zmiennych = int(input("Podaj ilosc zmiennych: "))
ile_wartosci = int(input("Podaj ilosc wartosci: ")) #ilosc gotowych danych do uczenia
tab = []
for i in range(ile_zmiennych):
  temp_tab = []
  for j in range(ile_wartosci):
    temp_tab.append(float(input(f"Podaj wartosc dla [{i}][{j}] zmiennej:  ")))
  tab.append(temp_tab)
  temp_tab.clear

# ile_zmiennych = 4
# ile_wartosci = 5
# tab = []
# tab.append([-1,3,7,11,13])
# tab.append([5,9,3,8,4])
# tab.append([2,3,6,-3,2])
# tab.append([1,12,11,33,28])

# /device:GPU:0
# /cpu:0
start = timeit.default_timer()
with tf.device('/device:GPU:0'): 
  np_tab = np.stack(tab[:-1], axis=1)
  np_wynik = np.array(tab[-1])
  dod = ile_zmiennych - ile_wartosci if ile_zmiennych - ile_wartosci >=1 else 0 
  model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[ile_zmiennych-1])])
  model.compile(optimizer='sgd', loss='mean_squared_error')
  model.fit(np_tab, np_wynik, epochs=(200*(ile_zmiennych+dod)))
end = timeit.default_timer()
print(f'Czas wykonania: {round(end-start, 8)}')

print(model.predict([[3, 8, 9]]))
