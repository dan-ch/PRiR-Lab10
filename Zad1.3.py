import tensorflow as tf
import timeit

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
  
# /device:GPU:0
# /cpu:0

def funkcja(x):
  with tf.device('/device:GPU:0'):
    return tf.math.sin(x**4 + 2*x)


def calka(x):
  with tf.device('/device:GPU:0'):
    dx = (x[-1] - x[0]) / len(x)
    return (4*tf.reduce_sum(funkcja(x[1:-1:2])) + 2*tf.reduce_sum(funkcja(x[2:-1:2])) + funkcja(x[1]) + funkcja(x[-1]))*dx/3
    
a = tf.constant(float(input("Podaj początek przedziału: ")), dtype=tf.float32)
b = tf.constant(float(input("Podaj koniec przedziału: ")), dtype=tf.float32)
x = tf.linspace(a, b, 2*int(input("Podaj ilosc podziałów: ")))
start = timeit.default_timer()
tf.print(calka(x))
end = timeit.default_timer()
print(f'Czas wykonania: {round(end-start, 8)}')
