# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

def my_func(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg

def hour_to_day_mean(array):
    time = 24
    # print('hour_to_day_mean')
    # print(array)
    result = tf.reduce_mean(tf.reshape(array, [array.shape[0] // time, time]), 1)
    # print(result)
    return result


def hour_to_day_mean_numpy(array):
    time = 24

    #array.reshape()
    array = np.reshape(array, (array.shape[0], array.shape[1]//time, time, array.shape[2]))
    #print(array.shape)
    array = np.mean(array, axis=2)

    #print(array.shape)

    return array


hour = 24 * 7
cols = 3
array = np.array(range(hour*cols))

print(array)

#print('input')
#print(array.shape)
#print(array)

array = array.reshape([1,hour,cols])

print('input')
print(array.shape)
print(array)


print('array[0:1,:,:].shape')
print(array[0:1,:,:])

array2 = hour_to_day_mean_numpy(array)


print('array2.shape')
print(array2.shape)

#print('array2[0,0,:]')
#print(array2[0,0,:])

#print('array2[0,:,0]')
#print(array2[0,:,0])

print('output')
print(array2.shape)
print(array2)


print('array2[0,:,0]')
print(array2[0,:,0])

print('array2[0,:,1]')
print(array2[0,:,1])


#def my_func(arg):
#  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
#  return arg

#value_1 = my_func(tf.constant(
#    [ [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.]]))
#print(value_1)



#array = tf.convert_to_tensor_c2(array, shape=(1,24,1), dtype=tf.float32)


#print(array)


#array = tf.