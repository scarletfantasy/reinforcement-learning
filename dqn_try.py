import gym,sys,numpy as np
from gym.envs.registration import register
import tensorflow as tf

from tensorflow import keras
env=gym.make('Breakout-v0')
model=keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (8, 8), (1, 1), padding='same', activation='relu',input_shape=(210,160,3)))
model.add(tf.keras.layers.Conv2D(64,(4,4),activation='relu'))
model.add(tf.keras.layers.Conv2D(128,(2,2),activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(env.action_space.n))
s=tf.placeholder(tf.uint8,[None,210,160,3])
q_value=model(s)
