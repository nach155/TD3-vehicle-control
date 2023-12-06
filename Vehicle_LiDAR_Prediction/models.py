import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras.layers as kl
import numpy as np


class ActorNetwork(tf.keras.Model):

    def __init__(self, action_space, max_action, min_action):

        super(ActorNetwork, self).__init__()
        self.action_space = action_space
        self.max_action = max_action
        self.min_action = min_action
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00003)
        # self.flatten1 = kl.Flatten()
        self.dense1 = kl.Dense(256, activation="relu")
        # self.dense2 = kl.Dense(128, activation="relu")
        self.dense3 = kl.Dense(256, activation="relu")
        self.actions = kl.Dense(self.action_space, activation="tanh")

    def call(self, s, training=True):
        # x = self.flatten1(s)
        x = self.dense1(s)
        # x = self.dense2(x)
        x = self.dense3(x)
        actions = self.actions(x)
        # actions = actions * self.max_action
        
        return actions

    def sample_action(self, state, noise=None):
        """
            ノイズつきアクションのサンプリング
            -1:1 -> min:max に変換
        """
        state = np.atleast_2d(state).astype(np.float32)
        action = self(state, training=False).numpy()[0]
        if noise:
            # action += np.random.normal(0, noise*self.max_action[0],size=self.action_space)
            # action += np.array([np.random.normal(0,noise*self.max_action[act]) for act in range(self.action_space)])
            action += np.random.normal(0,noise,size=self.action_space)

        # -1:1 -> min:max に変換
        # action = self.min_max_decode(action,self.min_action,self.max_action)
        # 入力飽和
        action = np.clip(action, -1, 1) 

        return action

    def min_max_decode(self, x_norm, source_min, source_max):
        return ((x_norm * source_max) - (x_norm * source_min) + source_max + source_min) * 0.5

class CriticNetwork(tf.keras.Model):

    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
        # self.flatten1 = kl.Flatten()
        self.dense1 = kl.Dense(256, activation="relu")
        # self.dense2 = kl.Dense(128, activation="relu")
        self.dense5 = kl.Dense(256, activation="relu")
        self.out1 = kl.Dense(1)

        # self.flatten2 = kl.Flatten()
        self.dense3 = kl.Dense(256, activation="relu")
        # self.dense4 = kl.Dense(128, activation="relu")
        self.dense6 = kl.Dense(256, activation="relu")
        self.out2 = kl.Dense(1)

    def call(self, s, a, training=True):
        # s = self.flatten1(s)
        x = tf.concat([s, a], 1)
        # x1 = self.flatten1(x)
        x1 = self.dense1(x)
        # x1 = self.dense2(x1)
        x1 = self.dense5(x1)
        q1 = self.out1(x1)

        # x2 = self.flatten1(x)
        x2 = self.dense3(x)
        # x2 = self.dense4(x2)
        x2 = self.dense6(x2)
        q2 = self.out2(x2)

        return q1, q2


if __name__ == "__main__":
    import gym
    #physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)

    actor = ActorNetwork(action_space=2, max_action=np.array([2,2]),min_action=np.array([-2,-2]))

    critic = CriticNetwork()

    env = gym.make("Pendulum-v1")

    s = env.reset()

    a_ = actor.sample_action(s)
    print(a_)

    a = np.atleast_2d(a_)
    s = np.atleast_2d(s)

    q = critic(s, a)

    s2, r, done, _ = env.step([-1])

    s_ = np.vstack([s]*5)
    a_ = np.vstack([a]*5)

    q1, q2 = critic(s_, a_)

    print(q1)
    for q in q1.numpy().flatten():
        print(q)
