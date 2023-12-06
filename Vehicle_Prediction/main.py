"""
TD3 を用いて移動ロボットの制御点を制御する強化学習プログラム
非線形MPCの代わりに強化学習で予測
参考:
https://horomary.hatenablog.com/entry/2020/07/01/001414
"""
from dataclasses import dataclass
import collections
# from datetime import date
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
# import gym
# from gym import wrappers
import matplotlib.pyplot as plt
# import cv2

from replay_buffer import ReplayBuffer
from models import ActorNetwork, CriticNetwork
from environment import MyEnv


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

class TD3Agent:
    def __init__(self):
        
        self.MAX_EXPERIENCES = 30000
        self.MIN_EXPERIENCES = 3000
        
        self.env = MyEnv()
        
        # 入力数
        self.ACTION_SPACE = self.env.ACTION_DIM
        
        # 入力制約
        self.MAX_ACTION = self.env.ACTION_HIGH
        self.MIN_ACTION = self.env.ACTION_LOW
        # 出力数
        self.OBSERVATION_SPACE = self.env.OUTPUT_DIM
        
        self.actor = ActorNetwork(action_space=self.ACTION_SPACE,max_action=self.MAX_ACTION,min_action=self.MIN_ACTION)
        self.target_actor = ActorNetwork(action_space=self.ACTION_SPACE,max_action=self.MAX_ACTION,min_action=self.MIN_ACTION)

        self.critic = CriticNetwork()
        self.target_critic = CriticNetwork()

        self.CRITIC_UPDATE_PERIOD = 4
        self.POLICY_UPDATE_PERIOD = 8
        self.TAU = 0.02
        self.GAMMA = 0.99
        self.BATCH_SIZE = 128
        # self.NOISE_STDDEV = 0.2
        # self.NOISE_STDDEV = lambda episode: 0.2 if episode <= self.N_EPISODE/10 else 0.2*(1-episode/self.N_EPISODE)
        
        self.buffer = ReplayBuffer(max_experiences=self.MAX_EXPERIENCES)
        self.global_steps = 0
        self.hiscore = None
        self._build_networks()

    def _build_networks(self):
        """パラメータの初期化
        """

        dummy_state = np.random.normal(0, 0.1, size=self.OBSERVATION_SPACE)
        dummy_state = (dummy_state[np.newaxis, ...]).astype(np.float32)

        dummy_action = np.random.normal(0, 0.1, size=self.ACTION_SPACE)
        dummy_action = (dummy_action[np.newaxis, ...]).astype(np.float32)

        self.actor.call(dummy_state)
        self.target_actor.call(dummy_state)
        self.target_actor.set_weights(self.actor.get_weights())

        self.critic.call(dummy_state, dummy_action, training=False)
        self.target_critic.call(dummy_state, dummy_action, training=False)
        self.target_critic.set_weights(self.critic.get_weights())

    def play(self,n_episode):

        total_rewards = []
        recent_scores = collections.deque(maxlen=10)
        for n in range(n_episode):
            self.NOISE_STDDEV = (lambda episode:0.2 if episode<=1000 else 0.2*(1-episode/self.n_episode))(n)
            total_reward, localsteps = self.play_episode()
            total_rewards.append(total_reward)
            recent_scores.append(total_reward)
            recent_average_score = sum(recent_scores) / len(recent_scores)

            print(f"Episode {n+1}: {total_reward}")
            print(f"Local steps {localsteps}")
            print(f"Experiences {len(self.buffer)}")
            print(f"Global step {self.global_steps}")
            print(f"Noise stdev {self.NOISE_STDDEV}")
            print(f"recent average score {recent_average_score}")
            print()

            if (self.hiscore is None) or (recent_average_score > self.hiscore):
                self.hiscore = recent_average_score
                print(f"HISCORE Updated: {self.hiscore}\n")
                self.save_model()

        return total_rewards

    def play_episode(self):
        total_reward = 0
        steps = 0
        done = False
        env = self.env
        state = env.reset()
        env.render()
        while not done:
            action = self.actor.sample_action(state, noise=self.NOISE_STDDEV)
            next_state, reward, done, _ = env.step(action)
            exp = Experience(state, action, reward, next_state, done)
            self.buffer.add_experience(exp)
            # state = next_state
            total_reward += reward
            steps += 1
            self.global_steps += 1
            
            #: Delayed Policy update
            if self.global_steps % self.CRITIC_UPDATE_PERIOD == 0:
                if self.global_steps % self.POLICY_UPDATE_PERIOD == 0:
                    self.update_network(self.BATCH_SIZE, update_policy=True)
                    self.update_target_network()
            else:
                self.update_network(self.BATCH_SIZE)
            env.render()
        return total_reward, steps

    def update_network(self, batch_size, update_policy=False):
        if len(self.buffer) < self.MIN_EXPERIENCES:
            return
        (states, actions, rewards, next_states, dones) = self.buffer.get_minibatch(batch_size)

        clipped_noise = np.clip(np.random.normal(0, 0.2, self.ACTION_SPACE), -0.5, 0.5)

        next_actions = self.target_actor(next_states) + clipped_noise #* self.MAX_ACTION

        q1, q2 = self.target_critic(next_states, next_actions)

        next_qvalues = [min(q1, q2) for q1, q2 in zip(q1.numpy().flatten(), q2.numpy().flatten())]

        #: Compute taeget values and update CriticNetwork
        target_values = np.vstack([
            reward + self.GAMMA * next_qvalue if not done else reward
            for reward, done, next_qvalue
            in zip(rewards, dones, next_qvalues)
        ]).astype(np.float32)

        #: Update Critic
        with tf.GradientTape() as tape:
            q1, q2 = self.critic(states, actions)
            loss1 = tf.reduce_mean(tf.square(target_values - q1))
            loss2 = tf.reduce_mean(tf.square(target_values - q2))
            loss = loss1 + loss2

        variables = self.critic.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.critic.optimizer.apply_gradients(zip(gradients, variables))

        #: Delayed Update ActorNetwork
        if update_policy:

            with tf.GradientTape() as tape:
                q1, _ = self.critic(states, self.actor(states))
                J = -1 * tf.reduce_mean(q1)

            variables = self.actor.trainable_variables
            gradients = tape.gradient(J, variables)
            self.actor.optimizer.apply_gradients(zip(gradients, variables))
        
    def update_target_network(self):

        # soft-target update Actor
        target_actor_weights = self.target_actor.get_weights()
        actor_weights = self.actor.get_weights()

        assert len(target_actor_weights) == len(actor_weights)

        self.target_actor.set_weights( (1 - self.TAU) * np.array(target_actor_weights,dtype=object) + (self.TAU) * np.array(actor_weights,dtype=object))

        # soft-target update Critic
        target_critic_weights = self.target_critic.get_weights()
        critic_weights = self.critic.get_weights()

        assert len(target_critic_weights) == len(critic_weights)

        self.target_critic.set_weights((1 - self.TAU) * np.array(target_critic_weights,dtype=object) + (self.TAU) * np.array(critic_weights,dtype=object))

    def save_model(self):

        self.actor.save_weights("checkpoints/actor")
        self.critic.save_weights("checkpoints/critic")

    def load_model(self):

        self.actor.load_weights("checkpoints/actor")

        self.target_actor.load_weights("checkpoints/actor")

        self.critic.load_weights("checkpoints/critic")

        self.target_critic.load_weights("checkpoints/critic")

    def test_play(self, n, monitordir, load_model=False):

        if load_model:
            self.load_model()
        # if monitordir:
        #     env = wrappers.RecordVideo(self.env,monitordir,episode_trigger=(lambda ep:ep%1==0))
        # else:
        #     env = self.env
        env = self.env

        for i in range(n):
            
            total_reward = 0
            steps = 0
            done = False
            state = env.reset()
            frame = env.render(mode='human')
            # fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
            # video  = cv2.VideoWriter(f'{monitordir}/Video-{date.today()}-{i}.mp4', 0x00000020, 10.0, frame.shape[:-1],isColor=True)
            # video.write(frame)
            while not done:
                action = self.actor.sample_action(state, noise=self.NOISE_STDDEV)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                total_reward += reward
                steps += 1
                frame = env.render(mode='human')
                # video.write(frame)
            print()
            print(f"Test Play {i+1}: {total_reward}")
            print(f"Steps:", steps)
            print()


def main():
    N_EPISODES = 10000
    agent = TD3Agent()
    history = agent.play(n_episode=N_EPISODES)

    print(history)
    plt.plot(range(len(history)), history)
    plt.xlabel("Episodes")
    plt.ylabel("Total Rewards")
    plt.savefig("history/log.png")
    print("save : history/log.png")
    
    # agent.test_play(n=5, monitordir="./history/movie", load_model=True)
    agent.test_play(n=10, monitordir=False, load_model=True)

if __name__ == "__main__":
    main()