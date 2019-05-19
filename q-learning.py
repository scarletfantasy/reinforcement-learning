import gym,sys,numpy as np
from gym.envs.registration import register

env = gym.make('FrozenLake-v0')
max_episode=10000
iter_max=1000
eps=0.4
gama=1
learningrate=0.01
q_value=np.zeros([env.observation_space.n,env.action_space.n])
for episode in range(max_episode):
    state=env.reset()
    for iter in range(iter_max):
        if np.random.uniform(0,1)<eps :
            action=np.random.choice(env.action_space.n)
        else:
            action=np.argmax(q_value[state,:])
        next_state,reward,terminal,_=env.step(action)
        if np.random.uniform(0,1)<eps:
            q_value[state,action]=q_value[state,action]+learningrate*(reward+gama*q_value[next_state,np.random.choice(env.action_space.n)]-q_value[state,action])
        else:
            q_value[state,action]=q_value[state,action]+learningrate*(reward+gama*np.max(q_value[next_state,:])-q_value[state,action])
        if terminal:
            break
        state=next_state
total_reward=1
for episode in range(max_episode):
    state=env.reset()
    for iter in range(iter_max):
        action=np.argmax(q_value[state,:])
        next_state,reward,terminal,_=env.step(action)

        if terminal:
            if reward==1:
                total_reward+=1
            break
        state=next_state
print(total_reward/10000)


