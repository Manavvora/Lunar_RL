import torch
import gym
import sys
import numpy as np
from deepq_network import LinearMapNet
from utils import epsilon_greedy


def main():
    _, episodes, model_path = sys.argv
    env = gym.make('LunarLander-v2')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device=torch.device('cpu')
    qnet = LinearMapNet(8, 4).to(device)
    qnet.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    qnet.eval()

    for episode in range(int(episodes)):
        episode_reward = 0
        curr_state, done = env.reset(), False
        print(np.shape(curr_state))
        # curr_state = np.expand_dims(curr_state, 0)
        curr_state=np.asarray(curr_state)
        curr_state = curr_state.reshape((1,2))
        print(np.shape(curr_state))
        #
        # curr_state=np.vstack(curr_state).astype(np.float)
        print(curr_state[-1][-1])
        print(curr_state)
        curr_state=curr_state[0][:-1]
        print("--------")
        print(curr_state)

        curr_state = torch.from_numpy(curr_state)

        while not done:
            env.render()
            action = epsilon_greedy(qnet, curr_state.to(device), 0.0001, 4)
            next_state, reward, done, _ = env.step(action)
            next_state = np.expand_dims(next_state, 0)
            next_state = torch.from_numpy(next_state)
            episode_reward += reward
            curr_state = next_state

        print(f"Episode reward: {episode_reward}")


if __name__ == '__main__':
    main()

