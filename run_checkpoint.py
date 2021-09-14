import numpy as np
import torch
import gym

from core.models import Actor

ckpt = torch.load('Results/Auxiliary/Humanoid-v2_best_p10_s2018')
dummy_env = gym.make('Humanoid-v2')

# ckpt = torch.load('Results/Auxiliary/HalfCheetah-v2_best_p10_s2018')
# dummy_env = gym.make('HalfCheetah-v2')

# print(dummy_env.observation_space.shape)
# print(dummy_env.action_space.shape)

model = Actor(state_dim=dummy_env.observation_space.shape[0], action_dim=dummy_env.action_space.shape[0], wwid=0)
model.load_state_dict(ckpt)
model.eval()
model = model.to('cuda')
model.double()

init_state = dummy_env.reset()

state = init_state
with torch.no_grad():
    while True:
        state = torch.from_numpy(np.asarray(state, dtype=np.float64)).to('cuda')
        action = model.forward(state)
        action = action.cpu().detach().numpy()
        action = np.clip(action, dummy_env.action_space.low[0], dummy_env.action_space.high[0])
        state, reward, done, info = dummy_env.step(action)
        dummy_env.render()
        print(f'{reward:.2f}', end=' ')
        if done:
            break