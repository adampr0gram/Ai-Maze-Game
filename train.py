import torch
import random
import numpy as np
from env import MazeEnv
from dqn import DQN

env = MazeEnv()
state_size = len(env.state())
actions = 4

model = DQN(state_size, actions)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

epsilon = 1.0

for episode in range(500):
    state = env.reset()
    done = False

    while not done:
        if random.random() < epsilon:
            action = random.randint(0,3)
        else:
            with torch.no_grad():
                action = torch.argmax(model(torch.FloatTensor(state))).item()

        next_state, reward, done = env.step_ai(action)

        target = reward
        if not done:
            target += 0.99 * torch.max(model(torch.FloatTensor(next_state)))

        prediction = model(torch.FloatTensor(state))[action]
        loss = loss_fn(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

    epsilon = max(0.1, epsilon * 0.995)
    print(f"Episode {episode}")
