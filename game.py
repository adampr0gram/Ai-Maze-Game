import os
import pygame
import torch
import torch.nn as nn
import numpy as np
from env import MazeEnv, SIZE
from dqn import DQN, ReplayBuffer
import random
import pickle

CELL = 50
FPS = 12

COLORS = {
    "bg": (18, 18, 18),
    "wall": (60, 60, 60),
    "wall_border": (90, 90, 90),
    "player": (255, 215, 0),
    "ai": (80, 160, 255),
    "goal": (0, 220, 120),
    "grid": (35, 35, 35),
    "text": (240, 240, 240),
    "timer_bg": (40, 40, 40),
    "timer_text": (255, 255, 255)
}

pygame.init()
screen = pygame.display.set_mode((SIZE*CELL, SIZE*CELL))
pygame.display.set_caption("Maze DRL AI Game")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 28)
timer_font = pygame.font.SysFont("consolas", 24, bold=True)

env = MazeEnv()
state = env.reset()
winner = None
running = True

# DRL setup
state_dim = len(state)
action_dim = 4
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
buffer = ReplayBuffer()
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
update_target_every = 50
step_count = 0

# Persistent AI memory
MEM_DIR = "ai_mem"
os.makedirs(MEM_DIR, exist_ok=True)
weights_path = os.path.join(MEM_DIR, "ai_weights.pt")
buffer_path = os.path.join(MEM_DIR, "replay_buffer.pkl")

if os.path.exists(weights_path):
    try:
        policy_net.load_state_dict(torch.load(weights_path))
        target_net.load_state_dict(policy_net.state_dict())
        print("Loaded AI weights from memory.")
    except:
        print("AI memory shape mismatch! Resetting memory.")
        os.remove(weights_path)
        if os.path.exists(buffer_path):
            os.remove(buffer_path)

if os.path.exists(buffer_path):
    with open(buffer_path, "rb") as f:
        buffer.buffer = pickle.load(f)
    print("Loaded replay buffer from memory.")

def save_memory():
    torch.save(policy_net.state_dict(), weights_path)
    with open(buffer_path, "wb") as f:
        pickle.dump(buffer.buffer, f)

# Timer variables
win_time = None
countdown = 3
countdown_started = False

def draw_arrow(surface, center, direction, color):
    cx, cy = center
    dx, dy = direction
    if dx==0 and dy==0:
        return
    tip = (cx + dx*12, cy + dy*12)
    left = (cx - dy*6, cy + dx*6)
    right = (cx + dy*6, cy - dx*6)
    pygame.draw.polygon(surface, color, [tip, left, right])

def select_action(state):
    global epsilon
    if random.random() < epsilon:
        return random.randint(0,3)
    state_t = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        qvals = policy_net(state_t)
    return int(torch.argmax(qvals).item())

def train():
    if len(buffer) < batch_size:
        return
    s,a,r,s2,d = buffer.sample(batch_size)
    s = torch.FloatTensor(s)
    s2 = torch.FloatTensor(s2)
    a = torch.LongTensor(a)
    r = torch.FloatTensor(r)
    d = torch.FloatTensor(d)
    q_values = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        q_next = target_net(s2).max(1)[0]
        q_target = r + gamma*(1-d)*q_next
    loss = nn.MSELoss()(q_values, q_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ================= GAME LOOP =================
while running:
    clock.tick(FPS)
    step_count += 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            save_memory()
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_r]:
        state = env.reset()
        winner = None
        win_time = None
        countdown_started = False

    # Player + AI moves
    if winner is None:
        # Player
        dx,dy = 0,0
        if keys[pygame.K_w] or keys[pygame.K_UP]: dy=-1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: dy=1
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: dx=-1
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: dx=1
        if dx!=0 or dy!=0:
            action_map = {(0,-1):0, (0,1):1, (-1,0):2, (1,0):3}
            state, reward, done, winner_temp = env.step_player(action_map[(dx,dy)])
            if winner_temp:
                winner = f"{winner_temp} WINS!"
                win_time = pygame.time.get_ticks()
                save_memory()

        # AI
        action = select_action(state)
        next_state, reward, done, winner_temp = env.step_ai(action)
        buffer.push(state, action, reward, next_state, done)
        train()
        state = next_state
        if winner_temp:
            winner = f"{winner_temp} WINS!"
            win_time = pygame.time.get_ticks()
            save_memory()

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if step_count % update_target_every == 0:
            target_net.load_state_dict(policy_net.state_dict())

    # DRAW
    screen.fill(COLORS["bg"])
    # Grid
    for x in range(SIZE):
        for y in range(SIZE):
            pygame.draw.rect(screen, COLORS["grid"], (x*CELL, y*CELL, CELL, CELL), 1)
    # Walls
    for x in range(SIZE):
        for y in range(SIZE):
            if env.grid[x, y]==1:
                rect = pygame.Rect(x*CELL, y*CELL, CELL, CELL)
                pygame.draw.rect(screen, COLORS["wall"], rect)
                pygame.draw.rect(screen, COLORS["wall_border"], rect, 2)
    # Goal
    gx,gy = env.goal
    goal_rect = pygame.Rect(gx*CELL, gy*CELL, CELL, CELL)
    pygame.draw.rect(screen, COLORS["goal"], goal_rect)
    pygame.draw.rect(screen, (0,255,160), goal_rect, 3)
    # Player
    px,py = env.player
    player_rect = pygame.Rect(px*CELL+5, py*CELL+5, CELL-10, CELL-10)
    pygame.draw.rect(screen, COLORS["player"], player_rect, border_radius=6)
    draw_arrow(screen, player_rect.center, env.player_dir, (0,0,0))
    # AI
    ax,ay = env.ai
    ai_rect = pygame.Rect(ax*CELL+5, ay*CELL+5, CELL-10, CELL-10)
    pygame.draw.rect(screen, COLORS["ai"], ai_rect, border_radius=6)
    draw_arrow(screen, ai_rect.center, env.ai_dir, (255,255,255))

    # Winner + timer
    if winner:
        text = font.render(winner, True, COLORS["text"])
        screen.blit(text, text.get_rect(center=(SIZE*CELL//2, SIZE*CELL//2)))

        if win_time:
            elapsed = (pygame.time.get_ticks() - win_time)/1000
            if elapsed >=5 and not countdown_started:
                countdown_started = True
                countdown_start = pygame.time.get_ticks()
            if countdown_started:
                remaining = max(0, countdown - (pygame.time.get_ticks() - countdown_start)/1000)
                rect_w, rect_h = 180, 40
                rect_x = SIZE*CELL//2 - rect_w//2
                rect_y = 10
                pygame.draw.rect(screen, COLORS["timer_bg"], (rect_x, rect_y, rect_w, rect_h), border_radius=8)
                text_cd = timer_font.render(f"Resetting in {int(remaining)+1}...", True, COLORS["timer_text"])
                text_rect = text_cd.get_rect(center=(rect_x + rect_w//2, rect_y + rect_h//2))
                screen.blit(text_cd, text_rect)
                if remaining <= 0:
                    state = env.reset()
                    winner = None
                    win_time = None
                    countdown_started = False

    pygame.display.flip()

pygame.quit()
