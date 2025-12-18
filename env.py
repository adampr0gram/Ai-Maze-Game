import numpy as np
import random

SIZE = 10  # maze size

class MazeEnv:
    def __init__(self):
        self.grid = np.ones((SIZE, SIZE), dtype=int)
        self.player = (0,0)
        self.ai = (SIZE-1,SIZE-1)
        self.goal = (SIZE//2,SIZE//2)
        self.player_dir = (1,0)  # initial arrow
        self.ai_dir = (-1,0)     # initial arrow
        self.generate_maze()

    def generate_maze(self):
        # DFS maze generation
        self.grid.fill(1)
        visited = np.zeros((SIZE,SIZE), dtype=bool)

        def dfs(x,y):
            visited[x,y] = True
            self.grid[x,y] = 0
            dirs = [(0,-1),(0,1),(-1,0),(1,0)]
            random.shuffle(dirs)
            for dx,dy in dirs:
                nx,ny = x+dx*2, y+dy*2
                if 0<=nx<SIZE and 0<=ny<SIZE and not visited[nx,ny]:
                    self.grid[x+dx,y+dy] = 0
                    dfs(nx,ny)

        # Start DFS from player
        dfs(self.player[0], self.player[1])

        # Ensure player, AI, goal are empty
        self.grid[self.player] = 0
        self.grid[self.goal] = 0
        self.grid[self.ai] = 0

        # Ensure AI has at least one free neighbor
        x,y = self.ai
        neighbors = [(x+dx,y+dy) for dx,dy in [(0,-1),(0,1),(-1,0),(1,0)] if 0<=x+dx<SIZE and 0<=y+dy<SIZE]
        if all(self.grid[nx,ny]==1 for nx,ny in neighbors):
            nx,ny = random.choice(neighbors)
            self.grid[nx,ny] = 0

    def step_player(self, action):
        dx,dy = [(0,-1),(0,1),(-1,0),(1,0)][action]
        nx, ny = self.player[0]+dx, self.player[1]+dy
        if 0 <= nx < SIZE and 0 <= ny < SIZE and self.grid[nx,ny]==0:
            self.player = (nx, ny)
            self.player_dir = (dx,dy)
        winner = None
        if self.player == self.goal:
            winner = "PLAYER"
        done = winner is not None
        return self.get_state(), 0, done, winner

    def step_ai(self, action):
        dx,dy = [(0,-1),(0,1),(-1,0),(1,0)][action]
        nx, ny = self.ai[0]+dx, self.ai[1]+dy
        if 0 <= nx < SIZE and 0 <= ny < SIZE and self.grid[nx,ny]==0:
            self.ai = (nx, ny)
            self.ai_dir = (dx,dy)
        winner = None
        if self.ai == self.goal:
            winner = "AI"
        done = winner is not None
        return self.get_state(), 0, done, winner

    def get_state(self):
        flat_grid = self.grid.flatten()
        return np.array([*self.player,*self.ai,*self.goal, *flat_grid], dtype=float)

    def reset(self):
        self.player = (0,0)
        self.ai = (SIZE-1,SIZE-1)
        self.goal = (SIZE//2,SIZE//2)
        self.player_dir = (1,0)
        self.ai_dir = (-1,0)
        self.generate_maze()
        return self.get_state()
