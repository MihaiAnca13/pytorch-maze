import numpy as np
from random import shuffle, randrange
import cv2
from PIL import Image

CRASH_PENALTY = 100
MOVE_PENALTY = 1
TARGET_REACHED_REWARD = 25

COLORS = {'AGENT': (100, 255, 0),
          'TARGET': (25, 0, 255)}


class Maze:
    def __init__(self, width=81, height=51):
        self.MAX_X = height
        self.MAX_Y = width
        self.x = 1
        self.y = 1
        self.previous_x = 1
        self.previous_y = 1
        # generate maze
        self.maze = None
        self.generate_maze(width, height)

    def generate_maze(self, w, h):
        # Only odd shapes
        vis = [[0] * w + [1] for _ in range(h)] + [[1] * (w + 1)]

        def walk(x, y):
            vis[y][x] = 1

            d = [(x - 1, y), (x, y + 1), (x + 1, y), (x, y - 1)]
            shuffle(d)
            for (xx, yy) in d:
                if vis[yy][xx]: continue
                walk(xx, yy)

        walk(randrange(w), randrange(h))
        self.maze = vis
        a = 1

    def verify_action(self):
        # 0 means wall
        if not self.maze[self.y, self.x]:
            # restore previous position
            self.x = self.previous_x
            self.y = self.previous_y
            return CRASH_PENALTY
        elif self.x in [1, self.MAX_X - 1] or self.y in [1, self.MAX_Y - 1]:
            return TARGET_REACHED_REWARD
        else:
            return MOVE_PENALTY

    def action(self, choice):
        # N, E, S, W => 0, 1, 2, 3
        if choice == 0:
            self.move(x=0, y=-1)
        elif choice == 1:
            self.move(x=1, y=0)
        elif choice == 2:
            self.move(x=0, y=1)
        elif choice == 3:
            self.move(x=-1, y=0)

    def move(self, x=None, y=None):
        # save previous position in case it crashes
        self.previous_x = x
        self.previous_y = y

        # if no x/y, move randomly
        if x:
            self.x += x
        else:
            self.x += np.random.randint(-1, 2)

        if y:
            self.y += y
        else:
            self.y += np.random.randint(-1, 2)

    def show_maze(self, delay=1):
        img = np.zeros((self.MAX_X, self.MAX_Y, 3), dtype=np.uint8)
        for i in range(3):
            img[:, :, i] = (1 - self.maze) * 255.0
        img[self.x][self.y] = COLORS['AGENT']

        # check where exists are
        # for i in range(self.MAX_X):
        #     if img[1][i][0] == 255:
        #         img[1][i] = COLORS['TARGET']
        # for i in range(self.MAX_X):
        #     if img[self.MAX_Y - 1][i][0] == 255:
        #         img[self.MAX_Y - 1][i] = COLORS['TARGET']
        # for j in range(self.MAX_Y):
        #     if img[j][1][0] == 255:
        #         img[j][1] = COLORS['TARGET']
        # for j in range(self.MAX_Y):
        #     if img[j][self.MAX_X - 1][0] == 255:
        #         img[j][self.MAX_X - 1] = COLORS['TARGET']

        # img[self.TARGET] = COLORS['TARGET']
        img = Image.fromarray(img, 'RGB')
        img = img.resize((300, 300))
        cv2.imshow("image", np.array(img))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            return
