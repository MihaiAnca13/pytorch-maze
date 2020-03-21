import numpy as np
from random import shuffle, randrange
import cv2
from PIL import Image

CRASH_PENALTY = -5
MOVE_PENALTY = -1
TARGET_REACHED_REWARD = 25
MAX_STEPS_PER_EPISODE = 200

COLORS = {'AGENT': (100, 255, 0),
          'TARGET': (25, 0, 255),
          'PATH': (255, 0, 25)}


class Maze:
    def __init__(self, width=50, height=50):
        self.MAX_X = height
        self.MAX_Y = width
        self.x = 0
        self.y = 0
        self.previous_x = 1
        self.previous_y = 1
        # generate maze
        self.maze = None
        self.solution = None
        self.generate_maze(width, height)
        self.maze[0][0] = 0.5

    def generate_maze(self, w, h):
        self.maze = np.zeros((h, w))

        the_maze = set()
        # first_goal = self.get_random(the_maze)
        first_goal = (self.MAX_X - 1, self.MAX_Y - 1)
        the_maze.add(first_goal)

        count = 0
        while len(the_maze) != h * w:
            path = list()

            if count > 0:
                current = self.get_random(the_maze)
            else:
                current = (0, 0)
            path.append(current)

            while current not in the_maze:
                neighbours = self.get_neighbours(current, validation=False)
                index = np.random.randint(len(neighbours))
                current = neighbours[index]

                if current not in path:
                    path.append(current)
                else:
                    index = path.index(current)
                    path = path[:index + 1]

            if count == 0:
                self.solution = path

            for cell in path:
                neighbours = self.get_neighbours(cell, validation=False)
                for neighbour in neighbours:
                    if neighbour not in path and neighbour not in the_maze:
                        self.maze[neighbour] = 1
                the_maze.add(cell)
            self.maze[path[-1]] = 0

            count += 1

    def get_random(self, maze):
        full_maze = [(i, j) for i in range(self.MAX_X) for j in range(self.MAX_Y)]
        pick_from = list(set(full_maze) - set(maze))
        index = np.random.randint(len(pick_from))
        return pick_from[index]

    def get_neighbours(self, current, validation=True):
        neighbours = []
        x = current[0]
        y = current[1]
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 or j == 0:
                    if 0 <= x + i < self.MAX_X and 0 <= y + j < self.MAX_Y:
                        if validation:
                            if self.maze[x + i][y + j] == 0:
                                neighbours.append((x + i, y + j))
                        else:
                            neighbours.append((x + i, y + j))
        return neighbours

    def verify_action(self):
        # 0 means wall
        reward = None
        if self.y < 0 or self.x < 0 or self.y >= self.MAX_Y or self.x >= self.MAX_X or self.maze[self.x][self.y] == 1:
            # restore previous position
            self.x = self.previous_x
            self.y = self.previous_y
            reward = CRASH_PENALTY
        elif self.x == self.MAX_X - 1 and self.y == self.MAX_Y - 1:
            reward = TARGET_REACHED_REWARD
        else:
            reward = MOVE_PENALTY


        self.maze[self.previous_x][self.previous_y] = 0
        self.maze[self.x][self.y] = 0.5
        return reward

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
        self.previous_x = self.x
        self.previous_y = self.y

        # if no x/y, move randomly
        if x is not None and y is not None:
            self.x += x
            self.y += y
        else:
            i = np.random.randint(0, 4)
            x_a, y_a = [(0, 1), (1, 0), (0, -1), (-1, 0)][i]
            self.x += x_a
            self.y += y_a

    def step(self, action, step_count, show=False):
        self.action(action)
        reward = self.verify_action()

        done = False
        if (self.x == self.MAX_X - 1 and self.y == self.MAX_Y - 1) or step_count >= MAX_STEPS_PER_EPISODE:
            done = True

        if show:
            self.show_maze()

        return self.get_state(), reward, done

    def show_maze(self):
        img = np.zeros((self.MAX_X, self.MAX_Y, 3), dtype=np.uint8)
        for i in range(3):
            img[:, :, i] = (1 - self.maze) * 255.0

        # if show_solution:
        #     for x, y in self.solution:
        #         img[x][y] = COLORS['PATH']

        img[self.x][self.y] = COLORS['AGENT']
        img[self.MAX_X - 1][self.MAX_Y - 1] = COLORS['TARGET']

        img = Image.fromarray(img, 'RGB')

        img_size_x = self.MAX_X * 10
        img_size_y = self.MAX_Y * 10
        resized_img = img.resize((img_size_x, img_size_y))
        cv2.imshow("image", np.array(resized_img))
        cv2.waitKey(1)

    def reset(self):
        self.x = 0
        self.y = 0
        self.previous_x = 1
        self.previous_y = 1
        # generate maze
        self.generate_maze(self.MAX_Y, self.MAX_X)
        self.maze[0][0] = 0.5

        return self.get_state()

    def get_state(self):
        return self.maze


if __name__ == "__main__":
    m = Maze()
    # a = m.step(show=True, delay=10000, show_solution=True)
    # print(a)
    m.show_maze()
