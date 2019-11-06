from maze import Maze
import numpy as np

SIZE = (41, 41)

env = Maze(SIZE[0], SIZE[1])
env.show_maze(10000)