from game import Interceptor_V2
from game.Interceptor_V2 import Init, Draw, Game_step, World
import numpy as np
import time
import matplotlib.pyplot as plt


action_button = 0
scores = []
for game in range(50):
    Init()
    score = 0
    ang = 0
    for stp in range(1000):
        action_button = action_button ^ 3#np.random.randint(0, 4)
        r_locs, i_locs, c_locs, ang, score = Game_step(3 if ang >= 65 else 2)

        # Draw()
        # print("Step: " + str(stp))
        # print("r_locs: " + str(r_locs))
        # print("i_locs: " + str(i_locs))
        # print("c_locs: " + str(c_locs))
        # print("ang: " + str(ang))
        # print("score: " + str(score))
    print("Game: " + str(game) + ", resulted in score: " + str(score))
    scores.append(score)

print("Average result for games is: " + str(np.average(scores)))