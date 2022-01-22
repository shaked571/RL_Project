import numpy as np
import random
import gym
import math
from gym.wrappers import Monitor

from collections import defaultdict
import matplotlib.pyplot as graph
import datetime

now_str = datetime.datetime.now().strftime("%d-%m_%H-%M-%S")

EPISODES = 1001
GAMMA = 0.99
ALPHA = 0.01
HIGHSCORE = -200
INF = 1000
stateBounds = [(0, 2 * math.pi),  # hull_angle
               (-INF, INF),  # hull_angularVelocity
               (-1, 1),  # vel_x
               (-1, 1),  # vel_y
               (-INF, INF),  # hip_joint_1_angle
               (-INF, INF),  # hip_joint_1_speed
               (-INF, INF),  # knee_joint_1_angle
               (-INF, INF),  # knee_joint_1_speed
               (0, 1),  # leg_1_ground_contact_flag
               (-INF, INF),  # hip_joint_2_angle
               (-INF, INF),  # hip_joint_2_speed
               (-INF, INF),  # knee_joint_2_angle
               (-INF, INF),  # knee_joint_2_speed
               (0, 1),  # leg_2_ground_contact_flag
               (-INF, INF),  # lidar 1
               (-INF, INF),  # lidar 2
               (-INF, INF),  # lidar 3
               (-INF, INF),  # lidar 4
               (-INF, INF),  # lidar 5
               (-INF, INF),  # lidar 6
               (-INF, INF),  # lidar 7
               (-INF, INF),  # lidar 8
               (-INF, INF),  # lidar 9
               (-INF, INF)  # lidar 10
               ]

actionBounds = (-1, 1)


def updateQTable(Qtable, state, action, reward, nextState=None):
    global ALPHA
    global GAMMA

    current = Qtable[state][action]
    qNext = np.max(Qtable[nextState]) if nextState is not None else 0
    target = reward + (GAMMA * qNext)
    new_value = current + (ALPHA * (target - current))
    return new_value


def getNextAction(qTable, epsilon, state):
    if random.random() < epsilon:

        action = ()
        for i in range(0, 4):
            action += (random.randint(0, 9),)

    else:

        action = np.unravel_index(np.argmax(qTable[state]), qTable[state].shape)

    return action


def discretizeState(state):
    discreteState = []

    for i in range(len(state)):
        index = int((state[i] - stateBounds[i][0]) / (stateBounds[i][1] - stateBounds[i][0]) * 19)
        discreteState.append(index)

    return tuple(discreteState)


def convertNextAction(nextAction):
    action = []

    for i in range(len(nextAction)):
        nextVal = nextAction[i] / 9 * 2 - 1

        action.append(nextVal)

    return tuple(action)


def plotEpisode(myGraph, xval, yval, epScore, plotLine, i):
    xval.append(i)
    yval.append(epScore)

    plotLine.set_xdata(xval)
    plotLine.set_ydata(yval)
    myGraph.savefig(f"./plots/{now_str}/graph.png")


def runAlgorithmStep(env, i, qTable):
    global HIGHSCORE

    print("Episode #: ", i)
    env_state = env.reset()
    state = discretizeState(env_state)
    total_reward = 0
    epsilon = 1.0 / (i * .004)
    steps = 0
    while True:

        if i == 1000:
            break
        steps += 1
        nextAction = convertNextAction(getNextAction(qTable, epsilon, state))
        nextActionDiscretized = getNextAction(qTable, epsilon, state)
        nextState, reward, done, info = env.step(nextAction)
        nextState = discretizeState(nextState)
        total_reward += reward
        qTable[state][nextActionDiscretized] = updateQTable(qTable, state, nextActionDiscretized, reward, nextState)
        state = nextState
        if done:
            break

    if total_reward > HIGHSCORE:
        HIGHSCORE = total_reward

    return total_reward


def wrap_env(env):
    env = Monitor(env, f'./plots/{now_str}/video',video_callable=lambda episode_id: episode_id % 50 == 0, force=True)
    return env


def main():
    global HIGHSCORE

    doRender = False
    env = gym.make("BipedalWalker-v3")
    env = wrap_env(env)
    state_size = env.observation_space
    action_space = env.action_space
    qTable = defaultdict(lambda: np.zeros((40, 40, 40, 40)))

    myGraph = graph.figure()
    xval, yval = [], []
    mySubPlot = myGraph.add_subplot()
    graph.xlabel("Episode #")
    graph.ylabel("Score")
    graph.title("Scores vs Episode")
    plotLine, = mySubPlot.plot(xval, yval)
    mySubPlot.set_xlim([0, EPISODES])
    mySubPlot.set_ylim([-220, 100])

    for i in range(1, EPISODES + 1):
        epScore = runAlgorithmStep(env, i, qTable)
        print("Episode finished. Now plotting..")
        plotEpisode(myGraph, xval, yval, epScore, plotLine, i)

    print("All episodes finished. Highest score achieved: " + str(HIGHSCORE))

main()