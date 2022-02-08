from gym.wrappers import Monitor
import matplotlib.pyplot as plt
import datetime
import pandas as pd

class Algo:
    EPISODES = 3000

    def __init__(self, env):
        self.max_steps = 2500
        self.now_str = datetime.datetime.now().strftime("%d-%m_%H-%M-%S")
        env._max_episode_steps = self.max_steps
        self.env = self.wrap_env(env)
        self.score_graph = plt.figure()
        self.xval, self.yval = [], []
        self.sub_plot = self.score_graph.add_subplot()
        plt.xlabel("Episode #")
        plt.ylabel("Score")
        plt.title("Scores vs Episode")
        self.plot_line, = self.sub_plot.plot(self.xval, self.yval)
        self.sub_plot.set_xlim([0, self.EPISODES])
        self.sub_plot.set_ylim([-220, 400])
        self.x_label_title = "Episode #"
        self.DEBUG = False
        self.high_score = -200

    def run_all_episodes(self):
        for i in range(1, self.EPISODES + 1):
            ep_score = self.run_algo_step(i)
            print("Episode finished. Now plotting..")
            self.plot_episode(ep_score, i)

    def plot_episode(self, ep_score, i):
        self.xval.append(i)
        self.yval.append(ep_score)
        self.plot_line.set_xdata(self.xval)
        self.plot_line.set_ydata(self.yval)
        moving_avg = pd.Series(self.yval, index=self.xval).rolling(100, min_periods=1).mean()
        self.sub_plot.plot(self.xval, moving_avg, "--k")
        self.score_graph.savefig(f"./plots/{self.now_str}/score_graph.png")

    def wrap_env(self, env):
        env = Monitor(env, f'./plots/{self.now_str}/video', video_callable=lambda episode_id: episode_id % 50 == 0,
                      force=True)
        return env

    def run_algo_step(self, i):
        pass