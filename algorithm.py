from gym.wrappers import Monitor
import matplotlib.pyplot as plt
import datetime


class Algo:
    EPISODES = 3000

    def _init_(self, env):
        self.now_str = datetime.datetime.now().strftime("%d-%m_%H-%M-%S")
        self.env = self.wrap_env(env)

        self.my_graph = plt.figure()
        self.xval, self.yval = [], []
        self.sub_plot = self.my_graph.add_subplot()
        plt.xlabel("Episode #")
        plt.ylabel("Score")
        plt.title("Scores vs Episode")
        self.plot_line, = self.sub_plot.plot(self.xval, self.yval)
        self.sub_plot.set_xlim([0, self.EPISODES])
        self.sub_plot.set_ylim([-220, 100])
        self.DEBUG = False

    def run_all_episodes(self):
        for i in range(1, self.EPISODES + 1):
            ep_score = self.run_algo_step(i)
            print("Episode finished. Now plotting..")
            self.plot_episode(self.my_graph, self.xval, self.yval, ep_score, self.plot_line, i)

    def plot_episode(self, graph, xval, yval, ep_score, plot_line, i):
        xval.append(i)
        yval.append(ep_score)
        plot_line.set_xdata(xval)
        plot_line.set_ydata(yval)
        graph.savefig(f"./plots/{self.now_str}/graph.png")

    def wrap_env(self, env):
        env = Monitor(env, f'./plots/{self.now_str}/video', video_callable=lambda episode_id: episode_id % 50 == 0,
                      force=True)
        return env

    def run_algo_step(self, i):
        pass