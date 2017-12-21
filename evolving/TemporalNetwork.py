from Network import *


class TemporalNetwork(object):
    SECONDS_PER_DAY = 24 * 60 * 60

    def __init__(self, temporal_edge_list, t):
        """

        :param temporal_edge_list: a list of edges in the format:
                    [str(source), str(dest), int(birthtime in seconds)]
        """
        self.temporal_edge_list = temporal_edge_list
        self.t = t

    def get_static_graph(self, second):
        # can define time windows here
        window_start = second
        window_end = second + 2*(self.SECONDS_PER_DAY)
        alive = lambda x: x[2] < window_end and x[2] >= window_start

        alive_edge_list = [[x[0], x[1]] for x in self.temporal_edge_list if alive(x)]
        G = Network(local_list=alive_edge_list)

        return G

    def get_network_lifespan(self):
        max_time = max([x[2] for x in self.temporal_edge_list])

        return max_time

