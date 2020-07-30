"""
This demo aims to help player running system quickly by using the pypi library DTP-Emualtor https://pypi.org/project/DTP-Emulator/.
"""
from simple_emulator import PccEmulator, CongestionControl

# We provided a simple algorithms about packet selection to help you being familiar with this competition.
# In this example, it will select the packet according to packet's created time first and radio of rest life time to deadline secondly.
from simple_emulator import Packet_selection

# We provided some function of plotting to make you analyze result easily in utils.py
from simple_emulator import analyze_pcc_emulator, plot_cwnd, plot_rate

from simple_emulator import constant
from simple_emulator import cal_qoe
from config.constant import *
# from utils import debug_print
from objects.cc_base import CongestionControl
import numpy as np;

# for tf version < 2.0
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random

np.random.seed(2)
torch.manual_seed(1)

EVENT_TYPE_FINISHED='F'
EVENT_TYPE_DROP='D'
EVENT_TYPE_TEMP='T'

class Net(nn.Module):
    '''
    可指定大小的3层torch NN
    '''
    def __init__(self,
                 N_STATES=10,
                 N_ACTIONS=5,
                 N_HIDDEN=30):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(N_STATES, N_HIDDEN)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(N_HIDDEN, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self,
                 N_STATES=10,
                 N_ACTIONS=5,
                 LR=0.01,
                 GAMMA=0.9,
                 TARGET_REPLACE_ITER=100,
                 MEMORY_CAPACITY=200,
                 BATCH_SIZE=32
                 ):

        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
        self.LR = LR
        # self.EPSILON = EPSILON
        self.GAMMA = GAMMA
        self.TARGET_REPLACE_ITER = TARGET_REPLACE_ITER
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.BATCH_SIZE = BATCH_SIZE

        self.eval_net = Net(self.N_STATES, self.N_ACTIONS)
        self.target_net = Net(self.N_STATES, self.N_ACTIONS)

        self.learn_step_counter = 0  # 用于 target 更新计时
        self.memory_counter = 0  # 记忆库记数
        self.memory = np.zeros((self.MEMORY_CAPACITY, self.N_STATES * 2 + 2))  # 初始化记忆库
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # torch 的优化器
        self.loss_func = nn.MSELoss()  # 误差公式


    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        # 这里只输入一个 sample
        actions_value = self.eval_net.forward(x)
        action = torch.max(actions_value, 1)[1].data.numpy()[0]  # return the argmax
        return action


    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):
        # target net 参数更新
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :self.N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, self.N_STATES+1:self.N_STATES+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -self.N_STATES:]))

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)

        q_next = self.target_net(b_s_).detach()  # q_next 不进行反向传递误差, 所以 detach
        q_target = b_r + self.GAMMA * q_next.max(1)[0].reshape(-1, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# change every EPISODE times
EPISODE = 50 
# send rate, lost rate *2, rtt sample
N_F = 1 + 2 + 2 + EPISODE
# 1.4,1.1,0.4
N_A = 3 
# random choose
Lambda_init = 0.9
# decline Lambda after random_counter 
random_counter_init = 100 
# standardlize to 1
MAX_BANDWITH = 10000 

class RL(CongestionControl):

    def __init__(self):
        super(RL, self).__init__()
        self.USE_CWND=False
        self.send_rate = 500.0

        self.counter = 0 # EPISODE counter

        self.result_list = []
        self.latency_list = []
        self.last_state = []
        self.last_action = 1
        self.Lambda = Lambda_init
        self.random_counter = random_counter_init

        self.dqn = DQN(N_STATES=N_F,
                        N_ACTIONS=N_A,
                        LR=0.01,
                        GAMMA=0.9,
                        TARGET_REPLACE_ITER=100,
                        MEMORY_CAPACITY=500,
                        BATCH_SIZE=32
                    )

        self.last_state.append(self.send_rate/MAX_BANDWITH)
        self.last_state.extend([0]*4)
        for i in range(EPISODE):
            self.last_state.append(0)
        
        # update in 2020-7-30
        self.event_nums = 0
        self.event_lost_nums = 0
        self.event_ack_nums = 0

    def cc_trigger(self, data):

        event_type = data["event_type"]
        event_time = data["event_time"]
        self.latency_list.append(data["packet_information_dict"]["Latency"])
        self.event_nums += 1

        if event_type == EVENT_TYPE_DROP:
            self.result_list.append(1)
            self.event_lost_nums += 1
        else:
            self.result_list.append(0)
            self.event_ack_nums += 1

        self.counter += 1
        if self.counter == EPISODE: # choose action every EPISODE times
            self.counter = 0
            # print()
            # print("EPISODE: send_rate is {}".format(self.send_rate))
            # print()
            # loss rate
            sum_loss_rate = self.event_lost_nums / self.event_nums
            instant_packet = list(filter(lambda item: self._input_list[-1]["event_time"] - item["event_time"] < 1., self._input_list))
            instant_loss_nums = sum([1 for data in instant_packet if data["event_type"] == 'D']) 
            instant_loss_rate = instant_loss_nums / len(instant_packet) if len(instant_packet) > 0 else 0
            # throughput
            sum_rate = self.event_ack_nums / event_time
            instant_rate = (len(instant_packet) - instant_loss_nums) / (instant_packet[-1]["event_time"] - instant_packet[0]["event_time"]) if len(instant_packet) > 1 else 0

            # declining random rate
            self.random_counter-=1
            if self.random_counter <= 0:
                self.Lambda /=2.0
                if self.Lambda < 0.05:
                    self.Lambda = 0.05
                self.random_counter = 50

            # reward
            r = 0
            for i in range(EPISODE):
                if self.result_list[i] == 0:
                    r += self.send_rate * (i+1)
                else:
                    r -= self.send_rate * (i+1)

            # current_state
            s_ = []
            s_.append(self.send_rate/MAX_BANDWITH)
            s_.append(sum_loss_rate)
            s_.append(instant_loss_rate)
            s_.append(sum_rate)
            s_.append(instant_rate)
            for i in range(EPISODE):
                s_.append(self.latency_list[i])
            s_array=np.array(s_)

            # store
            self.dqn.store_transition(self.last_state, self.last_action, r, s_array)
            # choose action
            a = self.dqn.choose_action(s_array)
           
            # exploration
            if random.random() < self.Lambda:
                a = random.randint(0,2)
            if self.send_rate - 500 < 0.00001:
                self.send_rate = MAX_BANDWITH / 2
                a = 0
            elif MAX_BANDWITH - self.send_rate < 1.0000001:
                self.send_rate = MAX_BANDWITH / 2
                a = 2
            elif a == 0:
                self.send_rate *= 1.4
            elif a == 1:
                self.send_rate *= 1.
            else:
                self.send_rate *= 0.4
            self.last_action = a
            # DQN learn
            self.dqn.learn()

            self.last_state = s_
            self.result_list = []
            self.latency_list = []

    def append_input(self, data):
        self._input_list.append(data)

        if data["event_type"] != EVENT_TYPE_TEMP:
            self.cc_trigger(data)
            return {
                "cwnd" : self.cwnd,
                "send_rate" : self.send_rate
            }
        return None

# Your solution should include packet selection and congestion control.
# So, we recommend you to achieve it by inherit the objects we provided and overwritten necessary method.
class MySolution(Packet_selection, RL):

    def select_packet(self, cur_time, packet_queue):
        """
        The algorithm to select which packet in 'packet_queue' should be sent at time 'cur_time'.
        The following example is selecting packet by the create time firstly, and radio of rest life time to deadline secondly.
        See more at https://github.com/AItransCompetition/simple_emulator/tree/master#packet_selectionpy.
        :param cur_time: float
        :param packet_queue: the list of Packet.You can get more detail about Block in objects/packet.py
        :return: int
        """
        def is_better(packet):
            best_block_create_time = best_packet.block_info["Create_time"]
            packet_block_create_time = packet.block_info["Create_time"]
            # if packet is miss ddl
            if (cur_time - packet_block_create_time) >= packet.block_info["Deadline"]:
                return False
            if (cur_time - best_block_create_time) >= best_packet.block_info["Deadline"]:
                return True
            if best_block_create_time != packet_block_create_time:
                return best_block_create_time > packet_block_create_time
            return (cur_time - best_block_create_time) * best_packet.block_info["Deadline"] > \
                   (cur_time - packet_block_create_time) * packet.block_info["Deadline"]

        best_packet_idx = -1
        best_packet = None
        for idx, item in enumerate(packet_queue):
            if best_packet is None or is_better(item) :
                best_packet_idx = idx
                best_packet = item

        return best_packet_idx

    def make_decision(self, cur_time):
        """
        The part of algorithm to make congestion control, which will be call when sender need to send pacekt.
        See more at https://github.com/AItransCompetition/simple_emulator/tree/master#congestion_control_algorithmpy.
        """
        return super().make_decision(cur_time)

    def append_input(self, data):
        """
        The part of algorithm to make congestion control, which will be call when sender get an event about acknowledge or lost from reciever.
        See more at https://github.com/AItransCompetition/simple_emulator/tree/master#congestion_control_algorithmpy.
        """
        return super().append_input(data)


if __name__ == '__main__':
    
    # The file path of packets' log
    log_packet_file = "output/packet_log/packet-0.log"

    # Use the object you created above
    my_solution = MySolution()

    # Create the emulator using your solution
    # Specify USE_CWND to decide whether or not use crowded windows. USE_CWND=True by default.
    # Specify ENABLE_LOG to decide whether or not output the log of packets. ENABLE_LOG=True by default.
    # You can get more information about parameters at https://github.com/AItransCompetition/simple_emulator/tree/master#constant
    emulator = PccEmulator(
        block_file=["traces/data_video.csv", "traces/data_audio.csv"],
        trace_file="traces/trace.txt",
        solution=my_solution,
        SEED=1,
        ENABLE_LOG=True
    )

    # Run the emulator and you can specify the time for the emualtor's running.
    # It will run until there is no packet can sent by default.
    emulator.run_for_dur(15)

    # print the debug information of links and senders
    emulator.print_debug()

    # Output the picture of emulator-analysis.png
    # You can get more information from https://github.com/AItransCompetition/simple_emulator/tree/master#emulator-analysispng.
    analyze_pcc_emulator(log_packet_file, file_range="all")

    plot_rate(log_packet_file, trace_file="traces/trace.txt", file_range="all")

    print(cal_qoe())
