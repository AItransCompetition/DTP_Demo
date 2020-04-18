"""
This demo aims to help player running system quickly by using the pypi library DTP-Emualtor https://pypi.org/project/DTP-Emulator/.
"""
from simple_emulator import PccEmulator, CongestionControl, create_2flow_emulator

# We provided a simple algorithms about packet selection to help you being familiar with this competition.
# In this example, it will select the packet according to packet's created time first and radio of rest life time to deadline secondly.
from simple_emulator import Packet_selection

# We provided some simple algorithms about congestion control to help you being familiar with this competition.
# Like Reno and an example about reinforcement learning implemented by tensorflow
from simple_emulator import Reno
# Ensuring that you have installed tensorflow before you use it
# from simple_emulator import RL

# We provided some function of plotting to make you analyze result easily in utils.py
from simple_emulator import analyze_pcc_emulator, plot_cwnd, plot_throughput
from simple_emulator import constant

# Your solution should include packet selection and congestion control.
# So, we recommend you to achieve it by inherit the objects we provided and overwritten necessary method.
class MySolution(Packet_selection, Reno):

    def __init__(self):
        # base parameters in CongestionControl
        self._input_list = []
        self.cwnd = 1
        self.send_rate = float("inf")
        self.pacing_rate = float("inf")
        self.call_nums = 0

        # for reno
        self.ssthresh = float("inf")
        self.curr_state = "slow_start"
        self.states = ["slow_start", "congestion_avoidance", "fast_recovery"]
        self.drop_nums = 0
        self.ack_nums = 0

        self.cur_time = -1
        self.last_cwnd = 0
        self.instant_drop_nums = 0

    def select_packet(self, cur_time, packet_queue):
        """
        The algorithm to select which packet in 'packet_queue' should be sent at time 'cur_time'.
        The following example is selecting packet by the create time firstly, and radio of rest life time to deadline secondly.
        See more at https://github.com/Azson/DTP-emulator/tree/pcc-emulator#packet_selectionpy.
        :param cur_time: float
        :param packet_queue: the list of Packet.You can get more detail about Block in objects/packet.py
        :return: int
        """
        def is_better(packet):
            if best_packet.create_time != packet.create_time:
                return best_packet.create_time > packet.create_time
            return (cur_time - packet.create_time) * best_packet.block_info["Deadline"] > \
                    (cur_time - best_packet.create_time) * packet.block_info["Deadline"]

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
        See more at https://github.com/Azson/DTP-emulator/tree/pcc-emulator#congestion_control_algorithmpy.
        """
        return super().make_decision(cur_time)

    def cc_trigger(self, data):
        packet_type = data["packet_type"]
        event_time = data["event_time"]

        if self.cur_time < event_time:
            self.last_cwnd = 0
            self.instant_drop_nums = 0

        if packet_type == constant.PACKET_TYPE_DROP:
            if self.instant_drop_nums > 0:
                return
            self.instant_drop_nums += 1
            self.curr_state = self.states[2]
            self.drop_nums += 1
            self.ack_nums = 0
            # Ref 1 : For ensuring the event type, drop or ack?
            self.cur_time = event_time
            if self.last_cwnd > 0 and self.last_cwnd != self.cwnd:
                self.cwnd = self.last_cwnd
                self.last_cwnd = 0

        elif packet_type == constant.PACKET_TYPE_FINISHED:
            # Ref 1
            if event_time <= self.cur_time:
                return
            self.cur_time = event_time
            self.last_cwnd = self.cwnd

            self.ack_nums += 1
            if self.curr_state == self.states[0]:
                if self.ack_nums == self.cwnd:
                    self.cwnd *= 2
                    self.ack_nums = 0
                if self.cwnd >= self.ssthresh:
                    self.curr_state = self.states[1]

            elif self.curr_state == self.states[1]:
                if self.ack_nums == self.cwnd:
                    self.cwnd += 1
                    self.ack_nums = 0

        if self.curr_state == self.states[2]:
            self.ssthresh = max(self.cwnd // 2, 1)
            self.cwnd = self.ssthresh
            self.curr_state = self.states[1]

    def append_input(self, data):
        """
        The part of algorithm to make congestion control, which will be call when sender get an event about acknowledge or lost from reciever.
        See more at https://github.com/Azson/DTP-emulator/tree/pcc-emulator#congestion_control_algorithmpy.
        """
        self._input_list.append(data)

        if data["packet_type"] != constant.PACKET_TYPE_TEMP:
            self.cc_trigger(data)
            return {
                "cwnd" : self.cwnd,
                "send_rate" : self.send_rate
            }
        return None


# The file path of packets' log
log_packet_file = "output/packet_log/packet-0.log"

# Use the object you created above
my_solution = MySolution()

# Create the emulator using your solution
# Specify USE_CWND to decide whether or not use crowded windows. USE_CWND=True by default.
# Specify ENABLE_LOG to decide whether or not output the log of packets. ENABLE_LOG=True by default.
# You can get more information about parameters at https://github.com/Azson/DTP-emulator/tree/pcc-emulator#constant
emulator = create_2flow_emulator(
    block_file="../traces/block.txt",
    trace_file="../traces/trace.txt",
    solution=my_solution
)

# Run the emulator and you can specify the time for the emualtor's running.
# It will run until there is no packet can sent by default.
emulator.run_for_dur(20)

# print the debug information of links and senders
emulator.print_debug()

# Output the picture of pcc_emulator-analysis.png
# You can get more information from https://github.com/Azson/DTP-emulator/tree/pcc-emulator#pcc_emulator-analysispng.
analyze_pcc_emulator(log_packet_file, file_range="all", sender=[1])

# Output the picture of cwnd_changing.png
# You can get more information from https://github.com/Azson/DTP-emulator/tree/pcc-emulator#cwnd_changingpng
plot_cwnd(log_packet_file, file_range="all", sender=[1])