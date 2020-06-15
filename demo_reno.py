"""
This demo aims to help player running system quickly by using the pypi library DTP-Emualtor https://pypi.org/project/DTP-Emulator/.
"""
from simple_emulator import PccEmulator, CongestionControl

# We provided a simple algorithms about packet selection to help you being familiar with this competition.
# In this example, it will select the packet according to packet's created time first and radio of rest life time to deadline secondly.
from simple_emulator import Packet_selection

# We provided some simple algorithms about congestion control to help you being familiar with this competition.
# Like Reno and an example about reinforcement learning implemented by tensorflow
from simple_emulator import Reno
# Ensuring that you have installed tensorflow before you use it
# from simple_emulator import RL

# We provided some function of plotting to make you analyze result easily in utils.py
from simple_emulator import analyze_pcc_emulator, plot_rate
from simple_emulator import constant

from simple_emulator import cal_qoe

EVENT_TYPE_FINISHED='F'
EVENT_TYPE_DROP='D'
EVENT_TYPE_TEMP='T'

# Your solution should include packet selection and congestion control.
# So, we recommend you to achieve it by inherit the objects we provided and overwritten necessary method.
class MySolution(Packet_selection, Reno):

    def __init__(self):
        # base parameters in CongestionControl
        # the data appended in function "append_input"
        self._input_list = []
        # the value of crowded window
        self.cwnd = 1
        # the value of sending rate
        self.send_rate = float("inf")
        # the value of pacing rate
        self.pacing_rate = float("inf")
        # use cwnd
        self.USE_CWND=True

        # for reno
        self.ssthresh = float("inf")
        self.curr_state = "slow_start"
        self.states = ["slow_start", "congestion_avoidance", "fast_recovery"]
        # the number of lost packets
        self.drop_nums = 0
        # the number of acknowledgement packets
        self.ack_nums = 0

        # current time
        self.cur_time = -1
        # the value of cwnd at last packet event
        self.last_cwnd = 0
        # the number of lost packets received at the current moment
        self.instant_drop_nums = 0

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
            # select packet which is not missing ddl
            if (cur_time - best_block_create_time) >= best_packet.block_info["Deadline"]:
                return True
            # select packet which is created earlier
            if best_block_create_time != packet_block_create_time:
                return best_block_create_time > packet_block_create_time
            # selct packet which is more urgent
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

    def cc_trigger(self, data):
        event_type = data["event_type"]
        event_time = data["event_time"]

        # see Algorithm design problem in QA section of the official website
        if self.cur_time < event_time:
            # initial parameters at a new moment
            self.last_cwnd = 0
            self.instant_drop_nums = 0

        # if packet is dropped
        if event_type == EVENT_TYPE_DROP:
            # dropping more than one packet at a same time is considered one event of packet loss 
            if self.instant_drop_nums > 0:
                return
            self.instant_drop_nums += 1
            # step into fast recovery state
            self.curr_state = self.states[2]
            self.drop_nums += 1
            # clear acknowledgement count
            self.ack_nums = 0
            # Ref 1 : For ensuring the event type, drop or ack?
            self.cur_time = event_time
            if self.last_cwnd > 0 and self.last_cwnd != self.cwnd:
                # rollback to the old value of cwnd caused by acknowledgment first
                self.cwnd = self.last_cwnd
                self.last_cwnd = 0

        # if packet is acknowledged
        elif event_type == EVENT_TYPE_FINISHED:
            # Ref 1
            if event_time <= self.cur_time:
                return
            self.cur_time = event_time
            self.last_cwnd = self.cwnd

            # increase the number of acknowledgement packets
            self.ack_nums += 1
            # double cwnd in slow_start state
            if self.curr_state == self.states[0]:
                if self.ack_nums == self.cwnd:
                    self.cwnd *= 2
                    self.ack_nums = 0
                # step into congestion_avoidance state due to exceeding threshhold
                if self.cwnd >= self.ssthresh:
                    self.curr_state = self.states[1]

            # increase cwnd linearly in congestion_avoidance state
            elif self.curr_state == self.states[1]:
                if self.ack_nums == self.cwnd:
                    self.cwnd += 1
                    self.ack_nums = 0

        # reset threshhold and cwnd in fast_recovery state
        if self.curr_state == self.states[2]:
            self.ssthresh = max(self.cwnd // 2, 1)
            self.cwnd = self.ssthresh
            self.curr_state = self.states[1]

    def append_input(self, data):
        """
        The part of algorithm to make congestion control, which will be call when sender get an event about acknowledge or lost from reciever.
        See more at https://github.com/AItransCompetition/simple_emulator/tree/master#congestion_control_algorithmpy.
        """
        # add new data to history data
        self._input_list.append(data)

        # only handle acknowledge and lost packet
        if data["event_type"] != EVENT_TYPE_TEMP:
            # specify congestion control algorithm
            self.cc_trigger(data)
            # set cwnd or sending rate in sender
            return {
                "cwnd" : self.cwnd,
                "send_rate" : self.send_rate
            }
        return None


if __name__ == '__main__':
    # fixed random seed
    import random
    random.seed(1)
    
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
        # enable logging packet. You can train faster if USE_CWND=False
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

    # Output the picture of rate_changing.png
    # You can get more information from https://github.com/AItransCompetition/simple_emulator/tree/master#cwnd_changingpng
    plot_rate(log_packet_file, trace_file="traces/trace.txt", file_range="all", sender=[1])

    print("Qoe : %d" % (cal_qoe()) )
