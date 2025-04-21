
from util import Configure
import re
import sys
import os
import time
from defination import Host_info, Action_Class
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
sys.path.append(curr_path)  # add current terminal path to sys.path


class PortScan():



    def __init__(self, target_info: Host_info, env_data:dict=None):
        self.target_ip = target_info.ip
        self.target_info = target_info
        self.port_list = []
        self.env_data=env_data

    def act(self):
        
        port_list = self.simulate_act()
        
        self.target_info.port = port_list
        self.port_list = port_list
        result = True if port_list else False
        return result, self.target_info

    def simulate_act(self):

        if self.env_data["ip"] == self.target_ip:
            return self.env_data["port"]
        return []

    