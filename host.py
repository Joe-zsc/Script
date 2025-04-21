import sys, os
import copy
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
sys.path.append(curr_path)  # add current terminal path to sys.path
import numpy as np
from util import *
from NLP_Module.Encoder import encoder
from defination import Host_info, Action_Class
from actions import *
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
 
def normalization2(data):
    _range = np.max(abs(data))
    return data / _range

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

class HOST:
    def __init__(self,
                 ip,
                 prior_node=None,
                 pivot=0,
                 env_data: dict = None,
                 env_file:Path=None):
        self.ip = ip
        self.state_vector = StateEncoder(ip=self.ip)
        self.info = Host_info(ip=self.ip)
        self.action = Action()
        self.info.prior_node = prior_node
        self.info.pivot = pivot
        self.action_history = self.action.history_set
        self.env_file=env_file
        self.env_data = env_data  #环境数据，用于执行动作丛中获得反馈
        self.info_no_reset= copy.deepcopy(self.info)
        assert env_data['ip']==ip,env_data

    def reset(self):
        self.info_no_reset= copy.deepcopy(self.info)
        self.action.reset()
        self.info = Host_info(ip=self.ip)
        return self.state_vector.reset()

    def perform_action(self, action_mask):
        if Action.test_action(action_mask):
            if type(action_mask) == str:
                for id in range(len(Action.legal_actions)):
                    if Action.legal_actions[id].name == action_mask:
                        action_mask = id
                        break
            next_o, r, done, result = self.step( action_mask)
            self.action_history = self.action.history_set
            # UTIL.write_csv(self.info)
            return next_o, r, done, result

    def step(self,  action_mask: int):
        # The action_idx here is the index of the action set
        done = 0
        reward = 0
        # action_exec_vector = self.state_vector.change_action_history_vector(action_mask)
        a_: Action_Class = self.action.legal_actions[action_mask]  # 真实的动作编号
        a=copy.deepcopy(a_)
        action_constraint = self.action.action_constraint(action=a,host_info=self.info)
        if action_constraint:
            cost = action_constraint.cost
            result = action_constraint.message
        else:
            self.action.history_set.add(a.id)
            cost = a.act_cost
            if a.name == self.action.PORT_SCAN.name:
                action = PortScan(target_info=self.info,
                                  env_data=self.env_data)
                action.act()
                self.info.port = action.port_list
                self.state_vector.port = action.port_list
                if action.port_list:
                    self.access = "reachable"
                    self.state_vector.update_vector(port=True, access=False)
                    reward = a.success_reward
                result = action.port_list

            elif a.name == self.action.OS_SCAN.name:
                action = OSScan(target_info=self.info, env_data=self.env_data)
                action.act()
                self.info.os = action.os
                self.state_vector.os = action.os
                if action.os:
                    self.state_vector.update_vector(os=True)
                    reward = a.success_reward
                result = action.os

            elif a.name == self.action.SERVICE_SCAN.name:
                action = ServicesScan(target_info=self.info,
                                      env_data=self.env_data)
                action.act()
                self.info.services = action.services_list
                self.state_vector.services = action.services_list
                if action.services_list:
                    self.state_vector.update_vector(service=True)
                    reward = a.success_reward
                result = action.services_list

            elif a.name == self.action.PORT_SERVICE_SCAN.name:

                action = ServicesScan(target_info=self.info,
                                      port_scan=True,
                                      env_data=self.env_data)
                action.act()
                self.info.port = action.port
                self.state_vector.port = action.port
                if action.port:
                    self.access = "reachable"
                    self.state_vector.update_vector(port=True, access=True)
                self.info.services = action.services_list
                self.services = action.services_list
                if action.services_list:
                    self.state_vector.update_vector(service=True)
                result = action.services_list

            elif a.name == self.action.WEB_SCAN.name:

                action = WebScan(target_info=self.info, env_data=self.env_data)
                action.act()
                self.info.web_fingerprint = action.fliter_info
                self.state_vector.web_fingerprint = action.fliter_info
                result = action.fliter_info
                if result:
                    self.action.webscan_counts += 1
                    self.state_vector.update_vector(web_fingerprint=True)
                    reward = a.success_reward if self.action.webscan_counts == 1 else 0  #*math.exp(-Action.webscan_counts)

            elif a.type == self.action.All_EXP[0].type:
                action = Exploit(target_info=self.info,
                                 exp=a,
                                 env_data=self.env_data)
                result, target_info = action.act()
                self.action.exp_counts += 1
                if result:
                    self.info = target_info
                    self.state_vector.access = "compromised"
                    self.state_vector.update_vector(access=True)
                    reward = a.success_reward  #*math.exp(-Action.exp_counts)
                # else:
                #     cost += self.action.action_failed['cost']
        reward = int(reward - cost)
        # cost = cost * action_exec_vector[action_idx]
        done = self.state_vector.goal_reached()
        next_state = self.state_vector.host_vector
        if isinstance(result, list):
            result = ','.join(result)
        # reward = reward - cost
        
        return next_state, reward, done, result

class StateEncoder:
    '''
    先初始化动作空间，才初始化状态空间
    '''
    '''
    state space = vector of [os] + vector of [port] + vector of [services]

    '''

    # 状态空间划分
    state_vector_key = [
        "access", "os", "port", "service", "web_fingerprint", "action_history"
    ]
    state_vector = dict.fromkeys(state_vector_key, 0)

    access_dim = 2
    state_vector["access"] = access_dim
    os_dim = encoder.SBERT_model_dim
    state_vector["os"] = os_dim
    port_dim = encoder.SBERT_model_dim
    state_vector["port"] = port_dim
    service_dim = encoder.SBERT_model_dim
    state_vector["service"] = service_dim
    web_fingerprint_dim = encoder.SBERT_model_dim
    state_vector["web_fingerprint"] = web_fingerprint_dim

    action_history_window = 5
    # if Action_embedding.use_action_embedding:
    #     action_dim = Action.action_embedding.vector_space.shape[1]
    # else:
    #     action_dim=1
    # action_history_dim =action_history_window*action_dim
    action_history_dim = 0  #弃用
    # action_history_dim = 4
    state_vector["action_history"] = action_history_dim  #弃用

    access = 2
    OS_vector_idx = access_dim
    port_vector_idx = access_dim + os_dim
    services_vector_idx = access_dim + os_dim + port_dim
    web_fingerprint_idx = access_dim + os_dim + port_dim + service_dim
    action_history_idx = access_dim + os_dim + port_dim + service_dim + web_fingerprint_dim
    
    state_space = access_dim + os_dim + port_dim + \
        service_dim+ web_fingerprint_dim + action_history_dim

    def __init__(self, ip):
        self.ip = ip
        '''
        state related info
        '''
        self.os:str = None  # string
        # string:unknown,reachable,compromised, 00:unknow , 11:compromised , 01: uncompromised
        self.access:int = None
        self.port:list = None  # list of string
        self.services:list = None  # list of string
        self.web_fingerprint:list = None  # str
        self.host_vector = self.initialize()
        '''
        reforcement learning related info
        '''
        self.done:int = 0
        self.reward:int = 0
        self.steps:int = 0
        '''
        host info
        '''
        self.port_vector = self.host_vector[self.port_vector_idx:self.
                                            services_vector_idx]
        self.serv_vector = self.host_vector[self.services_vector_idx:self.
                                            web_fingerprint_idx]
        self.os_vector = self.host_vector[self.OS_vector_idx:self.
                                          port_vector_idx]
        self.web_vector = self.host_vector[self.web_fingerprint_idx:self.
                                           action_history_idx]
        self.act_vector = self.host_vector[self.action_history_idx:]

    def observ(self):
        return self.host_vector

    def reset(self):

        self.done = 0
        self.reward = 0
        self.access = None
        self.port = None
        self.services = None
        self.os = None
        self.web_fingerprint = None
        self.steps = 0
        # self.host_info = dict.fromkeys(self.info, None)
        self.host_vector = self.initialize()

        return self.host_vector

    def goal_reached(self):
        self.done = 0
        if self.access == "compromised":
            self.done = 1
        return self.done

    def change_os_vector(self):
        os_vector = np.zeros(encoder.SBERT_model_dim, dtype=np.float32)
        all_possible_os = []
        if self.os.find("or") != -1:
            all_possible_os = self.os.split("or")
        else:
            all_possible_os.append(self.os)
        for i in range(len(all_possible_os)):
            os = all_possible_os[i]
            vec = encoder.encode_SBERT(os).flatten()
            os_vector += vec
        vector = os_vector / len(all_possible_os)
        # vector=standardization(vector)
        # vector=normalization(vector)
        # vector=standardization(vector)
        # vector=normalization(vector)
        return vector

    def change_port_vector(self):
        vector = np.zeros(encoder.SBERT_model_dim, dtype=np.float32)
        all_ports=','.join(self.port)
        vector=encoder.encode_SBERT(all_ports).flatten()
        # vector=standardization(vector)
        # vector=normalization(vector)
        # vector=standardization(vector)
        # vector=normalization(vector)
        return vector
        # self.port_index = []
        # for p in self.port:
        #     if p in self.support_port:
        #         idx = self.support_port.index(p)
        #         self.port_index.append(idx)
        #         vector += self.port_vector_eye[idx]
        # return vector.reshape(-1, 1).squeeze()

    def change_services_vector(self):
        assert len(self.port) > 0
        assert len(self.services) == len(self.port)
        vector = np.zeros(encoder.SBERT_model_dim, dtype=np.float32)
        all_services=','.join(self.services)
        vector=encoder.encode_SBERT(all_services).flatten()
        # vector=standardization(vector)
        # vector=normalization(vector)
        # vector=standardization(vector)
        # vector=normalization(vector)
        return vector
    

    def change_access_vector(self):
        vector = np.zeros(2, dtype=np.float32)
        if self.access == 'reachable':
            vector[1] = 1
        elif self.access == 'compromised':
            vector[0] = 1
        return vector

    def change_web_fingerprint_vector(self):
        wp_vector = np.zeros(encoder.SBERT_model_dim, dtype=np.float32)
        for wp in self.web_fingerprint:
            # vector = get_vector(
            #     wp, dim=encoder.SBERT_model_dim).detach().numpy().flatten()
            vector = encoder.encode_SBERT(wp).flatten()
            wp_vector += vector
        vector = wp_vector / len(self.web_fingerprint)
        # vector=standardization(wp_vector)
        # vector=normalization(wp_vector)
        return vector

    def update_vector(self,
                      access=False,
                      os=False,
                      port=False,
                      service=False,
                      web_fingerprint=False):
        if access:
            vector = self.change_access_vector()
            self.host_vector[:self.OS_vector_idx] = vector
        if os:
            vector = self.change_os_vector()
            self.host_vector[self.OS_vector_idx:self.port_vector_idx] = vector
        if port:
            vector = self.change_port_vector()
            self.host_vector[self.port_vector_idx:self.
                             services_vector_idx] = vector
        if service:
            vector = self.change_services_vector()
            self.host_vector[self.services_vector_idx:self.
                             web_fingerprint_idx] = vector

        if web_fingerprint:
            vector = self.change_web_fingerprint_vector()
            self.host_vector[self.web_fingerprint_idx:self.
                             action_history_idx] = vector
        # self.host_vector=standardization(self.host_vector)
        # self.host_vector=standardization(self.host_vector)
        return self.host_vector

    def initialize(self):
        vector = np.zeros(self.state_space, dtype=np.float32)
        return vector