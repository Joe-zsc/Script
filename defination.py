from prettytable import *
from prettytable.colortable import ColorTable, Themes
from pprint import pprint, pformat
class Host_info:
    def __init__(self, ip):
        self.ip: str = ip
        self.os: str = ''
        self.port: list = []
        self.web_fingerprint: str = ''
        self.services: list = []
        self.session_id: int = -1
        self.session_info: dict = dict()
        self.vul: list = []
        self.flag: list = []
        self.flag_path: list = []
        self.neighbor_subnet: list = []
        self.neighbor_host: list = []
        self.intranet_ip: list = []
        self.prior_node = None
        self.pivot: int = 0
    def show(self):
        info=self.__dict__.copy()
        x = ColorTable(theme=Themes.OCEAN)
        x.field_names = ["Name", "Information"]
        for key,value in info.items():
            
            if value:
                if key=="prior_node":
                    value=value.ip
                x.add_row([key,pformat(value)])
        pprint(x)


class Action_Class:
    use_action_prob = False

    def __init__(
        self,
        id: int,
        name: str,
        act_cost: int,
        success_reward: int = 0,
        type: str = None,
        vulnerability=[],
        exp_info: list[dict] = [],
        setting: dict = None,
    ):
        self.id = id
        self.name = name

        self.type = type
        self.act_cost = act_cost
        self.success_reward = success_reward

        self.vulnerability = vulnerability
        self.exp_info = exp_info
        self.setting = setting
        self.set_success_prob()

    def set_success_prob(self):
        '''
        Manual
        Great
        Excellent
        Low
        Normal
        Good
        Average
        '''
        if (not self.use_action_prob) or (not self.exp_info):
            self.prob = 1
        else:
            rank = self.exp_info["rank"]
            if rank == "Excellent":
                self.prob = 1
            elif rank == "Great":
                self.prob = 0.9
            elif rank == "Good":
                self.prob = 0.7
            elif rank == "Normal":
                self.prob = 0.5
            elif rank == "Average":
                self.prob = 0.5
            elif rank == "Low":
                self.prob = 0.4
            elif rank == "Manual":
                self.prob = 0.5
            else:
                raise ValueError('unknown rank')


class Action_Result():
    def __init__(self, type, message='', cost=0):
        self.type = type
        self.message = type
        self.cost = cost

