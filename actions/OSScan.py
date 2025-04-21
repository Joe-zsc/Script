
from defination import Host_info, Action_Class

class OSScan():
    def __init__(self,
                 target_info: Host_info,env_data:dict=None):
        self.target_ip = target_info.ip
        self.target_info = target_info
        self.port = self.target_info.port
        self.os = ''
        self.env_data=env_data
    def act(self):
        os = self.simulate_act()
        self.os = os
        self.target_info.os=os
        result = True if os else False
        return result,self.target_info
    def simulate_act(self):

        if self.env_data["ip"] == self.target_ip:
            return self.env_data["os"]
        return

    
    