import configparser
from loguru import logger as logging
import time
import json
import yaml
from pprint import pprint, pformat
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import namedtuple, defaultdict
from easydict import EasyDict
from colorama import init, Fore, Back, Style
from rich.console import Console
import torch
import numpy as np
import random
console = Console()
Attack_Path_Transition = namedtuple(
    'Attack_Path_Transition', ('ip', 'step', 'action', 'result', 'reward'))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
class Configure():
    conf = configparser.ConfigParser()
    try:
        conf.read(Path(__file__).parent / "config.ini")
    except Exception as e:
        logging.error("config file not found" + e)

    @classmethod
    def get(cls, label, name):
        return cls.conf.get(label, name)

    @classmethod
    def getBool(cls, label, name):
        bl = cls.conf.get(label, name)
        if bl.lower() == "true" or bl == '1':
            return True
        elif bl.lower() == "false" or bl == '0':
            return False
        else:
            raise ValueError('Bool value must be true/1/True/false/False/0')

    @classmethod
    def set(cls, label, name, value):
        cls.conf.set(label, name, str(value))
        cls.conf.write(open("config.ini", "w"))


class Matrix:
    Train_Matrix = EasyDict({"all": [], "best": {}})
    '''
    Training signals
    '''
    EarlyTerminate = "early_terminate"
    Finished = "finished"
    Failed = "failed"

    def __init__(self) -> None:
        pass


class UTIL:
    '''
    Running Mode:
    '''

    today = datetime.now().strftime('%b%d')
    Running_title = ''
    project_name = Configure.get('common', 'project_name')

    project_path = Path(__file__).parent

    trained_agent_path = project_path / Configure.get('common',
                                                      'trained_agent_path')
    log_path = project_path / Configure.get('common', 'log_path')

    def __init__(self) -> None:
        pass

    @classmethod
    def show_banner(cls):

        banner = u"""
 
     ___      .______   .______       __   __      
    /   \     |   _  \  |   _  \     |  | |  |     
   /  ^  \    |  |_)  | |  |_)  |    |  | |  |     
  /  /_\  \   |   ___/  |      /     |  | |  |     
 /  _____  \  |  |      |  |\  \----.|  | |  `----.
/__/     \__\ | _|      | _| `._____||__| |_______|
                                                   

"""

        print(banner)
        cls.show_credit()
        time.sleep(2)

    # flag_log
    @classmethod
    def show_credit(cls):
        credit = u"""
+ -- --=[ APRIL\t: Autonomous Penetesting based on ReInforcement Learning             ]=-- -- +
+ -- --=[ Author\t: NUDT-HFBOT Team                                   ]=-- -- +
+ -- --=[ Website\t: https://gitee.com/JoeSC/April  ]=-- -- +
    """
        print(credit)

    @classmethod
    def line_break(cls, length=60, symbol='-'):
        line_break = symbol * length
        logging.info(line_break)

    def write_csv_DictList(file: Path, data: list):
        variables = list(data[0].keys())
        pd_data = pd.DataFrame([[i[j] for j in variables] for i in data],
                               columns=variables)
        pd_data.to_csv(file, mode='w', index=False)
        return pd_data

    @classmethod
    def smooth_data(cls, data: list, weight: float = 0.9):
        smoothed_data = []

        last = data[0]
        smoothed = []
        for point in data:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed_data.append(smoothed_val)
            last = smoothed_val
        return smoothed_data

    @classmethod
    def write_to_csv(cls, data: list, save_path: str):
        '''
        data: [dict1,dict2,...]
        '''
        import csv
        assert len(data) > 0, "the input data is empty"

        f = open(save_path, 'a', encoding='utf8', newline='')
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        for line in data:
            writer.writerow(line)
    @classmethod
    def save_json(cls, path, data):
        with open(path, "w", encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=4))

    @classmethod
    def read_yaml(cls,path):
        with open(path, 'r', encoding='utf-8') as f:
            result = yaml.load(f.read(), Loader=yaml.FullLoader)
        return result
    
    
def split_num_l(num_lst):
    """merge successive num, sort lst(ascending or descending): 'as' or 'des'
    eg: [1, 3,4,5,6, 9,10] -> [[1], [3, 4, 5, 6], [9, 10]]
    """
    num_lst_tmp = [int(n) for n in num_lst]
    sort_lst = sorted(num_lst_tmp)  # ascending
    len_lst = len(sort_lst)
    i = 0
    split_lst = []

    tmp_lst = [sort_lst[i]]
    while True:
        if i + 1 == len_lst:
            break
        next_n = sort_lst[i + 1]
        if sort_lst[i] + 1 == next_n:
            tmp_lst.append(next_n)
        else:
            split_lst.append(tmp_lst)
            tmp_lst = [next_n]
        i += 1
    split_lst.append(tmp_lst)
    return split_lst


def Merge_str_lst(num_lst):
    """[[1], [3, 4, 5, 6], [9, 10]] -> ['1', '3~6', '9~10']
    """
    if not num_lst:
        return []
    mylst = split_num_l(num_lst)
    mg_l = []
    for num_l in mylst:
        if len(num_l) == 1:
            mg_l.append(str(num_l[0]))
        else:
            mg_l.append(str(num_l[0]) + '~' + str(num_l[-1]))
    return mg_l


class color:

    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    GREY = "\033[1;30m"
    END = '\033[0m'
    init(autoreset=True)
    @classmethod
    def print(cls, s, c=GREEN, end='\n'):
        print(c + s + cls.END, end=end)

    @classmethod
    def color_str(cls, s, c=GREEN):
        s = pformat(s)
        return c + s + cls.END
    #  前景色:红色  背景色:默认
    @classmethod
    def red(cls, s):
        return Fore.RED + s + Fore.RESET

    #  前景色:绿色  背景色:默认
    @classmethod
    def green(cls, s):
        return Fore.GREEN + s + Fore.RESET

    #  前景色:黄色  背景色:默认
    @classmethod
    def yellow(cls, s):
        return Fore.YELLOW + s + Fore.RESET

    #  前景色:蓝色  背景色:默认
    @classmethod
    def blue(cls, s):
        return Fore.BLUE + s + Fore.RESET

    #  前景色:洋红色  背景色:默认
    @classmethod
    def magenta(cls, s):
        
        return Fore.MAGENTA + s + Fore.RESET

    #  前景色:青色  背景色:默认
    @classmethod
    def cyan(cls, s):
        return Fore.CYAN + s + Fore.RESET

    #  前景色:白色  背景色:默认
    @classmethod
    def white(cls, s):
        return Fore.WHITE + s + Fore.RESET

    #  前景色:黑色  背景色:默认
    @classmethod
    def black(cls, s):
        return Fore.BLACK

    #  前景色:白色  背景色:绿色
    @classmethod
    def white_green(cls, s):
        return Fore.WHITE + Back.GREEN + s
    @classmethod
    def dave(cls, s):
        return Style.BRIGHT + Fore.GREEN + s

# a=color.red(s="test")
# a=color.magenta(s="test")
# a=color.white_green(s="test")
# color = Colored()
# print color.red('I am red!')
# print color.green('I am gree!')
# print color.yellow('I am yellow!')
# print color.blue('I am blue!')
# print color.magenta('I am magenta!')
# print color.cyan('I am cyan!')
# print color.white('I am white!')
# print color.white_green('I am white green!')
def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log

    CRITICAL = 50
    FATAL = CRITICAL
    ERROR = 40
    WARNING = 30
    WARN = WARNING
    INFO = 20
    DEBUG = 10
    NOTSET = 0

    logging.critical()
    logging.fatal()
    logging.error()
    logging.warning()
    logging.warn()
    logging.info()
    logging.debug()

    """
    logger = logging.getLogger()
    # root_logger = logging.getLogger()
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
