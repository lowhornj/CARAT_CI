import torch
import numpy as np
import os
import random
import logging
import sys
sys.path.append("..")
import time
from functools import wraps
from pathlib import Path
from datetime import date, timedelta, datetime
import numpy as np
import random
import torch
import os
import statsmodels.api as sm
import pandas as pd

curr_date = date.today()
curr_date = curr_date.strftime("%Y_%b_%d")

logger = logging.getLogger(__name__)
# Misc logger setup so a debug log statement gets printed on stdout.
logger.setLevel(logging.DEBUG)
#fh = logging.FileHandler('/opt/production_apps/couchbase_poc/logs/log_'+ curr_date +'.txt')
fh = logging.FileHandler('logs/log_'+ curr_date +'.txt')
fh.setLevel(logging.DEBUG)
#console aka streamhandler
sh = logging.StreamHandler()
#create a format
log_format = "%(asctime)s %(levelname)s -- %(message)s"
formatter = logging.Formatter(log_format)
#set the format for each handler
fh.setFormatter(formatter)
sh.setFormatter(formatter)
#create the logs via streaming and file handling
logger.addHandler(sh)
logger.addHandler(fh)

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

def set_generator_seed(seed: int = 42):
    gen_cpu = torch.Generator()
    gen_cpu.manual_seed(seed)
    return gen_cpu

def parse_ne(string):
    ne_list = string.split('_')
    ne_list = [x for x in ne_list if str(x).isdigit()]
    ne_list = [x for x in ne_list if len(x)>4]
    ne_list = sorted(ne_list, key=len)
    ne_id = "::".join(ne_list)
    return ne_id

def timed(func):
    """This decorator prints the execution time for the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug("{} ran in {}s".format(func.__name__, round(end - start, 2)))
        return result

    return wrapper

class get_dates:
    def __init__(self,backfill_start=None,backfill_end=None):
        self.backfill_start = backfill_start
        self.backfill_end = backfill_end
        self.today = date.today()
        self.process_day = None
        self.raw_day = None
        self.start_time = None
        self.end_time = None
        self.alarm_config_days = None
        
    def get_days(self):
        if self.backfill_start is None and self.backfill_end is None:
            self.process_day = self.today - timedelta(days=2)
            self.raw_day = self.today - timedelta(days=1)
            self.start_time = datetime.combine(self.process_day, datetime.min.time())
            self.end_time = datetime.combine(self.process_day, datetime.max.time())
            self.end_time = datetime(self.end_time.year, 
                                 self.end_time.month,
                                 self.end_time.day,
                                 self.end_time.hour,
                                 self.end_time.minute)
            self.alarm_config_days = [self.raw_day.strftime("%Y%m%d")]

        else:
            self.start_time = datetime.combine(self.backfill_start, datetime.min.time())
            self.end_time = datetime.combine(self.backfill_end, datetime.min.time())
            self.end_time = datetime(self.end_time.year, 
                     self.end_time.month,
                     self.end_time.day,
                     self.end_time.hour,
                     self.end_time.minute)
            self.backfill_days = [(self.start_time+timedelta(days=x)) for x in range((self.end_time-self.start_time).days)]
            self.alarm_config_days = [x.strftime("%Y%m%d") for x in self.backfill_days]
            
            
def get_matches(cell, site_dict,affected_cols,degradations):
    kpi_match = []
    for kpi in site_dict['PM']:
        kpi_match.append((kpi,kpi in affected_cols))

    alarms = list(filter(lambda v: match('alarm', v), affected_cols))
    alarms_specific=list(filter(lambda v: match('group', v), affected_cols))
    if ((len(alarms)>0) or (len(alarms_specific) > 0)):
        alarms = True
    else:
        alarms = False
    #if alarms == False:
    alarms = alarms == site_dict['FM']
    
    cfgs = list(filter(lambda v: match('USER', v), affected_cols))
    cfgs_managed = list(filter(lambda v: match('managed-element', v), affected_cols))
    cfgs_default = list(filter(lambda v: match('default', v), affected_cols))
    if ((len(cfgs)>0) or (len(cfgs_managed)>0) or (len(cfgs_default) > 0)):
        cfgs = True
    else:
        cfgs = False
    #if cfgs == False:
    cfgs = cfgs == site_dict['CM']
       
    kpi_mtches = [item[1] for item in kpi_match]    
    

    if degradations.shape[0] > 0:
        event_detected = True
    else:
        event_detected = False
        
    cm = [alarms,cfgs,event_detected]
    cm.extend(kpi_mtches)
    
    return {cell:{"kpi_match":kpi_match,"alarm_match": alarms, 'cfg_match': cfgs ,'event_detected':event_detected,"cm":cm}}

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    #torch.set_default_dtype(torch.float)
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)
