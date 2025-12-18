import os
import yaml
import datetime
import logging
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT = sys.argv[1]          # input case

from HiveMinds.hive_mind import Hive_Mind
from HiveMinds.utils import setup_logging
from HiveMinds.utils import init_file
import dotenv
dotenv.load_dotenv('../.env')



def main(usr_file=INPUT, set_difficulty=None, model_info='deepseek-chat'):
    # usr_file = "../data/yaml/inputs/Allen-Cahn equation/eval1.yaml"
    with open(usr_file, 'r', encoding='utf-8') as file:
        print(f"reading file {usr_file}...")
        config = yaml.safe_load(file)
    
    # logging setup
    usr_file_name = usr_file.split('/')[-1].split('.')[0]
    now = datetime.datetime.now()
    
    results_dir = os.path.join('./logs', now.strftime("%Y-%m-%d_%H-%M-%S")+"."+usr_file_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    log_file = os.path.join('./logs', now.strftime("%Y-%m-%d_%H-%M-%S") + "." + usr_file_name + ".log.txt",)
    setup_logging(log_file)
        
    os.environ["OUTPUT_DIR"] = results_dir
    logging.info(f"Start processing...")
    logging.info(f"user input: \n{config['pde']}")
    init_file(config['pde'], file_path=usr_file)
    
    hive_minds = Hive_Mind(config['pde'], model_info=model_info)
    if set_difficulty == 'easy':
        hive_minds.pde_solver_easy()
    else:
        hive_minds.pde_solver()
    
    
if __name__ == '__main__':
    main()