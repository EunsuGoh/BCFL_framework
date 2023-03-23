import threading

import torch
import torch.nn.functional as F
from clients import ConsortiumSetupClient, ConsortiumClient
from utils_2cp import print_token_count

from test_utils.my import MyData, Mymodel
from test_utils.functions import same_weights
from consortium_conf import config
import os
import json
import re
import random
import numpy as np
import torch.backends.cudnn as cudnn
from dotenv import load_dotenv
import subprocess

from web3 import HTTPProvider, Web3 

load_dotenv()

TRAINING_ITERATIONS = config['TRAINING_ITERATIONS']
TRAINING_HYPERPARAMS = config['TRAINING_HYPERPARAMS']
BATCH_SIZE = TRAINING_HYPERPARAMS['batch_size']
EVAL_METHOD = config['EVAL_METHOD']
TORCH_SEED = 8888
ROUND_DURATION = config['ROUND_DURATION']  # expecting rounds to always end early
SELECTION_METHOD = config['SELECTION_METHOD']
NODE_PATH = os.environ["NODE_PATH"]

torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
np.random.seed(TORCH_SEED)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(TORCH_SEED)
# torch.use_deterministic_algorithms(True)
os.environ["PYTHONHASHSEED"] = str(TORCH_SEED)
account_list_path = os.path.dirname(os.path.abspath(__file__))
# torch.manual_seed(TORCH_SEED)

def test_consortium():
    # Genesis Setting
    trainers = []
    genesis = ConsortiumSetupClient("Genesis", Mymodel, 0, deploy=True)
    for number in range(config['NUMBER_OF_TRAINERS']):
        trainer_index = "trainer"+str(number+1)
        trainers.append(trainer_index)

    # Train Clients Setting
    train_clients = []
    for trainer in trainers :
        print("client is deploying....")
        trainer_index = trainer[-1:]
        # your trainset path
        trainset_path = os.path.realpath(os.path.dirname(__file__))+'/data/user_data/'+trainer+'_data.json'
        with open(trainset_path,'r') as f:
            train_dataset = json.load(f)
            train_data = train_dataset['x']
            train_targets = train_dataset['y']
        my_train_data = MyData(train_data, train_targets, True, device_num=str(trainer_index))
        train_client = ConsortiumClient(trainer,my_train_data,train_targets,Mymodel, F.cross_entropy,int(trainer_index),batch_size = BATCH_SIZE, contract_address=genesis.contract_address, device_num=str(trainer_index))
        train_clients.append(train_client)
    

    genesis.set_genesis_model(
        round_duration=ROUND_DURATION,
        max_num_updates=config['NUMBER_OF_TRAINERS'],
        scenario = "consortium"
    )

    genesis.add_auxiliaries([
        trainer.address for trainer in train_clients
    ])

# Add Consortium scenario's set Current Trainer Here
    # Training
    threads = [
        threading.Thread(
            target=trainer.train_until,
            kwargs=TRAINING_HYPERPARAMS,
            daemon=True
        ) for trainer in train_clients
    ]

    # Evaluation
    threads.extend([
        threading.Thread(
            target=trainer.evaluate_until,
            args=(TRAINING_ITERATIONS, EVAL_METHOD,"consortium"),
             kwargs={
                'selection_method':SELECTION_METHOD
            },
            daemon=True
        ) for trainer in train_clients
    ])

    # Run all threads in parallel
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
        
    for trainer in train_clients:
        print_token_count(trainer)

test_consortium()