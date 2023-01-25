import threading

import torch
import torch.nn.functional as F
from clients import CrowdsourceClient
from utils_2cp import print_token_count

from test_utils.my import MyData, Mymodel
from test_utils.functions import same_weights
from crowdsource_conf import config
import os
import json

TRAINING_ITERATIONS = config['TRAINING_ITERATIONS']
TRAINING_HYPERPARAMS = config['TRAINING_HYPERPARAMS']
EVAL_METHOD = config['EVAL_METHOD']
TORCH_SEED = 8888
ROUND_DURATION = config['ROUND_DURATION'] 
torch.manual_seed(TORCH_SEED)

@click.command()
@click.option("--ipfs",default='/dns/127.0.0.1/tcp/5001/http',help="IPFS API Provider")
@click.option("--provider",default='http://127.0.0.1:7545',help="web3 API Http Provider")

def test_crowdsource():
    trainers = []
    train_clients = []

    for number in range(config['NUMBER_OF_TRAINERS']):
        trainer_index = "trainer"+str(number+1)
        trainers.append(trainer_index)

    for trainer in trainers :
        print("client is deploying....")
        trainer_index = trainer[-1:]
        # your trainset Path
        trainset_path = os.path.realpath(os.path.dirname(__file__))+'/data/user_data/'+trainer+'_data.json'
        with open(trainset_path,'r') as f:
            train_dataset = json.load(f)
            train_data = train_dataset['x']
            train_targets = train_dataset['y']
        my_train_data = MyData(train_data, train_targets, True,device_num=str(trainer_index))
        train_client = CrowdsourceClient(trainer,my_train_data,train_targets,Mymodel, F.cross_entropy,int(trainer_index),contract_address=evaluator.contract_address, device_num= str(trainer_index))
        train_clients.append(train_client)

    evaluator.set_genesis_model(
        round_duration=ROUND_DURATION,
        max_num_updates=config['NUMBER_OF_TRAINERS']
    )

    # Training
    threads = [
        threading.Thread(
            target=trainer.train_until,
            kwargs=TRAINING_HYPERPARAMS,
            daemon=True
        ) for trainer in train_clients
    ]

    # Run all threads in parallel
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for trainer in train_clients:
        print_token_count(trainer)

test_crowdsource()