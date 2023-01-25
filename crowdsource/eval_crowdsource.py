
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
import click

TRAINING_ITERATIONS = config['TRAINING_ITERATIONS']
TRAINING_HYPERPARAMS = config['TRAINING_HYPERPARAMS']
EVAL_METHOD = config['EVAL_METHOD']
TORCH_SEED = 8888
ROUND_DURATION = config['ROUND_DURATION'] 
torch.manual_seed(TORCH_SEED)

@click.command()
@click.option("--ipfs",default='/dns/127.0.0.1/tcp/5001/http',help="IPFS API Provider")
@click.option("--provider",default='http://127.0.0.1:7545',help="web3 API Http Provider")

def test_crowdsource(ipfs, provider):
  # Define Eval data
  # your testset Path
    testset_path = os.path.realpath(os.path.dirname(__file__))+'/data/user_data/evaluator_data.json'
    print(testset_path)
    with open(testset_path,'r') as f:
        eval_dataset = json.load(f)
        eval_data = eval_dataset['x']
        eval_targets = eval_dataset['y']

    # print(eval_data)
    my_eval_data = MyData(eval_data, eval_targets, True)

  # Evaluator client setting
    evaluator = CrowdsourceClient(
        "Evaluator", my_eval_data, eval_targets, Mymodel, F.cross_entropy, provider, 0 ,ipfs, deploy=True)

    evaluator.set_genesis_model(
        round_duration=ROUND_DURATION,
        max_num_updates=config['NUMBER_OF_TRAINERS']
    )
    # Evaluation
    threads = [
        threading.Thread(
                    target=evaluator.evaluate_until,
                    args=(TRAINING_ITERATIONS, EVAL_METHOD,"crowdsource"),
                    daemon=True
        )
    ]
    # Run all threads in parallel
    for t in threads:
        t.start()
    for t in threads:
        t.join()

test_crowdsource()