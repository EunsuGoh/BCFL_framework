import threading

import torch
import torch.nn.functional as F
from clients import ConsortiumSetupClient, ConsortiumClient
from utils_2cp import print_token_count

from test_utils.xor import XORDataset, XORModel
from test_utils.functions import same_weights
from consortium_conf import config
import sys

TRAINING_ITERATIONS = config['TRAINING_ITERATIONS']
TRAINING_HYPERPARAMS = config['TRAINING_HYPERPARAMS']
EVAL_METHOD = config['EVAL_METHOD']
TORCH_SEED = 8888
ROUND_DURATION = config['ROUND_DURATION']  # expecting rounds to always end early

torch.manual_seed(TORCH_SEED)

bob_data, bob_targets, charlie_data, charlie_targets = XORDataset(
    64).split()
david_data, david_targets, eve_data, eve_targets = XORDataset(4).split()


def test_consortium():
    """
    Integration test for consortium setting.
    Alice sets up the main contract but doesn't participate.
    """

    trainers = []
    genesis = ConsortiumSetupClient("Genesis", XORModel, 0, deploy=True)
    for number in range(config['NUMBER_OF_TRAINERS']):
        trainer_index = "trainer"+str(number+1)
        trainers.append(trainer_index)

    train_clients = []
    for trainer in trainers :
        trainer_index = trainer[-1:]
        train_client = ConsortiumClient(trainer,bob_data,bob_targets,XORModel, F.mse_loss,int(trainer_index))
        train_clients.append(train_client)
    

    genesis.set_genesis_model(
        round_duration=ROUND_DURATION,
        max_num_updates=config['NUMBER_OF_TRAINERS']
    )

    genesis.add_auxiliaries([
        trainer.address for trainer in train_clients
    ])

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
            args=(TRAINING_ITERATIONS, EVAL_METHOD),
            daemon=True
        ) for trainer in train_clients
    ])

    # Run all threads in parallel
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # print_token_count(bob)
    # print_token_count(charlie)
    # print_token_count(david)

    # assert bob.get_token_count() > 0, "Bob ended up with 0 tokens"
    # assert charlie.get_token_count() > 0, "Charlie ended up with 0 tokens"

    # bob_global_model = bob.get_current_global_model()
    # charlie_global_model = charlie.get_current_global_model()

    # assert same_weights(
    #     bob_global_model,
    #     charlie_global_model
    # ), "Bob and Charlie ran the same aggregation but got different model weights"

    # assert str(bob_global_model.state_dict()) == \
    #     str(charlie_global_model.state_dict()), \
    #     "Bob and Charlie ran the same aggregation but got different model dicts"

test_consortium()