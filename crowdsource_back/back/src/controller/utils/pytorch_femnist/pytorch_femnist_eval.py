import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import normalize
import torch.optim as optim
from data_load import MyMnist
import time
import methodtools
import torch
import pyvacy.optim
from pyvacy.analysis import moments_accountant as epsilon
from ipfs_client import IPFSClient
from contract_clients import CrowdsourceContractClient
import contribution
import wandb
from utils import print_token_count
import json
from torchvision.transforms import ToTensor
from dotenv import load_dotenv
import os
import sys

load_dotenv()

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
wandb.init(project="2cp",entity="daeyeolkim")

class _BaseClient:
    """
    Abstract base client containing common features of the clients for both the
    Crowdsource Protocol and the Consortium Protocol.
    """

    def __init__(self, name, model_constructor, contract_constructor, account_idx, contract_address=None, deploy=False):
        self.name = name
        self._model_constructor = model_constructor
        if deploy:
            self._print("Deploying contract...")
        self._contract = contract_constructor(
            account_idx, contract_address, deploy)
        self._account_idx = account_idx
        self.address = self._contract.address
        self.contract_address = self._contract.contract_address
        self._print(
            f"Connected to contract at address {self.contract_address}")

    def get_token_count(self, address=None, training_round=None):
        return self._contract.countTokens(address, training_round)

    def get_total_token_count(self, training_round=None):
        return self._contract.countTotalTokens(training_round)

    def get_gas_used(self):
        return self._contract.get_gas_used()

    def wait_for_txs(self, txs):
        receipts = []
        if txs:
            self._print(f"Waiting for {len(txs)} transactions...")
            for tx in txs:
                receipts.append(self._contract.wait_for_tx(tx))
            self._print(f"{len(txs)} transactions mined")
        return receipts

    def _print(self, msg):
        print(f"{self.name}: {msg}")


class _GenesisClient(_BaseClient):
    """
    Extends upon base client with the ability to set the genesis model to start training.
    """

    def __init__(self, name, model_constructor, contract_constructor, account_idx, ipfs_api ,contract_address=None, deploy=False):
        super().__init__(name, model_constructor,
                         contract_constructor, account_idx, contract_address, deploy)
        self._ipfs_client = IPFSClient(model_constructor, ipfs_api)

    def set_genesis_model(self, round_duration, max_num_updates=0):
        """
        Create, upload and record the genesis model.
        """
        self._print("Setting genesis...")
        genesis_model = self._model_constructor()
        # genesis_model.to(device)
        genesis_cid = self._upload_model(genesis_model)
        self._print(f"Genesis model cid : {genesis_cid}")
        tx = self._contract.setGenesis(
            genesis_cid, round_duration, max_num_updates)
        self.wait_for_txs([tx])

    def _upload_model(self, model):
        """Uploads the given model to IPFS."""
        uploaded_cid = self._ipfs_client.add_model(model)
        return uploaded_cid


class CrowdsourceClient(_GenesisClient):
    """
    Full client for the Crowdsource Protocol.
    """

    

    TOKENS_PER_UNIT_LOSS = 1e18  # same as the number of wei per ether
    CURRENT_ROUND_POLL_INTERVAL = 1.  # Ganache can't mine quicker than once per second

    def __init__(self, name, data, targets, model_constructor, model_criterion, account_idx, contract_address=None, deploy=False):
        super().__init__(name,
                         model_constructor,
                         CrowdsourceContractClient,
                         account_idx,
                         contract_address,
                         deploy)
        self.data_length = data.__len__()
        

        # self._worker = sy.VirtualWorker(_hook, id=name)
        self._criterion = model_criterion
        # TODO: should actually send these to the syft worker.
        # Temporarily stopped doing this as a workaround for subtle multithreading bugs
        # in order to run experiments on contributivity
        # data.to(device)
        # targets.to(device)
        self._data = data  # .send(self._worker)
        self._targets = targets  # .send(self._worker)
        # self._data.to(device)
        # self._targets.to(device)
        ###테스트 로더 수정
        # self._test_loader = torch.utils.data.DataLoader(
        #     sy.BaseDataset(self._data, self._targets),
        #     batch_size=len(data)
        # )
        self._test_loader = torch.utils.data.DataLoader(self._data, batch_size=len(data),
                                         shuffle=False, num_workers=0)
        ###
        # train loader is defined each time training is run
        self._gas_history = {}

    def train_one_round(self, batch_size, epochs, learning_rate, dp_params=None):
        cur_round = self._contract.currentRound()
        print("cur_round : ",cur_round)
        tx,model = self._train_single_round(
            cur_round,
            batch_size,
            epochs,
            learning_rate,
            dp_params
        )
        self.wait_for_txs([tx])
        self._gas_history[cur_round] = self.get_gas_used()
        self._print(f"Done training. Gas used: {self.get_gas_used()}")

    def train_until(self, final_round_num, batch_size, epochs, learning_rate, dp_params=None):
        start_round = self._contract.currentRound()
        # print(start_round)
        for r in range(start_round, final_round_num+1):
            self.wait_for_round(r)
            tx,model = self._train_single_round(
                r,
                batch_size,
                epochs,
                learning_rate,
                dp_params
            )
            self.wait_for_txs([tx])
            self._gas_history[r] = self.get_gas_used()
        self._print(f"Done training. Gas used: {self.get_gas_used()}")

    def evaluate_until(self, final_round_num, method):
        self._gas_history[1] = self.get_gas_used()
        for r in range(1, final_round_num+1):
            self.wait_for_round(r + 1)
            scores = self._evaluate_single_round(r, method)
            txs = self._set_tokens(scores)
            self.wait_for_txs(txs)
            self._gas_history[r+1] = self.get_gas_used()
            global_loss = self.evaluate_global(r)
            wandb.log({"global_loss": global_loss})
        self._print(f"Done evaluating. Gas used: {self.get_gas_used()}")
        

    def is_evaluator(self):
        return self._contract.evaluator() == self._contract.address

    def get_current_global_model(self):
        """
        Calculate, or get from cache, the current global model.
        """
        current_training_round = self._contract.currentRound()
        current_global_model = self._get_global_model(current_training_round)
        return current_global_model

    @methodtools.lru_cache()
    def evaluate_global(self, training_round):
        """
        Evaluate the global model at the given training round.
        """
        model = self._get_global_model(training_round)
        loss = self._evaluate_model(model)
        return loss

    def evaluate_current_global(self):
        """
        Evaluate the current global model using own data.
        """
        current_training_round = self._contract.currentRound()
        return self.evaluate_global(current_training_round)

    def predict(self):
        model = self.get_current_global_model()
        # model = model.send(self._worker)
        predictions = []
        with torch.no_grad():
            for data, labels in self._test_loader:
                data, labels = data.float(), labels.float()
                pred = model(data)  # .get()
                predictions.append(pred)
        return torch.stack(predictions)

    def wait_for_round(self, n):
        self._print(
            f"Waiting for round {n} ({self._contract.secondsRemaining()} seconds remaining)...")
        while(self._contract.currentRound() < n):
            time.sleep(self.CURRENT_ROUND_POLL_INTERVAL)
        self._print(f"Round {n} started")

    def get_gas_history(self):
        return self._gas_history

    @methodtools.lru_cache()
    def _get_global_model(self, training_round):
        """
        Calculate global model at the the given training round by aggregating updates from previous round.

        This can only be done if training_round matches the current contract training round.
        """
        model_cids = self._get_cids(training_round - 1)
        models = self._get_models(model_cids)
        avg_model = self._avg_model(models)
        return avg_model

    def _train_single_round(self, round_num, batch_size, epochs, learning_rate, dp_params):
        """
        Run a round of training using own data, upload and record the contribution.
        """
        model = self.get_current_global_model()
        self._print(f"Training model, round {round_num}...")
        model = self._train_model(
            model, batch_size, epochs, learning_rate, dp_params)
        uploaded_cid = self._upload_model(model)
        self._print(f"Adding model update..., local model cid : {uploaded_cid}, round : {round_num}")
        tx = self._record_model(uploaded_cid, round_num)
        return tx,model

    def _train_model(self, model, batch_size, epochs, lr, dp_params):
        ### train 데이터 로더 수정
        # train_loader = torch.utils.data.DataLoader(
        #     sy.BaseDataset(self._data, self._targets),
        #     batch_size=batch_size)
        train_loader = torch.utils.data.DataLoader(self._data, batch_size=batch_size)
        ###
        # model = model.send(self._worker)
        model.train()
        if dp_params is not None:
            eps = epsilon(
                    N=self.data_length,
                    batch_size=batch_size,
                    noise_multiplier=dp_params['noise_multiplier'],
                    epochs=epochs,
                    delta=dp_params['delta']
            )
            self._print(f"DP enabled, eps={eps} delta={dp_params['delta']}")
            optimizer = pyvacy.optim.DPSGD(
                params=model.parameters(),
                lr=lr,
                batch_size=batch_size,
                l2_norm_clip=dp_params['l2_norm_clip'],
                noise_multiplier=dp_params['noise_multiplier']
            )
        else:
            optimizer = torch.optim.SGD(
                params=model.parameters(),
                lr=lr,
                momentum=0.9
            )
        for epoch in range(epochs):
            for data, labels in train_loader:
                optimizer.zero_grad()
                pred = model(data)
                loss = self._criterion(pred, labels)
                loss.backward()
                optimizer.step()
                # wandb.watch(model)
                # wandb.log({"loss": loss})
        # model.get()
        return model

    def _evaluate_model(self, model, *localFlag):
        model = model  # .send(self._worker)
        model.to(device)
        # self._test_loader.to(device)
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for data, labels in self._test_loader:
                # data.to(device)
                pred = model(data)
                total_loss += self._criterion(pred, labels
                                              ).item()
                # ).get().item()
        avg_loss = total_loss / len(self._test_loader)
        ##HERE!
        if localFlag:
            wandb.log({"avg_loss": avg_loss})
        return avg_loss

    def _record_model(self, uploaded_cid, training_round):
        """Records the given model IPFS cid on the smart contract."""
        return self._contract.addModelUpdate(uploaded_cid, training_round)

    def _get_cids(self, training_round):
        if training_round < 0 or training_round > self._contract.currentRound():
            raise ValueError(
                f"training_round={training_round} out of bounds [0, {self._contract.currentRound()}]")
        if training_round == 0:
            return [self._contract.genesis()]
        cids = self._contract.updates(training_round)
        if not cids:  # if cids is empty, refer to previous round
            return self._get_cids(training_round - 1)
        return cids

    def _get_models(self, model_cids):
        models = []
        for cid in model_cids:
            model = self._ipfs_client.get_model(cid)
            models.append(model)
        return models

    def _get_genesis_model(self):
        gen_cid = self._contract.genesis()
        return self._ipfs_client.get_model(gen_cid)

    def _avg_model(self, models):
        avg_model = self._model_constructor()
        # avg_model.to(device)
        with torch.no_grad():
            for params in avg_model.parameters():
                params *= 0
            for client_model in models:
                for avg_param, client_param in zip(avg_model.parameters(), client_model.parameters()):
                    avg_param += client_param / len(models)
        return avg_model

    def _evaluate_single_round(self, training_round, method):
        """
        Provide Shapley Value score for each update in the given training round.
        """
        self._print(f"Evaluating updates in round {training_round}...")

        cids = self._get_cids(training_round)

        if method == 'shapley':
            def characteristic_function(*c):
                return self._marginal_value(training_round, *c)
            scores = contribution.values(
                characteristic_function, cids)
        if method == 'step':
            scores = {}
            idx = 0
            for cid in cids:
                scores[cid] = self._marginal_value(training_round, cid)
                ########### Trainer 2로만 두고 했으니 유의 바람!!
                print(idx)
                print(scores[cid])
                if idx==0:
                    wandb.log({"marginal_value_trainer1": scores[cid]})
                elif idx==1:
                    wandb.log({"marginal_value_trainer2": scores[cid]})
                elif idx==2:
                    wandb.log({"marginal_value_trainer3": scores[cid]})
                else:
                    wandb.log({"marginal_value_trainer4": scores[cid]})
                idx+=1

        self._print(
            f"Scores in round :{training_round} are :{list(scores.values())}: and cids :{cids}")
        return scores

    def _set_tokens(self, cid_scores):
        """
        Record the given Shapley value scores for the given contributions.
        """
        txs = []
        self._print(f"Setting {len(cid_scores.values())} scores...")
        for cid, score in cid_scores.items():
            num_tokens = max(0, int(score * self.TOKENS_PER_UNIT_LOSS))
            tx = self._contract.setTokens(cid, num_tokens)
            txs.append(tx)
        return txs

    @methodtools.lru_cache()
    #기여도측정?
    def _marginal_value(self, training_round, *update_cids):
        """
        The characteristic function used to calculate Shapley Value.
        The Shapley Value of a coalition of trainers is the marginal loss reduction
        of the average of their models
        """
        start_loss = self.evaluate_global(training_round)
        models = self._get_models(update_cids)
        avg_model = self._avg_model(models)
        loss = self._evaluate_model(avg_model,True)
        return start_loss - loss

wandb.init(project="2cp",entity="daeyeolkim")
wandb.run.name = "FeMNist-Eval-round15"
wandb.config = {
  "learning_rate": 0.3,
  "epochs": 2,
  "batch_size": 64
}

TRAINING_ITERATIONS = 15
TRAINING_HYPERPARAMS = {
    'final_round_num': TRAINING_ITERATIONS,
    'batch_size': 64,
    'epochs': 2,
    'learning_rate': 0.3,
}
TORCH_SEED = 8888
EVAL_METHOD = 'step'
ROUND_DURATION = 30000


model_name = "cifar-10-client-"
option="data"
torch.manual_seed(TORCH_SEED)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    

class FeMnistNet(nn.Module):
  def __init__(self): # layer 정의
        super(FeMnistNet, self).__init__()
        # input size = 28x28 
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # input channel = 1, filter = 10, kernel size = 5, zero padding = 0, stribe = 1
        # ((W-K+2P)/S)+1 공식으로 인해 ((28-5+0)/1)+1=24 -> 24x24로 변환
        # maxpooling하면 12x12
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # input channel = 1, filter = 10, kernel size = 5, zero padding = 0, stribe = 1
        # ((12-5+0)/1)+1=8 -> 8x8로 변환
        # maxpooling하면 4x4
        self.drop2D = nn.Dropout2d(p=0.25, inplace=False) # 랜덤하게 뉴런을 종료해서 학습을 방해해 학습이 학습용 데이터에 치우치는 현상을 막기 위해 사용
        self.mp = nn.MaxPool2d(2)  # 오버피팅을 방지하고, 연산에 들어가는 자원을 줄이기 위해 maxpolling
        self.fc1 = nn.Linear(320,100) # 4x4x20 vector로 flat한 것을 100개의 출력으로 변경
        self.fc2 = nn.Linear(100,62) # 100개의 출력을 10개의 출력으로 변경

  def forward(self, x):
        x = F.relu(self.mp(self.conv1(x))) # convolution layer 1번에 relu를 씌우고 maxpool, 결과값은 12x12x10
        x = F.relu(self.mp(self.conv2(x))) # convolution layer 2번에 relu를 씌우고 maxpool, 결과값은 4x4x20
        x = self.drop2D(x)
        x = x.view(x.size(0), -1) # flat
        x = self.fc1(x) # fc1 레이어에 삽입
        x = self.fc2(x) # fc2 레이어에 삽입
        return F.log_softmax(x) # fully-connected layer에 넣고 logsoftmax 적용

def custom_cifar_crowdsource():
    testset_path = 'crowdsource_back/back/src/utils/pytorch_femnist/data/user_data/evaluator_data.json'
    with open(testset_path,'r') as f:
        eval_dataset = json.load(f)
        eval_data = eval_dataset['x']
        eval_targets = eval_dataset['y']
    # print(eval_data)

    my_eval_data = MyMnist(eval_data, eval_targets, True)
    # print(my_eval_data.x_data) 
    eval = CrowdsourceClient("Evaluator",my_eval_data, eval_targets,FeMnistNet,F.cross_entropy,0,deploy=True)
    my_dict = {"eval_cont_addr":eval.contract_address}   
    tf = open("eval_contract.json","w")
    json.dump(my_dict,tf)
    tf.close()
    eval.set_genesis_model(
        round_duration=ROUND_DURATION,
        max_num_updates=int(sys.argv[1])
    )
    eval.evaluate_until(TRAINING_ITERATIONS,EVAL_METHOD)
custom_cifar_crowdsource()
