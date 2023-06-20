import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import normalize
import torch.optim as optim
import time
import methodtools
import torch
import pyvacy.optim
from pyvacy.analysis import moments_accountant as epsilon
from ipfs_client import IPFSClient
from contract_clients import CrowdsourceContractClient, ConsortiumContractClient
from token_contract import TokenContractClient
import contribution
import json
from torchvision.transforms import ToTensor
import sys
import os
import threading
from client_selection import random_selection,fcfs_selection,all_selection,score_order
import random
from collections import OrderedDict

import wandb


# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# print(device)
# wandb.init(project=os.environ.get("WANDB_PROJECT_NAME"),entity=os.environ.get("WANDB_USER_NAME"))

class _BaseClient:
    """
    Abstract base client containing common features of the clients for both the
    Crowdsource Protocol and the Consortium Protocol.
    """

    def __init__(self, name, model_constructor, contract_constructor,account_idx, token_contract_constructor, contract_address=None, deploy=False, token_contract_address=None, token_deploy = False):
        self.evt = threading.Event()
        self.name = name
        self._model_constructor = model_constructor
        if deploy:
            self._print("Deploying contract...")
        if token_deploy:
            self._print("Deploying token contract...")
        # self._print(f'contract_constructor : {contract_constructor}')
        self._contract = contract_constructor(
            account_idx, contract_address, deploy)
        self._token_contract = token_contract_constructor(account_idx,token_contract_address,token_deploy)
        self._account_idx = account_idx
        self.address = self._contract.address
        self.token_contract_address = self._token_contract.contract_address
        self.contract_address = self._contract.contract_address
        # self._print(f"name : {self.name}, model_constructor : {self._model_constructor},provider : {provider}, account_idx : {account_idx}, contract_address : {contract_address}, deploy : {deploy} ")
        self._print(
            f"Connected to contract at address {self.contract_address}")
        self._print(
            f"Connected to Token contract at address {self.token_contract_address}")

    def get_token_count(self, address=None, training_round=None):
        return self._contract.countTokens(address, training_round)

    def get_total_token_count(self, training_round=None):
        return self._contract.countTotalTokens(training_round)

    def get_gas_used(self):
        return self._contract.get_gas_used()
    
    # temp
    def get_accounts(self,max_num_updates):
        return self._contract.get_accounts(max_num_updates)

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

    def __init__(self, name, model_constructor,contract_constructor,account_idx,token_contract_constructor,contract_address=None, deploy=False, token_contract_address = None, token_deploy = False):
        super().__init__(name, model_constructor,
                         contract_constructor, account_idx,token_contract_constructor, contract_address, deploy,
                         token_contract_address,token_deploy)
        self._ipfs_client = IPFSClient(model_constructor,"/ip4/127.0.0.1/tcp/5001")

    def set_genesis_model(self, round_duration,scenario,max_num_updates=0):
        """
        Create, upload and record the genesis model.
        """
        self._print("Setting genesis...")
        genesis_model = self._model_constructor()
        # genesis_model.to(device)
        genesis_cid = self._upload_model(genesis_model)
        account_list = self.get_accounts(max_num_updates)
        self._print(f"accounts setting ... : {account_list}")
        self._print(f"Genesis model cid : {genesis_cid}")
        tx = self._contract.setGenesis(
            genesis_cid, round_duration, max_num_updates,account_list)
        self.wait_for_txs([tx])
        # if scenario == "crowdsource":
        #     # for i in range (max_num_updates):
        #     #     self._print(f"init genesis round's trainers : {account_list[i]}")
        #     #     # 첫 라운드는 모든 트레이너 등록
        #     #     tx = self._contract.setCurTrainer(1,account_address=account_list[i])
        #     #     self.wait_for_txs([tx])
        #     self._print(f"init genesis round's trainers : {account_list}")
        #     tx = self._contract.setCurTrainer(1,account_address=account_list)
        #     self.wait_for_txs([tx])
        # else:
        #     for i in range (max_num_updates):
        #         self._print(f"init genesis round's trainers : {account_list[i]}")
        #         # tx = self._contract.setConsEvaluator()
        #         # self.wait_for_txs([tx])
        #         # 첫 라운드는 모든 트레이너 등록
        #         # tx = self._contract.setConsCurTrainer(1,account_list[i], self.contract_address)
        #         # self.wait_for_txs([tx])

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

    def __init__(self, name, data, targets, model_constructor, model_criterion, account_idx, batch_size = 64, contract_address=None, deploy=False, device_num="0",did_address=None, token_contract_address = None, token_deploy=False):
        super().__init__(name,
                         model_constructor,
                         
                         CrowdsourceContractClient,
                         account_idx,
                         TokenContractClient,
                         contract_address,
                         deploy,
                         token_contract_address,
                         token_deploy
                         )
        self.did = did_address
        self.data_length = data.__len__()
        self.g = torch.Generator()
        self.g.manual_seed(8888)
        self.device = torch.device('cuda:'+ device_num) if torch.cuda.is_available() else torch.device('cpu')
        self._criterion = model_criterion
        self._data = data  # .send(self._worker)
        self._targets = targets  # .send(self._worker)
        self._test_loader = torch.utils.data.DataLoader(self._data, batch_size=batch_size,
                                         shuffle=False, num_workers=0, worker_init_fn =self.seed_worker , generator = self.g)
        ###
        # train loader is defined each time training is run
        self._gas_history = {}
        self._account_idx = account_idx
        
    # all
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # trainer
    def train_one_round(self, batch_size, epochs, learning_rate, dp_params=None):
        cur_round = self._contract.currentRound()
        # print("cur_round : ",cur_round)
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
        

    # trainer
    def train_until(self, final_round_num, batch_size, epochs, learning_rate, dp_params=None):
        
        start_round = self._contract.currentRound()
        for r in range(start_round, final_round_num+1):
            self.wait_for_round(r)
            # 라운드 시작 시 해당 라운드의 등록된 trainer인지 검사
            account = self._contract.get_account()
            self._print(f"Trainer account : {account}")
            isTrainer = self._contract.isTrainer(r,account_address=account)
            self._print(f"round {r}'s Train Flag : {isTrainer}")
            if isTrainer:
             # true를 반환했을 경우에는 학습 진행, 그렇지 않을 경우 학습 중단
                tx,model = self._train_single_round(
                    r,
                    batch_size,
                    epochs,
                    learning_rate,
                    dp_params
                )
                self.wait_for_txs([tx])
                self._gas_history[r] = self.get_gas_used()
            else : 
                self._print(f"this trainer is eliminated.")
                self._print(f"Done training. Gas used: {self.get_gas_used()}")
                break
            self._print(f"Done training. Gas used: {self.get_gas_used()}")
            # if self.name == "trainer1" :
            #         wandb.log({"training gas":self.get_gas_used()})
            
    # evaluator
    def weight_fn(self,did_info, accounts,scores,alpha):
        new_score = scores
        for cid,account in accounts.items():
            for address,did in did_info.items() :  
                if account == address:
                    _weighted_score = scores[cid]
                    _weighted_score *= alpha
                    weighted_score = _weighted_score + scores[cid]
                    new_score[cid] = weighted_score
        return new_score

    # evaluator
    def evaluate_until(self, final_round_num, method, scenario, selection_method="all", did_info = None, alpha=0.1):
        self._gas_history[1] = self.get_gas_used()
        for r in range(1, final_round_num+1):
            # self.wait_for_round(r + 1)
            self._print(f"wait for trainers...")
            self.wait_for_round(r)
            
            while(self._contract.waitTrainers(r) != True):
                time.sleep(self.CURRENT_ROUND_POLL_INTERVAL)
            scores,accounts = self._evaluate_single_round(r, method ,scenario)
            self._print(f"Before weight score : {scores}")
            new_score = self.weight_fn(did_info,accounts,scores,alpha)
            self._print(f"After weight score : {new_score}")
            tx = self._set_tokens(new_score)
            self.wait_for_txs([tx])
            self._gas_history[r] = self.get_gas_used()
            global_loss = self.evaluate_global(r)
            # wandb.log({"global_loss": global_loss})
            # CS 방법선택 : random, fcfs, all(default)
            client_list = self._contract.getCurTrainers(r)
            self._print(client_list)
            if selection_method == "all":
                client_list = all_selection(client_list)
            elif selection_method == "random":
                client_list = random_selection(client_list)
            elif selection_method == "fcfs":
                client_list = fcfs_selection(client_list)
            elif selection_method == "score_order":
                score_list = []
                client_list = []
                for key,value in scores.items() :
                    client_score = self.find_score(key, accounts[key])
                    score_list.append(client_score)
                    client_list.append(accounts[key])
                client_list = score_order(client_list,score_list)
            else:
                self._print("Client selection method is not assigned.")
                break
            
            # Save global model to IPFS and BlockChain
            tx = self.save_global_model(r)
            self.wait_for_txs([tx])
            # 해당 라운드 Esave_glovaluation 완료
            self._print(f"round {r} evaluation complete")
            tx = self._contract.completeEval(r)
            self.wait_for_txs([tx])

            # 다음 라운드(r+1)의 setCurTrainer, changeMaxNumUpdates 사용해서 상태변경
            self._print(f"client list changed by selection method '{selection_method}' : {client_list} ")
            self._print(f"setting next round's clients ... ")
            # for i in range (len(client_list)):
            #     tx = self._contract.setCurTrainer(r+1,account_address = client_list[i])
            #     self.wait_for_txs([tx])
            tx = self._contract.setCurTrainer(r+1,account_address = client_list)
            self.wait_for_txs([tx])

            self._print(f"Changing max number of updates ... ")
            tx = self._contract.changeMaxNumUpdates(len(client_list))
            self.wait_for_txs([tx])
            max_num = self._contract.getmaxNum()
            self._print(f"max number of update : {max_num}")

            if max_num == 0 :
                break
            # wandb.log({"evaluation_gas":self.get_gas_used()})
            # 라운드 넘기기
            self._print(f"skipping round to {r+1}")
            tx = self._contract.skipRound(r)
            self.wait_for_txs([tx])
        threading.Event()    
        self._print(f"Done evaluating. Gas used: {self.get_gas_used()}")
        self.evt.set()
        

    # evaluator
    def find_score(self, model_cid, account):
        score_info = self._contract.getScores(model_cid)
        if(score_info[0] != account):
            self._print(f"this account is not owner of this cid")
            sys.exit(1)
        else:
            score = score_info[1]
        return score

    # all
    def is_evaluator(self):
        return self._contract.evaluator() == self._contract.address

    # all
    def get_current_global_model(self):
        """
        Calculate, or get from cache, the current global model.
        """
        current_training_round = self._contract.currentRound()
        current_global_model = self._get_global_model(current_training_round)
        return current_global_model

    # evaluator
    @methodtools.lru_cache()
    def evaluate_global(self, training_round):
        """
        Evaluate the global model at the given training round.
        """
        model = self._get_global_model(training_round)
        loss = self._evaluate_model(model)
        return loss
    
    # evaluator
    def save_global_model (self, training_round):
        # get_global_model 사용 시 이전 라운드*(r-1) 의 글로벌 모델을 받아옴
        # 따라서 현재 라운드 저장이 필요,,
        # 1. _avg_global_model 사용해서 aggregation된 현재 라운드 글로벌 모델 생성
        # model = self._get_global_model(training_round)
        model = self._avg_global_model(training_round)
        # for param in model.parameters():
        #     self._print(f"{param}")
        # modified -> Add server's aggregation process 
        avg_cid = self._upload_model(model)
        tx = self._contract.saveGlobalmodel(avg_cid, training_round)
        # torch.save(model.state_dict(),"/media/hdd1/es_workspace/BCFL_framework_es/crowdsource/avg_model1.pth")
        return tx

    # evaluator
    def _avg_global_model(self, training_round):
        avg_model = self._model_constructor()
        cids = self._get_cids(training_round)
        models = self._get_models(cids) # who's cid? need information
        # with torch.no_grad():
        #     for params in avg_model.parameters():
        #         params *= 0
        #     for client_model in models:
        #         for avg_param, client_param in zip(avg_model.parameters(), client_model.parameters()):
        #             avg_param += client_param
        #     avg_param = avg_param / len(models)
        averaged_weights = []

        # averaged_weights = self.average_ordered_dict_values(models)

        # for 

        for layer_index in range(len(avg_model.state_dict().keys())):
            model_weights = []
            for model in models:
                # for value in list(model.state_dict().values())[layer_index]:
                #     value = torch.trunc(value*1000) / 1000
                    # self._print(value)
                model_weights.append(list(model.state_dict().values())[layer_index])
            # 여기 수정
            # for model_weight in model_weights:
            # model_weights = torch.Tensor(model_weights)
            averaged_layer_weights = torch.mean(torch.stack(model_weights,dim=0),dim=0)
            # averaged_layer_weights = torch.trunc((sum(model_weights) / len(model_weights))*1000)*1000
            # averaged_layer_weights = averaged_layer_weights/len(model_weights)
            averaged_layer_weights = averaged_layer_weights*100
            averaged_layer_weights = torch.trunc(averaged_layer_weights)
            averaged_layer_weights = averaged_layer_weights/100

            averaged_weights.append(averaged_layer_weights)
        
        avg_model.load_state_dict(dict(zip(avg_model.state_dict().keys(),averaged_weights)))

        return avg_model

    # evaluator
    def evaluate_current_global(self):
        """
        Evaluate the current global model using own data.
        """
        current_training_round = self._contract.currentRound()
        return self.evaluate_global(current_training_round)

    # evaluator
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

    # all
    def wait_for_round(self, n):
        self._print(
            f"Waiting for round {n} ({self._contract.secondsRemaining()} seconds remaining)...")
        while(self._contract.currentRound() < n):
            time.sleep(self.CURRENT_ROUND_POLL_INTERVAL)
        self._print(f"Round {n} started")

    # all
    def get_gas_history(self):
        return self._gas_history

    @methodtools.lru_cache()
    # def _get_global_model(self, training_round):
    #     """
    #     Calculate global model at the the given training round by aggregating updates from previous round.

    #     This can only be done if training_round matches the current contract training round.
    #     """
    #     model_cids = self._get_cids(training_round - 1)
    #     models = self._get_models(model_cids)
    #     avg_model = self._avg_model(models)
    #     return avg_model
    
    # all
    def _get_global_model(self, training_round):
        if training_round == 1:
            model_cid = self._contract.genesis()
        else:
            model_cid = self._contract.getGlobalmodel(training_round-1)
        current_global_model = self._ipfs_client.get_model(model_cid)
        return current_global_model

    # trainer
    def _train_single_round(self, round_num, batch_size, epochs, learning_rate, dp_params):
        """
        Run a round of training using own data, upload and record the contribution.
        """
        model = self.get_current_global_model()
        # global_params = list(model.parameters())
        self._print(f"Training model, round {round_num}...")
        # self._print(f"Device Info : {self.device}")
        model = self._train_model(
            model, batch_size, epochs, learning_rate, dp_params)
        uploaded_cid = self._upload_model(model)
        self._print(f"Adding model update..., local model cid : {uploaded_cid}, round : {round_num}")
        tx = self._record_model(uploaded_cid, round_num)
        # params = list(model.parameters())
        return tx,model

    # trainer
    def _train_model(self, model, batch_size, epochs, lr, dp_params):
        
        train_loader = torch.utils.data.DataLoader(self._data, batch_size=batch_size,
                                          shuffle=True, num_workers=0, worker_init_fn =self.seed_worker, generator=self.g)
        # if self.did is None :
        #     transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)])
        #     train_loader = train_loader.transform(transform)
        # ##
        # model = model.send(self._worker)
        model.to(self.device)
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
                # wandb.log({f"trainer{self._account_idx} loss": loss})
        # model.get()
        return model

    # evaluator
    def _evaluate_model(self, model, *localFlag):
        model = model  # .send(self._worker)
        # print(model)
        model.to(self.device)
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for data, labels in self._test_loader:
                # data.to(device)
                # print(data.size())

                pred = model(data)
                # print("prediction : ",pred)
                # print("label : ",labels)
                total_loss += self._criterion(pred, labels
                                              ).item()
                # ).get().item()
        
        avg_loss = total_loss / len(self._test_loader)
        return avg_loss

    # all
    def _record_model(self, uploaded_cid, training_round):
        """Records the given model IPFS cid on the smart contract."""
        return self._contract.addModelUpdate(uploaded_cid, training_round)

    # all
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

    # all
    def _get_models(self, model_cids):
        models = []
        for cid in model_cids:
            model = self._ipfs_client.get_model(cid)
            models.append(model)
        return models

    # all
    def _get_genesis_model(self):
        gen_cid = self._contract.genesis()
        return self._ipfs_client.get_model(gen_cid)

    # evaluator
    def _avg_model(self, models):
        avg_model = self._model_constructor()
        averaged_weights = []
        # with torch.no_grad():
        #     for params in avg_model.parameters():
        #         params *= 0
        #     for client_model in models:
        #         for avg_param, client_param in zip(avg_model.parameters(), client_model.parameters()):
        #             avg_param += client_param / len(models)
        # return avg_model
        for layer_index in range(len(avg_model.state_dict().keys())):
            model_weights = []
            for model in models:
                # for value in list(model.state_dict().values())[layer_index]:
                    # value = torch.trunc(value*1000) / 1000
                    # self._print(value)
                model_weights.append(list(model.state_dict().values())[layer_index])
            # 여기 수정
            # for model_weight in model_weights:
            # model_weights = torch.Tensor(model_weights)
            averaged_layer_weights = torch.mean(torch.stack(model_weights,dim=0),dim=0)
            # averaged_layer_weights = torch.trunc((sum(model_weights) / len(model_weights))*1000)*1000
            # averaged_layer_weights = averaged_layer_weights/len(model_weights)
            averaged_layer_weights = averaged_layer_weights*100
            averaged_layer_weights = torch.trunc(averaged_layer_weights)
            averaged_layer_weights = averaged_layer_weights/100

            averaged_weights.append(averaged_layer_weights)
        
        avg_model.load_state_dict(dict(zip(avg_model.state_dict().keys(),averaged_weights)))

        return avg_model

    # evaluatorß
    def _evaluate_single_round(self, training_round, method, scenario):
        """
        Provide Shapley Value score for each update in the given training round.
        scenario = "crowdsource" || "consortium"
        """
        self._print(f"Evaluating updates in round {training_round}...")

        cids = self._get_cids(training_round)

        if method == 'shapley':
            def characteristic_function(*c):
                return self._marginal_value(training_round, *c)
            scores = contribution.shapley_values(
                characteristic_function, cids)
           
            if scenario == 'crowdsource':
                # Connect to wandb
                for index, score in enumerate(scores) :
                    trainer_name = "SV_trainer"+str(index+1)
                    # wandb.log({trainer_name: scores[score]})
            elif scenario == 'consortium':
                # Connect to wandb
                print("test")
            else : 
                print("Error - Input scenario first")
        if method == 'step': # marginal-gain
            scores = {}
            idx = 0
            for cid in cids:
                scores[cid] = self._marginal_value(training_round, cid)
                print(idx)
                print(scores[cid])
        if method == 'loo':
            def characteristic_function(*c):
                return self._marginal_value(training_round,*c)
            scores = contribution.loo(
                characteristic_function,cids)

            if scenario == 'crowdsource':
                # Connect to wandb
                for index, score in enumerate(scores) :
                    trainer_name = "SV_trainer"+str(index+1)
                    # wandb.log({trainer_name: scores[score]})
            elif scenario == 'consortium':
                # Connect to wandb
                print("test")
            else : 
                print("Error - Input scenario first")

        accounts = {}
        for key, value in scores.items():
            ## key : cid
            ## value : score
            client_account = self._contract.getAccountfromUpdate(key)
            accounts[key] = client_account
            client_score = int(value * self.TOKENS_PER_UNIT_LOSS)
            tx = self._contract.saveScores(key,client_account, client_score)
            self.wait_for_txs([tx])
            # score = self._contract.getScores(key)

        self._print(
            f"Scores in round :{training_round} are :{list(scores.values())}: and cids :{cids}")
        return scores,accounts

    # evaluator
    def _set_tokens(self, cid_scores):
        """
        Record the given Shapley value scores for the given contributions.
        """
        txs = []
        self._print(f"Setting {len(cid_scores.values())} scores...")
        for cid, score in cid_scores.items():
            score_info = self._contract.getScores(cid)
            account = score_info[0]
            num_tokens = max(0, int(score * self.TOKENS_PER_UNIT_LOSS))
            self._print(f"account : {account} cid :{cid} score :{score}: and tokens :{num_tokens}")
            # tx = self._contract.setTokens(cid, num_tokens)
            # txs.append(tx)
            tx = self._token_contract.transfer(account,num_tokens)
            txs.append(tx)
        return txs

    # evaluator
    @methodtools.lru_cache()
    def _marginal_value(self, training_round, *update_cids):
        """
        The characteristic function used to calculate Shapley Value.
        The Shapley Value of a coalition of trainers is the marginal loss reduction
        of the average of their models
        """
        start_loss = self.evaluate_global(training_round)
        models = self._get_models(update_cids) 
        avg_model = self._avg_model(models)
        #avg model loss -> NaN (step, all)
        loss = self._evaluate_model(avg_model,True)
        
        return start_loss - loss


class ConsortiumSetupClient(_GenesisClient):
    """
    Client which sets up the Consortium Protocol but does not participate.
    """

    def __init__(self, name, model_constructor, account_idx, contract_address=None, deploy=False):
        super().__init__(name,
                         model_constructor,
                         ConsortiumContractClient,
                         account_idx,
                         contract_address,
                         deploy)

    def add_auxiliaries(self, evaluators):
        self._print(f"Setting {len(evaluators)} auxiliaries...")
        txs = []
        for evaluator in evaluators:
            trainer_list = []
            for trainer in evaluators:
                if trainer is not evaluator:
                    trainer_list.append(trainer)
            txs.append(
                self._contract.addAux(evaluator,trainer_list)
            )
        self.wait_for_txs(txs)

    def get_gas_history(self):
        return {
            '1': self.get_gas_used()
        }


class ConsortiumClient(_BaseClient):
    """
    Full client for the Consortium Protocol.
    """

    def __init__(self, name, data, targets, model_constructor, model_criterion, account_idx, batch_size = 64,contract_address=None, deploy=False, device_num="0"):
        super().__init__(name,
                         model_constructor,
                         ConsortiumContractClient,
                         account_idx,
                         contract_address)
        self.data_length = min(len(data), len(targets))
        self._data = data
        self._targets = targets
        self._criterion = model_criterion
        self.name = name
        self._main_client = CrowdsourceClient(name + " (main)",
                                              data,
                                              targets,
                                              model_constructor,
                                              model_criterion,
                                              account_idx,
                                              batch_size=batch_size,
                                              contract_address=self._contract.main(),
                                              device_num=device_num)
        self._aux_clients = {}  # cache, updated every time self._get_aux_clients() is called
        self.device_num = device_num

    def train_until(self, final_round_num, batch_size, epochs, learning_rate, dp_params=None):
        train_clients = self._get_train_clients()
        threads = [
            threading.Thread(
                target=train_client.train_until,
                args=(final_round_num, batch_size, epochs, learning_rate, dp_params),
                daemon=True
            ) for train_client in train_clients
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def evaluate_until(self, final_training_round, method, scenario,selection_method="all"):
        eval_clients = self._get_eval_clients()
        threads = [
            threading.Thread(
                target=eval_client.evaluate_until,
                args=(final_training_round, method,scenario),
                kwargs = {
                    "selection_method" : selection_method
                },
                daemon=True
            ) for eval_client in eval_clients
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def get_current_global_model(self):
        return self._main_client.get_current_global_model()

    def evaluate_current_global(self):
        return self._main_client.evaluate_current_global()

    def predict(self):
        return self._main_client.predict()

    def wait_for_round(self, n):
        self._main_client.wait_for_round(n)
        for client in self._get_aux_clients():
            client.wait_for_round(n)

    def get_gas_used(self):
        gas_used = self._main_client.get_gas_used()
        for aux in self._get_aux_clients():
            gas_used += aux.get_gas_used()
        return gas_used

    def get_gas_history(self):
        # has a bug
        # total_gas_history = self._main_client.get_gas_history()
        # for aux in self._get_aux_clients():
        #     aux_gas_history = aux.get_gas_history()
        #     for r in aux_gas_history.keys():
        #         if r not in total_gas_history:
        #             total_gas_history[r] = total_gas_history[r-1]
        #         total_gas_history[r] += aux_gas_history[r]
        # return total_gas_history
        gas_histories = {}
        gas_histories[self._main_client.name] = self._main_client.get_gas_history()
        for aux in self._get_aux_clients():
            gas_histories[aux.name] = aux.get_gas_history()
        return gas_histories

    def _get_aux_clients(self):
        """
        Updates self._aux_clients cache then returns it
        """
        for aux in self._contract.auxiliaries():
            if aux not in self._aux_clients.keys():
                self._aux_clients[aux] = CrowdsourceClient(
                    self.name + f" (aux {len(self._aux_clients)+1})",
                    self._data,
                    self._targets,
                    self._model_constructor,
                    self._criterion,
                    self._account_idx,
                    contract_address=aux,
                    device_num=self.device_num
                )
        # No need to check to remove aux clients as the contract does not allow it
        return self._aux_clients.values()

    def _get_train_clients(self):
        aux_clients = self._get_aux_clients()
        train_clients = [
            aux for aux in aux_clients if not aux.is_evaluator()
        ]
        train_clients.append(self._main_client)
        return train_clients

    def _get_eval_clients(self):
        aux_clients = self._get_aux_clients()
        return [
            aux for aux in aux_clients if aux.is_evaluator()]



