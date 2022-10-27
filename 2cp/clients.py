import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import normalize
import torch.optim as optim
# from data_load import MyShakespeare
import time
import methodtools
import torch
import pyvacy.optim
from pyvacy.analysis import moments_accountant as epsilon
from ipfs_client import IPFSClient
from contract_clients import CrowdsourceContractClient, ConsortiumContractClient
import shapley
import wandb
import json
from torchvision.transforms import ToTensor
import sys
import os
import threading


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# wandb.init(project=os.environ.get("WANDB_PROJECT_NAME"),entity=os.environ.get("WANDB_USER_NAME"))

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

    def __init__(self, name, model_constructor, contract_constructor, account_idx, contract_address=None, deploy=False):
        super().__init__(name, model_constructor,
                         contract_constructor, account_idx, contract_address, deploy)
        self._ipfs_client = IPFSClient(model_constructor)

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
            # wandb.log({"global_loss": global_loss})
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
        train_loader = torch.utils.data.DataLoader(self._data, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
        ###
        # model = model.send(self._worker)
        model.to(device)
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
        # print(model)
        model.to(device)
        # self._test_loader.to(device)
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
        ##HERE!
        # if localFlag:
            # wandb.log({"avg_loss": avg_loss})
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
            scores = shapley.values(
                characteristic_function, cids)
        if method == 'step':
            scores = {}
            idx = 0
            for cid in cids:
                scores[cid] = self._marginal_value(training_round, cid)
                ########### Trainer 2로만 두고 했으니 유의 바람!!
                print(idx)
                print(scores[cid])
                # if idx==0:
                #     wandb.log({"marginal_value_trainer1": scores[cid]})
                # elif idx==1:
                #     wandb.log({"marginal_value_trainer2": scores[cid]})
                # elif idx==2:
                #     wandb.log({"marginal_value_trainer3": scores[cid]})
                # else:
                #     wandb.log({"marginal_value_trainer4": scores[cid]})
                # idx+=1

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
            self._print(f"cid :{cid} score :{score}: and tokens :{num_tokens}")
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
            txs.append(
                self._contract.addAux(evaluator)
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

    def __init__(self, name, data, targets, model_constructor, model_criterion, account_idx, contract_address=None, deploy=False):
        super().__init__(name,
                         model_constructor,
                         ConsortiumContractClient,
                         account_idx,
                         contract_address)
        self.data_length = min(len(data), len(targets))
        self._data = data
        self._targets = targets
        self._criterion = model_criterion
        self._main_client = CrowdsourceClient(name + " (main)",
                                              data,
                                              targets,
                                              model_constructor,
                                              model_criterion,
                                              account_idx,
                                              contract_address=self._contract.main())
        self._aux_clients = {}  # cache, updated every time self._get_aux_clients() is called

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

    def evaluate_until(self, final_training_round, method):
        eval_clients = self._get_eval_clients()
        threads = [
            threading.Thread(
                target=eval_client.evaluate_until,
                args=(final_training_round, method),
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
                    contract_address=aux
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



