import json
import os
import base58
from web3 import HTTPProvider, Web3
from dotenv import load_dotenv

load_dotenv()

# 1. 이더리움 클라이언트
class BaseEthClient:
    """
    An ethereum client.
    """

    PROVIDER_ADDRESS = os.environ.get("RPC_URL")
    NETWORK_ID = os.environ.get("NETWORK_ID")


    def __init__(self, account_idx):
        self._w3 = Web3(HTTPProvider(self.PROVIDER_ADDRESS)) #json rpc 서버 연결(가나슈)

    #    #for ganache 
    #     accounts = self._w3.eth.accounts
    #     self.address = self._w3.eth.accounts[account_idx] # 이더리움 지갑주소
        
    #     self._w3.eth.defaultAccount = self.address # 지정된 지갑주소를 기본주소로

        # for sepolia test
        if account_idx == 0:
            self.address = os.environ.get("SEPOLIA_EAVLUATOR_ACCOUNT")
            self._w3.eth.defaultAccount = self.address
            self.privatekey = os.environ.get("SEPOLIA_EVALUATOR_PRIVATE_KEY")

        else :
            self.address = os.environ.get("SEPOLIA_TRAINER_ACCOUNT")
            self._w3.eth.defaultAccount = self.address
            self.privatekey = os.environ.get("SEPOLIA_TRAINER_PRIVATE_KEY")
        self.txs = [] # 트랜잭션 담을 리스트
    
    # 트랜잭션 해쉬를 입력해서 트랜잭션 상태를 받아옴
    def wait_for_tx(self, tx_hash):
        receipt = self._w3.eth.waitForTransactionReceipt(tx_hash)
        return receipt

    # 특정 어카운트가 사용한 총 가스비 출력
    def get_gas_used(self):
        receipts = [self._w3.eth.getTransactionReceipt(tx) for tx in self.txs]
        gas_amounts = [receipt['gasUsed'] for receipt in receipts]
        return sum(gas_amounts)
    
    #temp : 프론트 붙기 전까지 사용
    def get_accounts(self,max_num_updates):
        account_list = []
        for i in range(1,max_num_updates+1):
            account_list.append(self._w3.eth.accounts[i])
        return account_list
    
    def get_account(self):
        # print(self.address)
        return self.address


# 2. 컨소시엄, 크라우드 컨트랙트의 공통기능 포함
# 계약 설정, byte32 및 기타 유틸리티와의 변환 처리
class _BaseContractClient(BaseEthClient):
    """
    Contains common features of both contract clients.

    Handles contract setup, conversions to and from bytes32 and other utils.
    """

    IPFS_HASH_PREFIX = bytes.fromhex('1220') #IPFS 해쉬값 접두사

    def __init__(self, contract_json_path,account_idx, contract_address, deploy):
        super().__init__(account_idx)

        self._contract_json_path = contract_json_path

        self._contract, self.contract_address = self._instantiate_contract(contract_address, deploy)

    def _instantiate_contract(self, address=None, deploy=False):
        with open(self._contract_json_path) as json_file: 
            crt_json = json.load(json_file)
            abi = crt_json['abi']
            bytecode = crt_json['bytecode']
            if address is None:
                if deploy:
                    # 배포 트랜잭션에 대한 해쉬 반환
                    tx_hash = self._w3.eth.contract(
                        abi=abi,
                        bytecode=bytecode
                    ).constructor().transact() # 계약의 인스턴스가 배포됨
                    self.txs.append(tx_hash)
                    tx_receipt = self.wait_for_tx(tx_hash) 
                    address = tx_receipt.contractAddress
                else:
                    address = crt_json['networks'][self.NETWORK_ID]['address']
        instance = self._w3.eth.contract(
            abi=abi,
            address=address
        )
        return instance, address
 
    def _to_bytes32(self, model_cid):
        bytes34 = base58.b58decode(model_cid)
        assert bytes34[:2] == self.IPFS_HASH_PREFIX, \
            f"IPFS cid should begin with {self.IPFS_HASH_PREFIX} but got {bytes34[:2].hex()}"
        bytes32 = bytes34[2:]
        return bytes32
    
    def _from_bytes32(self, bytes32):
        bytes34 = self.IPFS_HASH_PREFIX + bytes32
        model_cid = base58.b58encode(bytes34).decode()
        return model_cid

class CrowdsourceContractClient(_BaseContractClient):
    """
    Wrapper over the Crowdsource.sol ABI, to gracefully bridge Python data to Solidity.

    The API of this class should match that of the smart contract.
    """

    def __init__(self,  account_idx, address, deploy):
        super().__init__(
            # "../build/contracts/Crowdsource.json",
            os.path.realpath(os.path.dirname(__file__))[0:-3]+"build/contracts/Crowdsource.json",
            account_idx,
            address,
            deploy
        )
        
    # os.system("pwd")
    def evaluator(self):
        return self._contract.functions.evaluator().call({"from":self.address})

    def genesis(self):
        cid_bytes = self._contract.functions.genesis().call({"from":self.address})
        return self._from_bytes32(cid_bytes)

    def updates(self, training_round):
        cid_bytes = self._contract.functions.updates(training_round).call({"from":self.address})
        return [self._from_bytes32(b) for b in cid_bytes]
    
    def saveGlobalmodel(self, model_cid, training_round):
        cid_bytes = self._to_bytes32(model_cid)
        tx = self._contract.functions.saveGlobalmodel(cid_bytes,training_round).transact()
        self.txs.append(tx)
        return tx

    def getGlobalmodel(self, training_round):
        cid_bytes = self._contract.functions.getGlobalmodel(training_round).call({"from":self.address})
        model_cid = self._from_bytes32(cid_bytes)
        return model_cid
        
    def saveScores(self, model_cid, account_address, score):
        cid_bytes = self._to_bytes32(model_cid)
        tx = self._contract.functions.saveScores(cid_bytes, account_address, score).transact()
        self.txs.append(tx)
        return tx

    def getScores(self, model_cid):
        cid_bytes = self._to_bytes32(model_cid)
        account, score = self._contract.functions.getScores(cid_bytes).call({"from":self.address})
        return account,score

    def completeEval(self, training_round):
        tx =  self._contract.functions.completeEval(training_round).transact()
        self.txs.append(tx)
        return tx

    def getCurTrainers(self, training_round):
        current_trainers = self._contract.functions.getCurTrainers(training_round).call({"from":self.address})
        return current_trainers

    def isTrainer(self,training_round,account_address=None):
        if account_address is None:
            account_address = self.address
        trainCheckFlag = self._contract.functions.isTrainer(account_address,training_round).call({"from":self.address})
        return trainCheckFlag
        

    def setCurTrainer(self, training_round, account_address=None, contract_address = None):
        if account_address is None:
            account_address = self.address
        tx =  self._contract.functions.setCurTrainer(account_address,training_round).transact()
        self.txs.append(tx)
        return tx

    def changeMaxNumUpdates(self, max_num):
        tx = self._contract.functions.changeMaxNumUpdates(max_num).transact()
        self.txs.append(tx)
        return tx

    def currentRound(self):
        return self._contract.functions.currentRound().call({"from":self.address})

    def secondsRemaining(self):
        return self._contract.functions.secondsRemaining().call({"from":self.address})

    def countTokens(self, address=None, training_round=None):
        if address is None:
            address = self.address
        if training_round is None:
            training_round = self.currentRound()
        return self._contract.functions.countTokens(address, training_round).call({"from":self.address})

    def countTotalTokens(self, training_round=None):
        if training_round is None:
            training_round = self.currentRound()
        return self._contract.functions.countTotalTokens(training_round).call({"from":self.address})

    def madeContribution(self, address, training_round):
        return self._contract.functions.madecontribution(address, training_round).call({"from":self.address})

    def setGenesis(self, model_cid, round_duration, max_num_updates,accounts):
        cid_bytes = self._to_bytes32(model_cid)
        self._contract.functions.setGenesis(
            cid_bytes, round_duration, max_num_updates,accounts).call({"from":self.address})
        tx = self._contract.functions.setGenesis(cid_bytes, round_duration, max_num_updates, accounts).transact()
        self.txs.append(tx)
        return tx

    def addModelUpdate(self, model_cid, training_round):
        cid_bytes = self._to_bytes32(model_cid)
        self._contract.functions.addModelUpdate(
            cid_bytes, training_round).call({"from":self.address})
        tx = self._contract.functions.addModelUpdate(
            cid_bytes, training_round).transact()
        self.txs.append(tx)
        return tx

    def skipRound(self, training_round):
        tx = self._contract.functions.skipRound(training_round).transact()
        self.txs.append(tx)
        return tx

    def waitTrainers (self, training_round):
        train_flag = self._contract.functions.waitTrainers(training_round).call({"from":self.address})
        return train_flag
    
    def getAccountfromUpdate(self, model_cid):
        cid_bytes = self._to_bytes32(model_cid)
        account = self._contract.functions.getAccountfromUpdate(cid_bytes).call({"from":self.address})
        return account 

    def setTokens(self, model_cid, num_tokens):
        cid_bytes = self._to_bytes32(model_cid)
        self._contract.functions.setTokens(cid_bytes, num_tokens).call({"from":self.address})
        tx = self._contract.functions.setTokens(
            cid_bytes, num_tokens).transact()
        self.txs.append(tx)
        return tx

    def getmaxNum (self):
        return self._contract.functions.getmaxNum().call({"from":self.address})


class ConsortiumContractClient(_BaseContractClient):
    """
    Wrapper over the Consortium.sol ABI, to gracefully bridge Python data to Solidity.

    The API of this class should match that of the smart contract.
    """

    def __init__(self, account_idx, address, deploy):
        super().__init__(
            os.path.realpath(os.path.dirname(__file__))[0:-3]+"build/contracts/Consortium.json",
            account_idx,
            address,
            deploy
        )

    def main(self):
        return self._contract.functions.main().call({"from":self.address})

    def auxiliaries(self):
        return self._contract.functions.auxiliaries().call({"from":self.address})

    def latestRound(self):
        return self._contract.functions.latestRound().call({"from":self.address})

    def countTokens(self, address=None, training_round=None):
        if address is None:
            address = self.address
        if training_round is None:
            training_round = self.latestRound()
        return self._contract.functions.countTokens(address, training_round).call({"from":self.address})

    def countTotalTokens(self, training_round=None):
        if training_round is None:
            training_round = self.latestRound()
        return self._contract.functions.countTotalTokens(training_round).call({"from":self.address})

    def setGenesis(self, model_cid, round_duration, num_trainers,accounts):
        cid_bytes = self._to_bytes32(model_cid)
        self._contract.functions.setGenesis(
            cid_bytes, round_duration, num_trainers,accounts).call({"from":self.address})
        tx = self._contract.functions.setGenesis(cid_bytes, round_duration, num_trainers,accounts).transact()
        self.txs.append(tx)
        return tx
    
    def setConsCurTrainer(self, training_round, account_address, contract_address):
        # print(type(account_address))
        print(contract_address)
        tx = self._contract.functions.setConsCurTrainer(account_address, training_round, contract_address).transact()
        self.txs.append(tx)
        return tx

    # def setConsEvaluator (self):
    #     tx = self._contract.functions.setConsEvaluator(self.address,self.contract_address)
    #     self.txs.append(tx)
    #     return tx

    def addAux(self, evaluator, accounts):
        self._contract.functions.addAux(evaluator,accounts).call({"from":self.address})
        tx = self._contract.functions.addAux(evaluator,accounts).transact()
        self.txs.append(tx)
        return tx