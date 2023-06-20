import json
import os
import base58
from web3 import HTTPProvider, Web3

# 1. 이더리움 클라이언트
class BaseEthClient:
    """
    An ethereum client.
    """

    PROVIDER_ADDRESS = "http://127.0.0.1:7545"
    NETWORK_ID = "5777"

    def __init__(self, account_idx):
        self._w3 = Web3(HTTPProvider(self.PROVIDER_ADDRESS)) #json rpc 서버 연결(가나슈)

        self.address = self._w3.eth.accounts[account_idx] # 이더리움 지갑주소
        self._w3.eth.defaultAccount = self.address # 지정된 지갑주소를 기본주소로

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
        # 가나슈에 배포된 컨트랙트의 json 파일을 인스턴스화
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

class NFTContract(_BaseContractClient):
    def __init__(self, account_idx, address, deploy):
        super().__init__(
            os.path.realpath(os.path.dirname(__file__))[0:-3]+"build/contracts/Crowdsource.json",
            account_idx,
            address,
            deploy
        )

    def mint(self, recipient, tokenUri):
        tx = self._contract.functions.mintNFT(recipient, tokenUri).transact()
        self.txs.append(tx)
        return tx