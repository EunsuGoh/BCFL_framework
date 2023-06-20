import threading

import torch
import torch.nn.functional as F

from BCFL.clients import CrowdsourceClient
from BCFL.utils_2cp import print_token_count, check_balance

from test_utils.my import MyData, Mymodel
from test_utils.functions import same_weights
from crowdsource_conf import config
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
EVAL_METHOD = config['EVAL_METHOD']
TORCH_SEED = 8888
ROUND_DURATION = config['ROUND_DURATION'] 
SELECTION_METHOD = config['SELECTION_METHOD']
NODE_PATH = os.environ["NODE_PATH"]
EVALUATOR_PRIVATE_KEY = os.environ["METAMASK_EVALUATOR_PRIVATE_KEY"]
TARINER_PRIVATE_KEY = os.environ["METAMASK_TRAINER_PRIVATE_KEY"]
BCFL_CONTRACT_PATH = os.path.realpath(os.path.dirname(__file__))[0:-16]+"build/contracts/Crowdsource.json"
TOKEN_CONTRACT_PATH = os.path.realpath(os.path.dirname(__file__))[0:-16]+"build/contracts/Token.json"
NETWORK_ID = os.environ["NETWORK_ID"]

torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
np.random.seed(TORCH_SEED)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(TORCH_SEED)
os.environ["PYTHONHASHSEED"] = str(TORCH_SEED)

def issueHolderDID (path,holder_public_key,holder_private_key):
    args = [holder_public_key,holder_private_key]
    function_name = "createHolderDID"
    result = subprocess.check_output([NODE_PATH,"-p",f"require('{path}').{function_name}(...{args})"])
    did = result.decode().split("\n")[0]
    return did

def issueIssuerDID (path, issuer_public_key, issuer_private_key):
    args = [issuer_public_key,issuer_private_key]
    function_name = "createIssuerDID"
    result = subprocess.check_output([NODE_PATH,"-p",f"require('{path}').{function_name}(...{args})"])
    issuer = result.decode()
    return issuer

def createVC (path, did, issuerPub,issuerPriv):
    args = [did,issuerPub,issuerPriv]
    function_name = "issueVC"
    result = subprocess.check_output([NODE_PATH,"-p",f"require('{path}').{function_name}(...{args})"])
    return result

def verifyVC(path, vc, issuer_did):
    function_name = "verifyVC"
    vc_str = str(vc).split("::")
    pre_vc = vc_str[1].split("\\")
    vc = pre_vc[0]
    print(vc)
    args = [vc,issuer_did]
    result = subprocess.check_output([NODE_PATH,"-p",f"require('{path}').{function_name}(...{args})"])
    return result

def test_crowdsource ():
  # Define Eval data
  # your testset Path
    testset_path = os.path.realpath(os.path.dirname(__file__))+'/data/user_data/evaluator_data.json'

    with open(testset_path,'r') as f:
        eval_dataset = json.load(f)
        eval_data = eval_dataset['x']
        eval_targets = eval_dataset['y']

    # print(eval_data)
    my_eval_data = MyData(eval_data, eval_targets, True, eval_flag=True)

  # Evaluator client setting
    with open (BCFL_CONTRACT_PATH) as json_file:
        crt_json = json.load(json_file)
        abi = crt_json['abi']
        bytecode = crt_json['bytecode']
        bcfl_contract_address = crt_json['networks'][NETWORK_ID]['address']
    with open (TOKEN_CONTRACT_PATH) as json_file:
        crt_json = json.load(json_file)
        abi = crt_json['abi']
        bytecode = crt_json['bytecode']
        token_contract_address = crt_json['networks'][NETWORK_ID]['address']

    evaluator = CrowdsourceClient(
        "Evaluator", my_eval_data, eval_targets, Mymodel, F.cross_entropy, 0, contract_address=bcfl_contract_address, token_contract_address =token_contract_address)


    trainers = [] # trainer1, trainer2 ...
    train_clients = [] # CrowdsourceClients 
    trainer_did = {} #issued did 

    # # temp issuer (ganache account's last index account)
    # issuer_account = trainer_accounts[0]
    # issuer_info = {}

    #Issueing issuer's did
    # account_list_path = os.path.dirname(os.path.abspath(__file__))
    # did_script_path = account_list_path+"/src/did.js"
    # with open(account_list_path+"/accounts.json","r") as f:
    #     data = json.load(f)
    #     # print(data[trainer.address.lower()])
    #     # pubkey = data[trainer.address.lower()]['pubkey']
    #     pubkey = issuer_account
    #     privkey = data[issuer_account.lower()]['privkey']
    #     issuer_info["pubkey"] = pubkey
    #     issuer_info["privkey"] = privkey
    #     did_script_path = account_list_path+"/src/did.js"
    #     issuer = issueIssuerDID(did_script_path,pubkey,privkey)
    #     issuer_info["did"] = issuer.split("\n")[0]

    # for number in range(config['NUMBER_OF_TRAINERS']):
    #     trainer_index = "trainer"+str(number+1)
    #     trainers.append(trainer_index)

    did_script_path = os.path.dirname(os.path.abspath(__file__))+"/src/did.js"
    
    issuer_info = {}
    issuer_info["pubkey"] = os.environ["METAMASK_EAVLUATOR_ACCOUNT"]
    issuer_info["privkey"] = os.environ["METAMASK_EVALUATOR_PRIVATE_KEY"]
    issuer = issueIssuerDID(did_script_path,issuer_info["pubkey"],issuer_info["privkey"])
    issuer_info["did"] = issuer.split("\n")[0]

    

    # ISSUEING Trainers' DID Process
    vcList = []

    holder_info = {}
    holder_info["pubkey"] = os.environ["METAMASK_TRAINER_ACCOUNT"]
    holder_info["privkey"] = os.environ["METAMASK_TRAINER_PRIVATE_KEY"]
    holder_did = issueHolderDID(did_script_path,holder_info["pubkey"],holder_info["privkey"])
    holder_info["did"] = holder_did

    # Get VC
    vc = createVC(did_script_path,holder_info["did"],issuer_info["pubkey"],issuer_info["privkey"])
    
    #Verify VC
    verifiedVC = verifyVC(did_script_path,vc, issuer_info["did"])
    print(verifiedVC)
    print("\n")


    # n = round(len(trainers)*0.5) # 50% of trainers can issue did
    # vcList = []
    # for index,account in enumerate(trainer_accounts):
    #     if(index <= n):
    #         account_list_path = os.path.dirname(os.path.abspath(__file__))
    #         with open(account_list_path+"/accounts.json","r") as f:
    #             data = json.load(f)
    #             # print(data[trainer.address.lower()])
    #             # pubkey = data[trainer.address.lower()]['pubkey']
    #             pubkey = account
    #             privkey = data[account.lower()]['privkey']
    #             did_script_path = account_list_path+"/src/did.js"
    #             did = issueHolderDID(did_script_path,pubkey,privkey)
    #             trainer_did[account] = did
    #             vc = createVC(did_script_path, did, issuer_info["pubkey"], issuer_info["privkey"]  )

    #             verifiedVC = verifyVC(did_script_path,vc, issuer_info["did"])
    #             print(verifiedVC)
    #             print("\n")



    # Trainer Client Generation
    # for trainer in trainers :
    #     print("client is deploying....")
    #     trainer_index = re.sub(r'[^0-9]','',trainer)
    #     # your trainset Path
    #     trainset_path = os.path.realpath(os.path.dirname(__file__))+'/data/user_data/'+trainer+'_data.json'
    #     did = None
    #     with open(trainset_path,'r') as f:
    #         train_dataset = json.load(f)
    #         train_data = train_dataset['x']
    #         train_targets = train_dataset['y']
    #     if trainer_accounts[int(trainer_index)] in trainer_did:
    #         did = trainer_did[trainer_accounts[int(trainer_index)]]
    #     my_train_data = MyData(train_data, train_targets, True,device_num=str(trainer_index[-1:]), did_address = did)
    #     train_client = CrowdsourceClient(trainer,my_train_data,train_targets,Mymodel, F.cross_entropy,int(trainer_index),contract_address=evaluator.contract_address, device_num= str(trainer_index[-1:]), did_address = did)
    #     train_clients.append(train_client)
    
    print("client is deploying....")
    trainset_path = os.path.realpath(os.path.dirname(__file__))+'/data/user_data/trainer1_data.json'
    if holder_info["did"]:
        did=holder_info["did"]
    with open(trainset_path,'r') as f:
            train_dataset = json.load(f)
            train_data = train_dataset['x']
            train_targets = train_dataset['y']
    my_train_data = MyData(train_data, train_targets, True,device_num="1", did_address = did)
    train_client = CrowdsourceClient("trainer1",my_train_data,train_targets,Mymodel, F.cross_entropy,1,contract_address=evaluator.contract_address, device_num= "1", did_address = did)
    train_clients.append(train_client)

    evaluator.set_genesis_model(
        round_duration=ROUND_DURATION,
        max_num_updates=config['NUMBER_OF_TRAINERS'],
        scenario = "crowdsource"
    )

    # Training
    threads = [
        threading.Thread(
            target=trainer.train_until,
            kwargs=TRAINING_HYPERPARAMS,
            daemon=True
        ) for trainer in train_clients
    ]

    # Evaluation
    threads.append(
        threading.Thread(
            target=evaluator.evaluate_until,
            args=(TRAINING_ITERATIONS, EVAL_METHOD,"crowdsource"),
            kwargs={
                'selection_method':SELECTION_METHOD,
                "did_info":trainer_did,
                "alpha":config['ALPHA']
            },
            daemon=True
        )
    )

    # Run all threads in parallel
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Fake token print
    # for trainer in train_clients:
    #     print_token_count(trainer)
    
    final_global_model = evaluator._get_global_model(TRAINING_ITERATIONS+1)
    # print(final_global_model)
    print("\n\n")
    #erc20 token print
    for trainer in train_clients:
        balance = check_balance(evaluator,trainer.address)
        print(f"\t {trainer.name} 's token balance is {balance}")

test_crowdsource()