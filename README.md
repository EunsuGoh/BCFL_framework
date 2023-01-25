# BCFL_framework_es

# FILE STRUCTURE
`2cp/`: 2cp crowdsource, consortium protocol's client utils

`contracts/`: Ethereum smart contracts(crowdsource, consosrtium)

`migrations/`: Use for smart contract deployment using truffle

`crowdsource` : Includes scenario code for Crowdsource protocol

`consortium` : Includes scenario code for Consortium protocol

`crowdsource_back/`: Node-js backend server for crowdsource web demo
    - `pytorch_mnist/` : Includes iid Mnist dataset and model
    - `pytorch_femnist/` : Includes non-iid Femnist dataset and model
    - `pytorch_shakespeare/` : Includes non-iid Shakespeare dataset and model

# Add 2cp TO PYTHONPATH
- conda develop 2cp

# ANACONDA ENV
- We recommend ANACONDA virtual env
- After making anaconda virtual env..
    1. conda create -m {your_env}
    2. conda activate {your_env}
- Download requirement(** Warning, you should download pytorch and torchvision matching with your machine's cuda version)
    1. pip install -r requirement.txt

## DATA
- you should save your data below consortium/data/user_data or crowdsource/data/user_data
- ex) consortium/data/user_data/trainer1_data.json
- ex)  crowdsource/data/user_data/evaluator_data.json

## EXPERIMENT DATA
- mnist : torchvision dataset (preprocess + data_load)
- femnist, shakespeare : leaf dataset (data_load)
    femnist : ./preprocess.sh -s niid --iu 4 --sf 0.2  -k 0 -t user -tf 0.8
    shakespeare : ./preprocess.sh -s niid --iu 4 --sf 0.2 -k 0 -t user -tf 0.8

## DATA PREPROCESSING
- mnist
    step 1. python data_preprocess.py
- shakespeare, femnist 
    step 1. download leaf dataset, https://github.com/TalwalkarLab/leaf
    step 2. remove annotation and run the script(python data_load.py)

# dependencies (need to download)
- ganache
- ipfs daemon
- truffle




# run test consortium code
1. ganache-cli --port=7545 --networkId 5777
2. ipfs daemon
3. truffle migrate --network development --reset
4. cd consortium
5. Edit consortium_conf.py
6. python test_consortium.py

# run test crowdsource code
1. ganache-cli --port=7545 --networkId 5777
2. ipfs daemon
3. truffle migrate --network development --reset
4. cd crowdsource
5. Edit crowdsource_conf.py
6. python test_consortium.py

# run test crowdsource code with front page
1. ganache-cli --port=7545
2. ipfs daemon
3. truffle migrate --network development --reset
4. cd crowdsource_back/back
5. npm run start-eval
6. cd crowdsource_back/back (new terminal open)

## run test crowdsource code (run single python script from backend side)
1. ganache-cli --port=7545
2. ipfs daemon
3. truffle migrate --network development --reset
4. cd crowdsource_back/back/src/utils/pytorch_{DATASET}
5. If you didn't preprocess data, Note DATA PREPROCESSING section
6. python pytorch_{DATASET}_eval.py {TRAINER_NUMBER}
        i.e.) python pytorch_shakespeare_eval.py 4
7. Open new terminal
8. python pytorch_{DATASET}_train.py {TRAINER_INDEX}
    i.e.) python pytorch_femnist_train.py 1 
        (*Warning : TRAINER_INDEX shoul start from 1)


