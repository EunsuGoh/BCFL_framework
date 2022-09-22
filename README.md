# BCFL_framework_es

# FILE STRUCTURE
`2cp/`: 2cp 프로토콜 크라우드소스, 컨소시움 세팅관련 파일 폴더

`contracts/`: 스마트 컨트랙트 폴더

`migrations/`: 트러플 스마트 컨트랙트 배포용 폴더

`crowdsource_back/`: 크라우드소스 시나리오 테스트용 백엔드
    - `pytorch_cifar10/` : iid cifar-10 데이터 / 모델 포함
    - `pytorch_mnist/` : iid mnist 데이터 / 모델 포함
    - `pytorch_femnist/` : non-iid FeMnist 데이터 / 모델 포함
    - `pytorch_shakespeare/` : non-iid shakespeare 데이터 / 모델 포함

# ANACONDA ENV
- 아나콘다 가상환경 추천
- 가상환경 생성 및 활성화 후
    1. conda create -m {your_env}
    2. conda activate {your_env}
- requirement 다운로드 (** 주의, 파이토치-토치비전 본인 환경에 맞는 버전으로 받을 것)
    1. pip install -r requirement.txt

# DATA
- cifar-10, mnist : torchvision 데이터셋 사용 (preprocess + data_load)
- femnist, shakespeare : leaf 데이터셋 사용 (data_load)
    femnist : ./preprocess.sh -s niid --iu 4 --sf 0.2  -k 0 -t user -tf 0.8
    shakespeare : ./preprocess.sh -s niid --iu 4 --sf 0.4  -k 0 -t user -tf 0.8

# DATA PREPROCESSING
- cifar-10, mnist
    step 1. python data_preprocess.py
- shakespeare, femnist 
    step 1. download leaf dataset (DATA 파트 참고), https://github.com/TalwalkarLab/leaf
    step 2. 주석 해제하고 데이터 저장(python data_load.py)

# dependencies (need to download)
- ganache
- ipfs daemon
- truffle

# run teset crowdsource code (단일 스크립트 실행)
1. ganache-cli --port=7545
2. ipfs daemon
3. truffle migrate --network development --reset
4. cd crowdsource_back/back/src/utils/pytorch_{테스트 하고자하는 데이터}
5. 데이터 전처리를 안했다면, DATA PREPROCESSING 참고하여 전처리부터 진행
6. python pytorch_{테스트 하고자하는 데이터}_eval.py
    **스크립트 내 max_num_updates 변수를 실행하고자 하는 클라이언트 수 만큼 업데이트 후 실행 요망
7. 새로운 터미널
8. python pytorch_{테스트 하고자하는 데이터}_train.py {트레이너 인덱스}
    i.e.) python pytorch_femnist_train.py 1 
        (주의 : 트레이너 인덱스 1부터 시작)

