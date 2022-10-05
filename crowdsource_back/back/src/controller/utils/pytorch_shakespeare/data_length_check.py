import json

train1_root = "/home/dy/2cp_new/crowdsource_back/back/src/utils/pytorch_shakespeare/data/user_data/trainer1_data.json"
train2_root = "/home/dy/2cp_new/crowdsource_back/back/src/utils/pytorch_shakespeare/data/user_data/trainer2_data.json"
train3_root = "/home/dy/2cp_new/crowdsource_back/back/src/utils/pytorch_shakespeare/data/user_data/trainer3_data.json"
train4_root = "/home/dy/2cp_new/crowdsource_back/back/src/utils/pytorch_shakespeare/data/user_data/trainer4_data.json"


with open (train1_root,'r') as f :
    data = json.load(f)
    print(len(data["x"]))

with open (train2_root,'r') as f :
    data = json.load(f)
    print(len(data["x"]))

with open (train3_root,'r') as f :
    data = json.load(f)
    print(len(data["x"]))

with open (train4_root,'r') as f :
    data = json.load(f)
    print(len(data["x"]))