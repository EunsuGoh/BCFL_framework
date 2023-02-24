import io

import ipfshttpclient
import torch
import os
from test_utils.my import Mymodel

import numpy as np

_cached_models = {}
_ipfs_api = "/ip4/127.0.0.1/tcp/5001"

def get_model(model_cid):
    model = Mymodel()
    if model_cid in _cached_models:
        # make a deep copy from cache
        model.load_state_dict(
            _cached_models[model_cid])
    else:
        # download from IPFS
        with ipfshttpclient.connect(_ipfs_api) as ipfs:
            model_bytes = ipfs.cat(model_cid)
        buffer = io.BytesIO(model_bytes)
        model.load_state_dict(torch.load(buffer))
        _cached_models[model_cid] = model.state_dict()
    return model

def compare_model_weights(model1, model2):
    """
    Compares the weights of two PyTorch models by comparing their state dictionaries.

    Args:
        model1 (torch.nn.Module): First model to compare.
        model2 (torch.nn.Module): Second model to compare.

    Returns:
        bool: True if the models have the same weights, False otherwise.
    """
    # Get the state dictionaries of both models
    state_dict_1 = model1.state_dict()
    state_dict_2 = model2.state_dict()

    key_list = []

    # Check if the keys in the state dictionaries are the same
    if set(state_dict_1.keys()) != set(state_dict_2.keys()):
        return False

    # Check if the values in the state dictionaries are the same
    for key in state_dict_1:
        if torch.equal(state_dict_1[key], state_dict_2[key]) == False:
            key_list.append(key)
            # print(state_dict_1[key],state_dict_2[key])
            # return False

    # If we reach this point, then the models have the same weights
    # return True
    return key_list



buffer = io.BytesIO()
buffer.seek(0)

model1 = get_model("Qmeqnty4mFRE6XZtbdWxRaw671xtXmSixPRaEXchhFHYdk")
model2 = get_model("QmRJ9ngx9iPbGa3UHiionFsBmSGBx2SRu1ywx1LSqPYSG6")
# state_dict_1 = torch.load("/media/hdd1/es_workspace/BCFL_framework_es/crowdsource/trainer2_model.pth")
# state_dict_2 = torch.load("/media/hdd1/es_workspace/BCFL_framework_es/crowdsource/_trainer2_model.pth")

state_dict_1 = model1.state_dict()
state_dict_2 = model2.state_dict()

state_dict_1_weight = state_dict_1['fc1.weight']
state_dict_2_weight = state_dict_2['fc1.weight']

for key in state_dict_1.keys():
    weight1 = state_dict_1[key]
    weight2 = state_dict_2[key]
    for a,b in zip(weight1,weight2):
        if not torch.eq(a,b).all():
            print(a-b)
            
            # print(A)


#     np.all((state_dict_1_weight - state_dict_2_weight).numpy() == 0)
for index, value in enumerate(state_dict_1_weight):
    print(index)
    if torch.equal(value,state_dict_2_weight[index]) == False:
        print(value)
        print(state_dict_2_weight[index]) 
    else:
        print(True)
    # print(index)
    # print(value)
    # print(state_dict_1_weight[index]) 

# print(state_dict_1_weight,state_dict_2_weight)

print(compare_model_weights(model1,model2))

