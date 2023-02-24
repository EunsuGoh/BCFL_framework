#!/usr/bin/env python
import json
from pprint import pprint
import sys

addresses = []
pubkeys = []
privkeys = []

file = "./ganache-accounts.json"
# print(file)s
json_data=open(file).read()
data = json.loads(json_data)

for a in data['addresses']:
    pubkeyArray=data['addresses'][a]['publicKey']['data']
    pubkey=""
    for pbk in pubkeyArray:
        pubkey = "%s%02x" % (pubkey,pbk)
    # print("Address:%s" %(a))
    addresses.append(a)
    # print("Public Key:0x%s" % (pubkey) )
    pubkeys.append("0x"+pubkey)
    # print("Private Key:0x%s" % (data['private_keys'][a]))
    privkeys.append(data['private_keys'][a])

accounts = {}

for address,pubkey,privkey in zip(addresses,pubkeys,privkeys):
    # print(address)
    # print(pubkey)
    # print(privkey)

    accounts[address] = {"pubkey":pubkey,"privkey":privkey}

with open("./accounts.json","w") as f:
    json_data = accounts
    json.dump(json_data,f)
# print(accounts)