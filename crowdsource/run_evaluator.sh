#!/bin/bash

conda develop 2cp 
cd crowdsource && python eval_crowdsource.py \
    --ipfs $IPFS_API\
    --provider $PROVIDER\

