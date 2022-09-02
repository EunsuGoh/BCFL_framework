import io

import ipfshttpclient
import torch

class IPFSClient:

    # class attribute so that IPFSClient's on the same machine can all benefit
    _cached_models = {}

    def __init__(self, model_constructor):
        self._model_constructor = model_constructor

    def get_model(self, model_cid):
        model = self._model_constructor()
        if model_cid in self._cached_models:
            # make a deep copy from cache
            model.load_state_dict(
                self._cached_models[model_cid])
        else:
            # download from IPFS
            with ipfshttpclient.connect() as ipfs:
                model_bytes = ipfs.cat(model_cid)
            buffer = io.BytesIO(model_bytes)
            model.load_state_dict(torch.load(buffer))
            self._cached_models[model_cid] = model.state_dict()
        return model

    def add_model(self, model):
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        with ipfshttpclient.connect() as ipfs:
            model_cid = ipfs.add_bytes(buffer.read())
        return model_cid

