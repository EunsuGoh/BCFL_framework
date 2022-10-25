
def same_weights(model_a, model_b):
    """
    Checks if two pytorch models have the same weights.
    https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/5
    """
    for params_a, params_b in zip(model_a.parameters(), model_b.parameters()):
        if (params_a.data != params_b.data).sum() > 0:
            return False
    return True