def print_global_performance(client):
    loss = client.evaluate_current_global()
    print(f"\t\t{client.name} got: Loss {loss}")


def print_token_count(client):
    tokens = client.get_token_count()
    total_tokens = client.get_total_token_count()
    percent = int(100*tokens/total_tokens) if tokens > 0 else 0
    print(f"\t\t{client.name} has {tokens} of {total_tokens} tokens ({percent}%)")
