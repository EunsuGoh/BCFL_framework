import random

# random trainers
def random_selection(client_list):
     n = random.randint(len(client_list) // 2, len(client_list))
     round_trainers = random.sample(client_list, n)

     return round_trainers

#first come, first served
def fcfs_selection(client_list):
    n = random.randint(len(client_list) // 2, len(client_list))
    round_trainers = client_list[0:n]
    return round_trainers

#select all trainers
def all_selection(client_list):
    return client_list

#select trainers by score order
def score_order(client_list, client_scores):
    new_client_list = []
    client_info = {}
    # n = math.floor(len(client_list)*0.1)
    n = round(len(client_list)*0.1)
    for i in range(len(client_list)):
        client_info[client_list[i]] = client_scores[i]

    client_info = sorted(client_info.items(), key = lambda item: item[1], reverse = True)
    
    for i in range(len(client_info)):
        if len(new_client_list) < (len(client_list))-n:
            new_client_list.append(client_info[i][0])
        else : 
            break
    return new_client_list
        
    
