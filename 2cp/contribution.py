from scipy.special import comb
from more_itertools import powerset

def shapley_value(c_fn, plrs, plr_i):
    players_excluding_i = set(plrs).difference({plr_i})
    sv = 0
    for coalition in powerset(players_excluding_i):
        coalition_plus_i = set(coalition).union({plr_i})
        marginal_contribution = c_fn(*coalition_plus_i) - c_fn(*coalition)
        sv += marginal_contribution / comb(len(plrs)-1, len(coalition))
    return sv / len(plrs)

def shapley_values(c_fn, players: set):
    sv = dict()
    for player in players:
        sv[player] = shapley_value(c_fn, players, player)
    return sv

def loo(c_fn,players):
    leave_one_out_contribution = dict()
    contribution = 0
    for player in players:
        players_including_i = set(players)
        players_excluding_i = set(players).difference({player})
        marginal_contribution =  c_fn(*players_including_i) - c_fn(*players_excluding_i)
        contribution += marginal_contribution / len(players)
        leave_one_out_contribution[player] = contribution 
        
    return leave_one_out_contribution

# def gtg_shapley(c_fn):
    # GTG Shapley method : 
    # 1. for horizontal learning
    # 2. guided Monte Carlo sampling - within-round, between round truncation
    

# def test( plrs, plr_i):
#     players_excluding_i = set(plrs).difference({plr_i})
#     sv = 0
#     print(len(plrs)-1)
#     for coalition in powerset(players_excluding_i):
#         coalition_plus_i = set(coalition).union({plr_i})
#         print(len(coalition))
#         print(coalition, comb(len(plrs)-1, len(coalition))) 

# list = ['A','B','C']
# test(list,'A')