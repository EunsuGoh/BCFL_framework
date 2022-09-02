from scipy.special import comb
from more_itertools import powerset

def value(c_fn, plrs, plr_i):
    players_excluding_i = set(plrs).difference({plr_i})
    sv = 0
    for coalition in powerset(players_excluding_i):
        coalition_plus_i = set(coalition).union({plr_i})
        marginal_contribution = c_fn(*coalition_plus_i) - c_fn(*coalition)
        sv += marginal_contribution / comb(len(plrs)-1, len(coalition))
    return sv / len(plrs)

def values(c_fn, players: set):
    sv = dict()
    for player in players:
        sv[player] = value(c_fn, players, player)
    return sv

