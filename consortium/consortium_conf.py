# config=\
#   {
#     'TRAINING_ITERATIONS':3,
#     'TRAINING_HYPERPARAMS':{
#     'final_round_num': 3,
#     'batch_size': 64,
#     'epochs': 2,
#     'learning_rate': 0.3,
#     },
#     'NUMBER_OF_TRAINERS':3,
#     #  Available evaluation methods : 'loo','shapley'
#     'EVAL_METHOD':'shapley',
#     'ROUND_DURATION':30000, # seconds for a round
#     # Available client selection methods : 'all', 'random', 'fcfs', "score_order"
#     'SELECTION_METHOD':'score_order'
#   }

config=\
  {
    'TRAINING_ITERATIONS':5,
    'TRAINING_HYPERPARAMS':{
    'final_round_num': 5, # sould be same with 'TRAINING_ITERATIONS'
    'batch_size': 16,
    'epochs': 2,
    'learning_rate': 0.001,
    },
    'NUMBER_OF_TRAINERS':3,
    #  Available evaluation methods : 'loo','shapley'
    'EVAL_METHOD':'shapley',
    'ROUND_DURATION':200, # seconds
    # Available client selection methods : 'all', 'random', 'fcfs', "score_order"
    'SELECTION_METHOD':'score_order'
  }
