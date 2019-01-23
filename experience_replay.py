import random

import numpy as np

'''Saves the search results of a single game
in order to be later saved into the experience replay
'''
class GameReplay():
    '''sets up 4 lists to store, in correlated order:
    inputs: board states
    policies: search tree results
    values: search tree results
    turns: 0 for blacks turn, 1 for whites turn
    '''
    def __init__(self, board_size, game_hist_size):
        self.board_size = board_size
        self.hist_size = game_hist_size
        
        self.inputs = []
        self.policies = []
        self.values = []
        self.turns = []
        
    '''saves the result of the root of the given tree into the object
    currently saves policies in a log format
    '''
    def save(self, tree):
        turn = tree.state.cur_player
        
        player_plane = np.full( (self.board_size, self.board_size), turn )
        input_board = np.empty( (self.board_size, self.board_size,
                                 (self.hist_size*2)+1) )
        input_board[:,:,:-1] = tree.state.board
        input_board[:,:,-1] = player_plane

        tree_value = tree.value()
        #TODO: fix log div-by-0
        policy = np.log(tree.policy())
        
        self.inputs.append(input_board)
        self.policies.append(policy)
        self.values.append(tree_value)
        self.turns.append(turn)

    '''called once at the end of a game
    game_result is 1 for a black win and -1 for a white win
    transfers all stored data from the game into exp_replay 
    '''
    def transfer(self, exp_replay, game_result):
        opposite_game_result = 1 if game_result==-1 else -1
        for board_in, policy, value, turn in zip(self.inputs, self.policies,
                                                 self.values, self.turns):
            result = opposite_game_result if turn else game_result
            exp_replay.save(board_in, policy, result, value)
        
class ExperienceReplay():
    '''sets up 4 numpy arrays to store, in correlated order:
    inputs: input boards (board_size x board_size x game_hist_size*2+1)
    policies: search policy targets (board_size**2 + 1)
    game_results: actual winner of game (-1 or 1, scalar)
    search_values: predicted winner of game (value in [-1,1], scalar)

    As well as some internal state:
    pointer: an integer in [0,exp_hist_size) that marks the next place to 
    store incoming data
    full: a bool that tracks whether the experience replay has been filled,
    so all values are valid, or whether every value at and beyond pointer is 
    garbage
    '''
    def __init__(self, board_size, game_hist_size, exp_replay_size):
        self.size = exp_replay_size
        
        self.inputs = np.empty( (exp_replay_size,
                            board_size, board_size, (game_hist_size*2)+1) )
        self.policies = np.empty( (exp_replay_size, (board_size**2)+1) )
        self.game_results = np.empty((exp_replay_size))
        self.search_values = np.empty((exp_replay_size))

        self.pointer = 0
        self.full = False

    def save(self, board_input, policy, game_result, search_result):
        self.inputs[self.pointer] = board_input
        self.policies[self.pointer] = policy
        self.game_results[self.pointer] = game_result
        self.search_values[self.pointer] = search_result
        #update pointer, with wraparound
        self.pointer = (self.pointer+1)%self.size
        if self.pointer==0:
            #we've wrapped around
            self.full=True

    #TODO: configure averaging z and w
    def select(self, num_values):
        maximum_index = self.exp_replay_size if self.full else self.pointer
        index_set = set()
        print("maximum_index:",maximum_index,"num_values", num_values)
        assert maximum_index > num_values*1.1, \
            "must have enough values to fill the set"
        while len(index_set)<num_values:
            index_set.add(random.randrange(maximum_index))
            
        indices = np.array(list(index_set), dtype=np.int32)
        input_subset = self.inputs[indices, :, :, :]
        policies_subset = self.policies[indices, :]
        game_results_subset = self.game_results[indices]
        search_values_subset = self.search_values[indices]
        values_subset = (game_results_subset+search_values_subset)*0.5

        return input_subset, policies_subset, values_subset
