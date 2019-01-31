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
        #TODO: fix log div-by-0 (hopefully fixed)
        policy = tree.policy()
        
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
    def __init__(self, board_size, game_hist_size, exp_replay_size,
                 checkpoint_filename=None):
        self.size = exp_replay_size
        self.board_size = board_size
        
        if checkpoint_filename:
            npzfile=np.load(checkpoint_filename)
            self.inputs = npzfile['inputs']
            self.policies = npzfile['policies']
            self.game_results = npzfile['game_results']
            self.search_values = npzfile['search_values']
            self.pointer = npzfile['pointer']
        else:
            self.inputs = np.empty( (exp_replay_size,
                            board_size, board_size, (game_hist_size*2)+1) )
            self.policies = np.empty( (exp_replay_size, (board_size**2)+1) )
            self.game_results = np.empty((exp_replay_size))
            self.search_values = np.empty((exp_replay_size))
            self.pointer = np.zeros(1, dtype=np.int32)
        self.full = False


    def permute_policy(self, policy):
        pass_prob = policy[0]
        board_probs = policy[1:].reshape( (self.board_size, self.board_size) )
        board_perms = self.permute_board(board_probs)
        out = [np.empty( (self.board_size**2+1) ) for _ in board_perms]
        for arr,board in zip(out, board_perms):
            arr[0] = pass_prob
            arr[1:] = board.reshape(self.board_size**2)
        #print("Policy Perms:\n" + str(out))
        return out
    
    ''' Takes in a given board (that may be 2D or 3D) and returns the 8 
    dihedral permutations of that board in a list
    Note: Axis 0,1 must be the x,y dimensions of the board
    Axis 2 is the "depth" and is not changed.
    '''
    def permute_board(self, board):
        out = [board]
        cur = board
        #get all rotations of the board
        for _ in range(3):
            #print("size", board.shape, "board:")
            #print(board)
            cur = np.rot90(cur)
            out.append(cur)
        #flip board vertically
        cur = np.flipud(cur)
        out.append(cur)
        for _ in range(3):
            cur = np.rot90(cur)
            out.append(cur)
        return out

    def save(self, board_input, policy_input, game_result, search_result):
        boards = self.permute_board(board_input)
        policies = self.permute_policy(policy_input)
        for board,policy in zip(boards, policies):
            self.save_one(board,policy,game_result,search_result)
            
    def save_one(self, board_input, policy, game_result, search_result):
        self.inputs[self.pointer[0]] = board_input
        self.policies[self.pointer[0]] = policy
        self.game_results[self.pointer[0]] = game_result
        self.search_values[self.pointer[0]] = search_result
        #update pointer, with wraparound
        self.pointer[0] = (self.pointer[0]+1)%self.size
        if self.pointer[0]==0:
            #we've wrapped around
            self.full=True
    def maximum_index(self):
        return self.exp_replay_size if self.full else self.pointer[0]

    #TODO: configure averaging z and w
    def select(self, num_values):
        maximum_index = self.maximum_index()
        print("maximum_index:",maximum_index,"num_values", num_values)
        if maximum_index > num_values*1.1:
            #we can pick randomly
            index_set = set()
            while len(index_set)<num_values:
                index_set.add(random.randrange(maximum_index))
            indices = np.array(list(index_set), dtype=np.int32)
        else:
            #otherwise just pick everything
            indices = np.arange(self.pointer[0])

        input_subset = self.inputs[indices, :, :, :]
        policies_subset = self.policies[indices, :]
        game_results_subset = self.game_results[indices]
        search_values_subset = self.search_values[indices]
        #print("game results:", game_results_subset)
        #print("search_values:", search_values_subset)
        values_subset = (game_results_subset+search_values_subset)*0.5

        return input_subset, policies_subset, values_subset

    def select_one(self, index):
        input_board = self.inputs[index, :, :, :]
        policy = self.policies[index, :]
        game_result = self.game_results[index]
        search_value = self.search_values[index]
        return input_board, policy, game_result, search_value
        
    def checkpoint(self, filename):
        np.savez(filename, inputs=self.inputs, policies=self.policies,
                 game_results=self.game_results, pointer=self.pointer,
                 search_values=self.search_values)

    def merge(self, other_ER):
        for i in range(other_ER.maximum_index()):
            self.save(*other_ER.select_one(i))
