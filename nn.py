import numpy as np
import keras

import go

class Network():
    '''stores the info needed to perform board evaluation
    and supports updating the contained parameters
    takes in a size x size x 2*hist_size board and a current player
    evaluates from the perspective of that player
    returns a size x size array of policy output that sums to 1
    and a scalar value output in (-1,1)
    '''

    '''initializes network to support inputs of sizexsizex(2*histsize+1)
    and performs any other initialization'''
    def __init__(self, size, hist_size):
        #maybe take a number of layers and a network type or something?
        self.size=size
        self.hist_size = hist_size
        self.network = None

    '''takes as input a board of sizexsizex(2*hist_size)
    returns a tuple p,v with p an array of sizexsize that sums to 1 
    of probabilities of selecting moves
    v a scalar in (-1, 1) representing the likelihood of winning from the 
    current player's persepctive
    '''
    def evaluate(self, board, player):
        player_plane = np.full( (self.size,self.size), player)
        input_block = np.empty( (self.size,self.size,(self.hist_size*2)+1) )
        input_block[:,:, :-1] = board
        input_block[:,:, -1 ] = player_plane

        #evaluate network here

        stub_p = np.full( (self.size, self.size) , 1 / (self.size**2) ) 
        return stub_p, 0
    
