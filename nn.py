import numpy as np
#import keras
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Dense, Activation,Add
from keras.activations import relu
import go

def Conv_Norm(filters, inputs):
    return Activation('relu')(BatchNormalization()(Conv2D(filters, (3,3), padding='same')(inputs)))

def Res_Block(filters, inputs):
    intermediate = BatchNormalization()(
        Conv2D(filters, (3,3), padding='same')(Conv_Norm(filters, inputs)))
    return Activation('relu')(Add()([inputs,intermediate]))

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
    def __init__(self, board_size, hist_size,
                 residual_filters, residual_blocks,
                 policy_filters, value_filters, value_hidden):
        #maybe take a number of layers and a network type or something?
        self.board_size=board_size
        self.hist_size = hist_size
        self.network = None

        #intialize model here
        inputs = Input(shape=(board_size,board_size,hist_size*2+1))
        intermediate = Conv_Norm(residual_filters, inputs)
        for _ in range(residual_blocks):
            intermediate = Res_Block(residual_filters, intermediate)
        policy = Conv_Norm(policy_filters, intermediate)
        policy_out = Dense(board_size**2+1)(policy)
        value = Conv_Norm(value_filters, intermediate)
        value = Dense(value_hidden, activation="relu")(value)
        value_out = Dense(1, activation="tanh")(value)
        self.model = Model(inputs=inputs, outputs=[policy_out, value_out])
        self.model.compile(optimizer="sgd",
                           loss=["binary_crossentropy","mean_squared_error"])
        self.model.summary()
                           
        
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
    
