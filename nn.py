import numpy as np
#import keras
from keras.models import Model
from keras.layers import (Input, Conv2D, BatchNormalization, Dense,
                          Activation,Add,Flatten)
from keras.activations import relu
import go

def Conv_Norm(filters, inputs):
    return BatchNormalization()(Conv2D(filters, (3,3), padding='same')(inputs))

def Conv_Norm_relu(filters, inputs):
    return Activation('relu')(Conv_Norm(filters, inputs))

def Res_Block(filters, inputs):
    intermediate = Conv_Norm_relu(filters, Conv_Norm_relu(filters, inputs))
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
        intermediate = Conv_Norm_relu(residual_filters, inputs)
        for _ in range(residual_blocks):
            intermediate = Res_Block(residual_filters, intermediate)
        policy = Conv_Norm_relu(policy_filters, intermediate)
        policy = Flatten()(policy)
        policy_out = Dense(board_size**2+1)(policy)
        value = Conv_Norm_relu(value_filters, intermediate)
        value = Flatten()(value)
        value = Dense(value_hidden, activation="relu")(value)
        value_out = Dense(1, activation="tanh")(value)
        self.model = Model(inputs=inputs, outputs=[policy_out, value_out])
        self.model.compile(optimizer="sgd",
                           loss=["binary_crossentropy","mean_squared_error"])
        #self.model.summary()
                           
        
    '''takes as input a board of sizexsizex(2*hist_size)
    returns a tuple p,v with p an array of sizexsize that contains the logarithms
    of probabilities of selecting moves
    NOTE: have to turn training data into log probabilities !
    v a scalar in (-1, 1) representing the likelihood of winning from the 
    current player's persepctive
    '''
    def evaluate(self, board, player):
        batch_size = 1
        player_plane = np.full( (self.board_size,self.board_size), player)
        input_block = np.empty( (batch_size, self.board_size,self.board_size,
                                 (self.hist_size*2)+1) )
        input_block[:,:,:, :-1] = board
        input_block[:,:,:, -1 ] = player_plane

        policy_out,value_out = self.model.predict(input_block, verbose=0)

        return policy_out, value_out

    '''
    Takes a 3 tuple of np arrays, which contain
    game boards, policy targets, value targets,
    in that order, and trains for one epoch on them
    '''
    def update(self, data, batch_size=32):
        inputs, policies, values = data
        self.model.fit(inputs, [policies, values], batch_size=batch_size)
