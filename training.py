import nn
import agent
import params as p

'''
Manages the main training loop.
Instantiates the network on creation
Self-plays the network against itself, 
keeps the experience replay and then
updates the network with the new info
creates checkpoints in training and 
evaluates different versions of the network 
to ensure that progress is being made
'''
class Training():
    def __init__(self, board_size, hist_size=8):
        self.main_network = nn.Network(board_size, hist_size,
                                       p.residual_filters, p.residual_blocks,
                                       p.policy_filters, p.value_filters,
                                       p.value_hidden)
