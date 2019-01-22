import go
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
    def __init__(self, board_size):
        self.board_size = board_size
        self.main_network = nn.Network(board_size, p.hist_size,
                                       p.residual_filters, p.residual_blocks,
                                       p.policy_filters, p.value_filters,
                                       p.value_hidden)
        #TODO: figure out how to implement experience replay
        #probably needs it's own object
        self.experience_replay = None
        self.best_agent = agent.Agent(self.main_network)

    #TODO:properly handle game end
    #TODO:dynamic temperature
    def play(self, agent_0, agent_1, temp, retain_tree=True, playouts=100):
        game = go.GoGame(self.board_size, p.hist_size, p.komi)
        while not game.is_over():
            if game.cur_player==0:
                x,y = agent_0.move(game.state, temp, retain_tree=retain_tree,
                                    playouts=playouts)
            else:
                x,y = agent_1.move(game.state, temp, retain_tree=retain_tree,
                                    playouts=playouts)
            game.move(x,y)
            
    def self_play(self, num_games):
        for _ in num_games:
            self.play(self.best_agent, self.best_agent)
