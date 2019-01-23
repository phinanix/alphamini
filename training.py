import go
import nn
import agent
import params as p
import experience_replay as exp_rp

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

        self.experience_replay = exp_rp.ExperienceReplay(self.board_size,
                                                         p.hist_size,
                                                         p.replay_length)
                                                        
        self.best_agent = agent.Agent(self.main_network)

    
    #TODO:dynamic temperature
    def play(self, agent_0, agent_1, temp,
             retain_tree=False, playouts=10, save=False, exp_replay=None):
        game = go.GoGame(self.board_size, p.hist_size, p.komi)
        replay = exp_rp.GameReplay(self.board_size, p.hist_size)
        while not game.is_over():
            print("Turn:", game.turn)
            print("Board:\n", game.get_board_str())
            if game.cur_player==0:
               agent = agent_0
            else:
               agent = agent_1
            x,y = agent.move(game, temp,
                             retain_tree=retain_tree, playouts=playouts,
                             save=True, replay=replay)
            game.move(x,y, error=True)
            
        replay.transfer(exp_replay, game.result())
            
    def self_play(self, num_games, temp, save=True):
        for _ in range(num_games):
            self.play(self.best_agent, self.best_agent, temp,
                      save=save, exp_replay=self.experience_replay)
            
    #TODO: decide whether to implement tournament
    def self_train(self, num_positions=1024, batch_size=32):
        data = self.experience_replay.select(num_positions)
        self.main_network.update(data, batch_size=batch_size)
