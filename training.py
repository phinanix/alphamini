import multiprocessing as mp
from itertools import repeat

import go
import nn
import agent
import params as p
import experience_replay as exp_rp

import os
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['GOTO_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['openmp'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf



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
 
def parallel_play(board_size, agent_0, agent_1, temp,
         retain_tree=False, playouts=100, save=False, exp_replay=None):
    game = go.GoGame(board_size, p.hist_size, p.komi)
    replay = exp_rp.GameReplay(board_size, p.hist_size)
    while not game.is_over():
        #print("Turn:", game.turn)
        #print("Board:\n" + game.get_board_str())
        if game.cur_player==0:
           agent = agent_0
        else:
           agent = agent_1
        x,y = agent.move(game, temp,
                         retain_tree=retain_tree, playouts=playouts,
                         save=True, replay=replay)
        result = game.unsafe_move(x,y, error=True)
    replay.transfer(exp_replay, game.result())

def parallel_play_many(num_games, temp, board_size, agent,
                       save, exp_rp, playouts=100):
    import os
    print('hello from process id:', os.getpid())
    for _ in range(num_games):
        print("game:", _)
        parallel_play(board_size, agent, agent, temp,
                      save=save, exp_replay=exp_rp,
                      playouts=playouts)
    return exp_rp


class Training():
    def __init__(self, board_size, network_filename=None, exp_rp_filename=None):
        self.board_size = board_size
        self.main_network = nn.Network(board_size, p.hist_size,
                                       p.residual_filters, p.residual_blocks,
                                       p.policy_filters, p.value_filters,
                                       p.value_hidden)
        if network_filename:
            self.main_network.model.load_weights(network_filename)
        self.training_cycles = 0
    
        self.experience_replay = exp_rp.ExperienceReplay(
            self.board_size, p.hist_size, p.replay_length,
            checkpoint_filename=exp_rp_filename)
        
        self.self_play_cycles = 0
        
        self.best_agent = agent.Agent(self.main_network)

    
    #TODO:dynamic temperature
    def play(self, agent_0, agent_1, temp,
             retain_tree=False, playouts=100, save=False, exp_replay=None):
        game = go.GoGame(self.board_size, p.hist_size, p.komi)
        replay = exp_rp.GameReplay(self.board_size, p.hist_size)
        while not game.is_over():
            #print("Turn:", game.turn)
            #print("Board:\n" + game.get_board_str())
            if game.cur_player==0:
               agent = agent_0
            else:
               agent = agent_1
            x,y = agent.move(game, temp,
                             retain_tree=retain_tree, playouts=playouts,
                             save=True, replay=replay)
            result = game.unsafe_move(x,y, error=False)
        replay.transfer(exp_replay, game.result())
            
    def self_play(self, num_games, temp, filename, save=True, playouts=100):
        for _ in range(num_games):
            #print("game:", _)
            self.play(self.best_agent, self.best_agent, temp,
                      save=save, exp_replay=self.experience_replay,
                      playouts=playouts)
            
        self.self_play_cycles += 1
        if self.self_play_cycles%p.save_replay_every == 0 or save:
            print("saving to:", filename)
            self.experience_replay.checkpoint(filename)

    def parallel_self_play(self, num_games, num_processes, temp, filename,
                           save=True, playouts=100):
        games_per = num_games // num_processes
        pool = mp.Pool(num_processes)
        exp_rps = [exp_rp.ExperienceReplay(self.board_size, p.hist_size,
                                           p.replay_length//10)
                   for _ in range(num_processes)]

        #self.best_agent.network.model._make_predict_function() 
        args = zip(repeat(games_per), repeat(temp), repeat(self.board_size),
                   repeat(self.best_agent), repeat(save), exp_rps,
                   repeat(playouts))
        print("the before time")

        ers = pool.starmap(parallel_play_many,  args)
        print('the after time')
        print('len ers:', len(ers))
        print(er.maximum_index() for er in ers)
        for replay in ers:
            self.experience_replay.merge(replay)

        self.self_play_cycles += 1
        if self.self_play_cycles%p.save_replay_every == 0:
            self.experience_replay.checkpoint(filename)
            
    #TODO: decide whether to implement tournament
    def self_train(self, filename, num_positions=1024, batch_size=32,save=True,
                   logfile='train_log.csv'):
        data = self.experience_replay.select(num_positions)
        self.main_network.update(data, batch_size=batch_size, verbose=1,
                                 logfile=logfile)
        self.training_cycles += 1
        if self.training_cycles % p.save_network_every == 0 or save:
            self.main_network.checkpoint(filename)
            
    def training_loop(self, stub_exp_rp_name, stub_network_name, train_log,
                      rounds=3, games_per_round=100, positions_per_round=1024):
        for r in range(rounds):
            print("round:", r)
            self.self_play(games_per_round, p.temp,
                           stub_exp_rp_name+"_round_"+str(r),
                           playouts=p.playouts)
            self.self_train(stub_network_name+"_round_"+str(r),
                            num_positions=positions_per_round,
                            logfile=train_log)
        return self.experience_replay.maximum_index()            

    def parallel_loop(self, stub_exp_rp_name, stub_network_name, train_log,
                      rounds=3, games_per_round=100, positions_per_round=1024,
                      processes=2):
        for r in range(rounds):
            print("round:", r)
            self.parallel_self_play(games_per_round, processes, p.temp,
                                    stub_exp_rp_name+"_round_"+str(r),
                                    playouts=p.playouts)
            self.self_train(stub_network_name+"_round_"+str(r),
                            num_positions=positions_per_round,
                            logfile=train_log)
        return self.experience_replay.maximum_index()
