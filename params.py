
#exploration exploitation tradeoff
c_puct = 1

#network parameters
residual_filters = 64
residual_blocks = 5
policy_filters = 8
value_filters = 8
value_hidden = 256

#size of history that's input to the network
hist_size = 8
komi = 4.5

#training parameters
replay_length = 10**4
save_network_every = 1 #training cycles
save_replay_every  = 1 #self play cycles
