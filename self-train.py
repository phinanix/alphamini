import argparse

import params as p
import training

parser = argparse.ArgumentParser(description='train a network')
parser.add_argument('num_positions', type=int)
parser.add_argument('ER_save_file_in')
parser.add_argument('network_file_in')
parser.add_argument('network_file_out')
parser.add_argument('--logfile', default='train_log.csv')
args = parser.parse_args()

t = training.Training(p.board_size, network_filename=args.network_file_in,
                      exp_rp_filename=args.ER_save_file_in)
t.self_train(args.network_file_out, args.num_positions, logfile=args.logfile)
