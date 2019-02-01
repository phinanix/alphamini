import argparse

import params as p
import training

parser = argparse.ArgumentParser(description='self-play games given a network')
parser.add_argument('num_games', type=int)
parser.add_argument('network_file')
parser.add_argument('ER_save_file')
args = parser.parse_args()

t = training.Training(p.board_size, network_filename=args.network_file)
t.self_play(args.num_games, p.temp, args.ER_save_file,
            save=True, playouts=p.playouts)
