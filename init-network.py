import argparse

import params as p
import nn

parser = argparse.ArgumentParser(description='create a random network')
parser.add_argument('network_out')
args = parser.parse_args()

#note that the first file in is "special" in that all the other ERs are
#merged into it
network = nn.Network(p.board_size, p.hist_size,
                     p.residual_filters, p.residual_blocks,
                     p.policy_filters, p.value_filters, p.value_hidden)
network.checkpoint(args.network_out)
