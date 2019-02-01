import argparse

import params as p
import experience_replay as er

parser = argparse.ArgumentParser(description='combine multiple ERs to one')
parser.add_argument('ER_save_file_out')
parser.add_argument('files_in', nargs='+')
args = parser.parse_args()

#note that the first file in is "special" in that all the other ERs are
#merged into it
out = er.ExperienceReplay(p.board_size, p.hist_size, p.replay_length,
                          checkpoint_filename=args.files_in[0])

for replay_name in args.files_in[1:]:
    print('processing:', replay_name)
    new = er.ExperienceReplay(p.board_size, p.hist_size, p.replay_length,
                              checkpoint_filename=replay_name)
    out.merge(new)
    
out.checkpoint(args.ER_save_file_out)

