# -*- coding:utf-8 -*-

import sys
import os
import re
import mregn_runner

class my_runner(mregn_runner.MREGNRunner):
    def output_type(self):
        return 0

def main():
    seed = int(sys.argv[1]) if str.isdigit(sys.argv[1]) else 0
    ignore_index = [int(x) for x in sys.argv[2].split(',') if str.isdigit(x)]
    mre_file = sys.argv[3]
    gn_file = sys.argv[4]
    sequence_file = sys.argv[5]

    mregn_runner.init_genrand(seed)
    runner = my_runner()
    runner.init(mre_file, gn_file)
    runner.set_time_series_id()

    p = re.compile(r'(^#)|(^$)')
    out_state_queue = []
    for line in open(sequence_file, 'r'):
        if p.match(line) == None:
            input = map(float, line[:-1].split())
            if len(out_state_queue) >= runner.mre_delay_length():
                out_state = out_state_queue.pop(0)
                for i in ignore_index:
                    input[i] = out_state[i]
            runner.mre_in_state(input)
            runner.update()
            out_state = runner.mre_out_state()
            print '\t'.join([str(x) for x in out_state])
            out_state_queue.append(out_state)

if __name__ == '__main__':
    main()

