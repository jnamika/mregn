# -*- coding:utf-8 -*-

import sys
import re
import math
import mregn_runner
import rnn_kl_div

class my_runner(mregn_runner.MREGNRunner):
    def output_type(self):
        return 0


def main():
    seed, length, samples, truncate_length, block_length, divide_num = \
            map(lambda x: int(x) if str.isdigit(x) else 0, sys.argv[1:7])
    mre_file = sys.argv[7]
    gn_file = sys.argv[8]
    mregn_runner.init_genrand(seed)
    runner = my_runner()
    runner.init(mre_file, gn_file)
    print rnn_kl_div.get_KL_div(length, samples, truncate_length, block_length,
            divide_num, runner, sys.argv[9:])


if __name__ == '__main__':
    main()

