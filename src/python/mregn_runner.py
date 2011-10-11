# -*- coding:utf-8 -*-

import sys
import os
import datetime
from ctypes import *
from ctypes.util import find_library

libpath = find_library('mregnrunner')
librunner = cdll.LoadLibrary(libpath)

librunner.init_genrand.argtype = c_ulong
librunner.gn_in_state_from_runner.restype = POINTER(c_double)
librunner.gn_c_state_from_runner.restype = POINTER(c_double)
librunner.gn_c_inter_state_from_runner.restype = POINTER(c_double)
librunner.gn_out_state_from_runner.restype = POINTER(c_double)
librunner.mre_in_state_from_runner.restype = POINTER(c_double)
librunner.mre_out_state_from_runner.restype = POINTER(c_double)
librunner.expert_rnn_c_state_from_runner.restype = POINTER(c_double)
librunner.expert_rnn_c_inter_state_from_runner.restype = POINTER(c_double)


def init_genrand(seed):
    if seed == 0:
        now = datetime.datetime.utcnow()
        seed = ((now.hour * 3600 + now.minute * 60 + now.second) *
                now.microsecond)
    librunner.init_genrand(c_ulong(seed % 4294967295 + 1))


class MREGNRunner(object):
    def __init__(self, librunner=librunner):
        self.runner = c_void_p()
        self.librunner = librunner
        self.librunner._new_mregn_runner(byref(self.runner))
        self.is_initialized = False

    def __del__(self):
        self.free()
        self.librunner._delete_mregn_runner(self.runner)

    def init(self, mre_file_name, gn_file_name):
        self.free()
        self.librunner.init_mregn_runner_with_filename(self.runner,
                mre_file_name.encode(), gn_file_name.encode())
        self.is_initialized = True

    def free(self):
        if self.is_initialized:
            self.librunner.free_mregn_runner(self.runner)
        self.is_initialized = False

    def set_time_series_id(self, id=0):
        self.librunner.set_init_state_of_mregn_runner(self.runner, id)

    def target_num(self):
        return self.librunner.gn_target_num_from_runner(self.runner)

    def gn_in_state_size(self):
        return self.librunner.gn_in_state_size_from_runner(self.runner)

    def gn_c_state_size(self):
        return self.librunner.gn_c_state_size_from_runner(self.runner)

    def gn_out_state_size(self):
        return self.librunner.gn_out_state_size_from_runner(self.runner)

    def gn_delay_length(self):
        return self.librunner.gn_delay_length_from_runner(self.runner)

    def mre_in_state_size(self):
        return self.librunner.mre_in_state_size_from_runner(self.runner)

    def mre_out_state_size(self):
        return self.librunner.mre_in_state_size_from_runner(self.runner)

    def mre_delay_length(self):
        return self.librunner.mre_delay_length_from_runner(self.runner)

    def expert_rnn_c_state_size(self, index):
        return self.librunner.expert_rnn_c_state_size_from_runner(self.runner,
                c_int(index))

    def update(self):
        self.librunner.update_mregn_runner(self.runner)

    def closed_loop(self, length):
        for n in range(length):
            self.update()
            yield self.mre_out_state(), self.gn_out_state(), \
                    self.gn_c_inter_state()

    def gn_in_state(self, in_state=None):
        x = self.librunner.gn_in_state_from_runner(self.runner)
        if in_state != None:
            for i in range(len(in_state)):
                x[i] = c_double(in_state[i])
        return [x[i] for i in range(self.gn_in_state_size())]

    def gn_c_state(self, c_state=None):
        x = self.librunner.gn_c_state_from_runner(self.runner)
        if c_state != None:
            for i in range(len(c_state)):
                x[i] = c_double(c_state[i])
        return [x[i] for i in range(self.gn_c_state_size())]

    def gn_c_inter_state(self, c_inter_state=None):
        x = self.librunner.gn_c_inter_state_from_runner(self.runner)
        if c_inter_state != None:
            for i in range(len(c_inter_state)):
                x[i] = c_double(c_inter_state[i])
        return [x[i] for i in range(self.gn_c_state_size())]

    def gn_out_state(self, out_state=None):
        x = self.librunner.gn_out_state_from_runner(self.runner)
        if out_state != None:
            for i in range(len(out_state)):
                x[i] = c_double(out_state[i])
        return [x[i] for i in range(self.gn_out_state_size())]

    def mre_in_state(self, in_state=None):
        x = self.librunner.mre_in_state_from_runner(self.runner)
        if in_state != None:
            for i in range(len(in_state)):
                x[i] = c_double(in_state[i])
        return [x[i] for i in range(self.mre_in_state_size())]

    def mre_out_state(self, out_state=None):
        x = self.librunner.mre_out_state_from_runner(self.runner)
        if out_state != None:
            for i in range(len(out_state)):
                x[i] = c_double(out_state[i])
        return [x[i] for i in range(self.mre_out_state_size())]

    def expert_rnn_c_state(self, index, c_state=None):
        x = self.librunner.expert_rnn_c_state_from_runner(self.runner,
                c_int(index))
        if c_state != None:
            for i in range(len(c_state)):
                x[i] = c_double(c_state[i])
        return [x[i] for i in range(self.expert_rnn_c_state_size(index))]

    def expert_rnn_c_inter_state(self, index, c_inter_state=None):
        x = self.librunner.expert_rnn_c_inter_state_from_runner(self.runner,
                c_int(index))
        if c_inter_state != None:
            for i in range(len(c_inter_state)):
                x[i] = c_double(c_inter_state[i])
        return [x[i] for i in range(self.expert_rnn_c_state_size(index))]

def main():
    seed, steps, index = [int(x) if str.isdigit(x) else 0 for x in
            sys.argv[1:4]]
    mre_file = sys.argv[4]
    gn_file = sys.argv[5]
    init_genrand(seed)
    runner = MREGNRunner()
    runner.init(mre_file, gn_file)
    runner.set_time_series_id(index)
    for x,y,z in runner.closed_loop(steps):
        print('\t'.join([str(x) for x in x]))


if __name__ == '__main__':
    main()

