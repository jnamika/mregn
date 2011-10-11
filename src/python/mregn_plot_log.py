# -*- coding:utf-8 -*-

import sys
import re
import subprocess
import tempfile
import mregn_print_log
import rnn_plot_log

def plot_state(f, filename, epoch):
    params = mregn_print_log.read_parameter(f)
    out_state_size = int(params['out_state_size'])
    tmp = tempfile.NamedTemporaryFile('w+')
    sys.stdout = tmp
    mregn_print_log.print_state(f, epoch)
    sys.stdout.flush()
    ptype = {}
    ptype['Target'] = (out_state_size, lambda x: 2 * x + 2)
    ptype['Output'] = (out_state_size, lambda x: 2 * x + 3)
    for k,v in ptype.items():
        p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                shell=True)
        gnuplot = lambda s: p.stdin.write(s.encode())
        gnuplot('set nokey;')
        gnuplot("set title 'Type=%s  File=%s';" % (k, filename))
        gnuplot("set xlabel 'Time step';")
        gnuplot("set ylabel '%s';" % k)
        command = ['plot ']
        for i in range(v[0]):
            command.append("'%s' u 1:%d w l," % (tmp.name, v[1](i)))
        gnuplot(''.join(command)[:-1])
        gnuplot('\n')
        gnuplot('exit\n')
        p.wait()
    sys.stdout = sys.__stdout__

def plot_gate(f, filename, epoch):
    params = mregn_print_log.read_parameter(f)
    expert_num = int(params['expert_num'])
    tmp = tempfile.NamedTemporaryFile('w+')
    sys.stdout = tmp
    mregn_print_log.print_state(f, epoch)
    sys.stdout.flush()
    p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
            shell=True)
    gnuplot = lambda s: p.stdin.write(s.encode())
    gnuplot('set nokey;')
    gnuplot("set title 'Type=Gate  File=%s';" % filename)
    gnuplot("set xlabel 'Time step';")
    gnuplot("set ylabel 'Gate opening value';")
    command = ['plot ']
    for i in range(expert_num):
        command.append("'%s' u 1:%d w l," % (tmp.name, i+2))
    gnuplot(''.join(command)[:-1])
    gnuplot('\n')
    gnuplot('exit\n')
    p.wait()
    sys.stdout = sys.__stdout__

def plot_weight(f, filename):
    params = mregn_print_log.read_parameter(f)
    expert_num = int(params['expert_num'])
    in_state_size = int(params['in_state_size'])
    c_state_size = int(params['c_state_size'])
    out_state_size = int(params['out_state_size'])
    s = [x+2 for x in range(expert_num * c_state_size *
        (in_state_size + c_state_size + out_state_size))]
    for i in range(expert_num):
        index_i2c, index_c2c, index_c2o = [], [], []
        for j in range(c_state_size):
            index_i2c.extend(s[:in_state_size])
            s = s[in_state_size:]
            index_c2c.extend(s[:c_state_size])
            s = s[c_state_size:]
        for j in range(out_state_size):
            index_c2o.extend(s[:c_state_size])
            s = s[c_state_size:]
        ptype = {'Weight (input to context)':index_i2c,
                'Weight (context to context)':index_c2c,
                'Weight (context to output)':index_c2o}
        for k,v in ptype.items():
            p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                    shell=True)
            gnuplot = lambda s: p.stdin.write(s.encode())
            gnuplot('set nokey;')
            gnuplot("set title 'Type=Weight(Expert %d)  File=%s';" % (i,
                filename))
            gnuplot("set xlabel 'Learning epoch';")
            gnuplot("set ylabel '%s';" % k)
            command = ['plot ']
            for j in v:
                command.append("'%s' u 1:%d w l," % (filename, j))
            gnuplot(''.join(command)[:-1])
            gnuplot('\n')

def plot_threshold(f, filename):
    params = mregn_print_log.read_parameter(f)
    expert_num = int(params['expert_num'])
    c_state_size = int(params['c_state_size'])
    out_state_size = int(params['out_state_size'])
    s = [x+2 for x in range(expert_num * (c_state_size + out_state_size))]
    for i in range(expert_num):
        index_c = s[:c_state_size]
        s = s[c_state_size:]
        index_o = s[:out_state_size]
        s = s[out_state_size:]
        ptype = {'Threshold (context)':index_c, 'Threshold (output)':index_o}
        for k,v in ptype.items():
            p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                    shell=True)
            gnuplot = lambda s: p.stdin.write(s.encode())
            gnuplot('set nokey;')
            gnuplot("set title 'Type=Threshold(Expert %d)  File=%s';" %
                    (i, filename))
            gnuplot("set xlabel 'Learning epoch';")
            gnuplot("set ylabel '%s';" % k)
            command = ['plot ']
            for j in v:
                command.append("'%s' u 1:%d w l," % (filename, j))
            gnuplot(''.join(command)[:-1])
            gnuplot('\n')

def plot_tau(f, filename):
    params = mregn_print_log.read_parameter(f)
    expert_num = int(params['expert_num'])
    c_state_size = int(params['c_state_size'])
    for i in range(expert_num):
        p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                shell=True)
        gnuplot = lambda s: p.stdin.write(s.encode())
        gnuplot('set nokey;')
        gnuplot("set title 'Type=Time-constant(Expert %d)  File=%s';" %
                (i, filename))
        gnuplot("set xlabel 'Learning epoch';")
        gnuplot("set ylabel 'Time constant';")
        command = ['plot ']
        for j in range(c_state_size):
            command.append("'%s' u 1:%d w l," % (filename, i*expert_num+j+2))
        gnuplot(''.join(command)[:-1])
        gnuplot('\n')

def plot_sigma(f, filename):
    params = mregn_print_log.read_parameter(f)
    expert_num = int(params['expert_num'])
    p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
            shell=True)
    gnuplot = lambda s: p.stdin.write(s.encode())
    gnuplot('set nokey;')
    gnuplot("set title 'Type=Variance  File=%s';" % filename)
    gnuplot("set xlabel 'Learning epoch';")
    gnuplot("set ylabel 'Variance';")
    command = ['plot ']
    for i in range(expert_num):
        command.append("'%s' u 1:%d w l," % (filename, 2*i+3))
    gnuplot(''.join(command)[:-1])
    gnuplot('\n')

def plot_init(f, filename, epoch):
    params = mregn_print_log.read_parameter(f)
    expert_num = int(params['expert_num'])
    c_state_size = int(params['c_state_size'])
    tmp = tempfile.NamedTemporaryFile('w+')
    sys.stdout = tmp
    mregn_print_log.print_init(f, epoch)
    sys.stdout.flush()
    index = [(2*x,(2*x+1)%c_state_size) for x in range(c_state_size) if 2*x <
            c_state_size]
    for i in range(expert_num):
        p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                shell=True)
        gnuplot = lambda s: p.stdin.write(s.encode())
        gnuplot('set nokey;')
        gnuplot("set title 'Type=Init(Expert %d)  File=%s';" % (i, filename))
        gnuplot("set xlabel 'x';")
        gnuplot("set ylabel 'y';")
        gnuplot('set pointsize 3;')
        command = ['plot ']
        k = expert_num * i + 2
        for x in index:
            command.append("'%s' u %d:%d w p," % (tmp.name, x[0] + k,
                x[1] + k))
        gnuplot(''.join(command)[:-1])
        gnuplot('\n')
        gnuplot('exit\n')
        p.wait()
    sys.stdout = sys.__stdout__

def plot_adapt_lr(f, filename):
    rnn_plot_log.plot_adapt_lr(f, filename)

def plot_error(f, filename):
    params = mregn_print_log.read_parameter(f)
    target_num = int(params['target_num'])
    ptype = {}
    ptype['Error'] = ('Error / (Length times Dimension)', lambda x: 3 * x + 2,
            'set logscale y;')
    ptype['Joint likelihood'] = ('Joint-likelihood / Length',
            lambda x: 3 * x + 3, '')
    ptype['Total likelihood'] = ('Total-likelihood / Length',
            lambda x: 3 * x + 4, '')
    for k,v in ptype.items():
        p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                shell=True)
        gnuplot = lambda s: p.stdin.write(s.encode())
        gnuplot('set nokey;')
        gnuplot("set title 'Type=%s  File=%s';" % (k, filename))
        gnuplot("set xlabel 'Learning epoch';")
        gnuplot("set ylabel '%s';" % v[0])
        gnuplot(v[2])
        command = ['plot ']
        for i in range(target_num):
            command.append("'%s' u 1:%d w l," % (filename, v[1](i)))
        gnuplot(''.join(command)[:-1])
        gnuplot('\n')
        gnuplot('exit\n')
        p.wait()

def plot_log(f, file, epoch):
    line = f.readline()
    if re.compile(r'^# MRE STATE FILE').match(line):
        plot_state(f, file, epoch)
    elif re.compile(r'^# MRE GATE FILE').match(line):
        plot_gate(f, file, epoch)
    elif re.compile(r'^# MRE WEIGHT FILE').match(line):
        plot_weight(f, file)
    elif re.compile(r'^# MRE THRESHOLD FILE').match(line):
        plot_threshold(f, file)
    elif re.compile(r'^# MRE TAU FILE').match(line):
        plot_tau(f, file)
    elif re.compile(r'^# MRE SIGMA FILE').match(line):
        plot_sigma(f, file)
    elif re.compile(r'^# MRE INIT FILE').match(line):
        plot_init(f, file, epoch)
    elif re.compile(r'^# MRE ADAPT_LR FILE').match(line):
        plot_adapt_lr(f, file)
    elif re.compile(r'^# MRE ERROR FILE').match(line):
        plot_error(f, file)
    else:
        f.seek(0)
        rnn_plot_log.plot_log(f, file, epoch)


def main():
    epoch = None
    if str.isdigit(sys.argv[1]):
        epoch = int(sys.argv[1])
    for file in sys.argv[2:]:
        f = open(file, 'r')
        plot_log(f, file, epoch)
        f.close()


if __name__ == '__main__':
    main()

