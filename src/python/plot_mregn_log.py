# -*- coding:utf-8 -*-

import sys
import re
import subprocess
import tempfile
import print_mregn_log
import plot_log

def plot_state(f, filename, epoch):
    params = print_mregn_log.read_parameter(f)
    out_state_size = int(params['out_state_size'])
    tmp = tempfile.NamedTemporaryFile()
    sys.stdout = tmp
    print_mregn_log.print_state(f, epoch)
    sys.stdout.flush()
    type = {}
    type['Target'] = (out_state_size, lambda x: 2 * x + 2)
    type['Output'] = (out_state_size, lambda x: 2 * x + 3)
    for k,v in type.iteritems():
        p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                shell=True)
        p.stdin.write("set nokey;")
        p.stdin.write("set title 'Type=%s  File=%s';" % (k, filename))
        p.stdin.write("set xlabel 'Time step';")
        p.stdin.write("set ylabel '%s';" % k)
        command = ["plot "]
        for i in xrange(v[0]):
            command.append("'%s' u 1:%d w l," % (tmp.name, v[1](i)))
        p.stdin.write(''.join(command)[:-1])
        p.stdin.write('\n')
        p.stdin.write('exit\n')
        p.wait()
    sys.stdout = sys.__stdout__

def plot_gate(f, filename, epoch):
    params = print_mregn_log.read_parameter(f)
    expert_num = int(params['expert_num'])
    tmp = tempfile.NamedTemporaryFile()
    sys.stdout = tmp
    print_mregn_log.print_state(f, epoch)
    sys.stdout.flush()
    p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
            shell=True)
    p.stdin.write("set nokey;")
    p.stdin.write("set title 'Type=Gate  File=%s';" % filename)
    p.stdin.write("set xlabel 'Time step';")
    p.stdin.write("set ylabel 'Gate opening value';")
    command = ["plot "]
    for i in xrange(expert_num):
        command.append("'%s' u 1:%d w l," % (tmp.name, i+2))
    p.stdin.write(''.join(command)[:-1])
    p.stdin.write('\n')
    p.stdin.write('exit\n')
    p.wait()
    sys.stdout = sys.__stdout__

def plot_weight(f, filename):
    params = print_mregn_log.read_parameter(f)
    expert_num = int(params['expert_num'])
    in_state_size = int(params['in_state_size'])
    c_state_size = int(params['c_state_size'])
    out_state_size = int(params['out_state_size'])
    s = [x+2 for x in xrange(expert_num * c_state_size *
        (in_state_size + c_state_size + out_state_size))]
    for i in xrange(expert_num):
        index_i2c, index_c2c, index_c2o = [], [], []
        for j in xrange(c_state_size):
            index_i2c.extend(s[:in_state_size])
            s = s[in_state_size:]
            index_c2c.extend(s[:c_state_size])
            s = s[c_state_size:]
        for j in xrange(out_state_size):
            index_c2o.extend(s[:c_state_size])
            s = s[c_state_size:]
        type = {'Weight (input to context)':index_i2c,
                'Weight (context to context)':index_c2c,
                'Weight (context to output)':index_c2o}
        for k,v in type.iteritems():
            p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                    shell=True)
            p.stdin.write("set nokey;")
            p.stdin.write("set title 'Type=Weight(Expert %d)  File=%s';" % (i,
                filename))
            p.stdin.write("set xlabel 'Learning epoch';")
            p.stdin.write("set ylabel '%s';" % k)
            command = ["plot "]
            for j in v:
                command.append("'%s' u 1:%d w l," % (filename, j))
            p.stdin.write(''.join(command)[:-1])
            p.stdin.write('\n')

def plot_threshold(f, filename):
    params = print_mregn_log.read_parameter(f)
    expert_num = int(params['expert_num'])
    c_state_size = int(params['c_state_size'])
    out_state_size = int(params['out_state_size'])
    s = [x+2 for x in xrange(expert_num * (c_state_size + out_state_size))]
    for i in xrange(expert_num):
        index_c = s[:c_state_size]
        s = s[c_state_size:]
        index_o = s[:out_state_size]
        s = s[out_state_size:]
        type = {'Threshold (context)':index_c, 'Threshold (output)':index_o}
        for k,v in type.iteritems():
            p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                    shell=True)
            p.stdin.write("set nokey;")
            p.stdin.write("set title 'Type=Threshold(Expert %d)  File=%s';" % \
                    (i, filename))
            p.stdin.write("set xlabel 'Learning epoch';")
            p.stdin.write("set ylabel '%s';" % k)
            command = ["plot "]
            for j in v:
                command.append("'%s' u 1:%d w l," % (filename, j))
            p.stdin.write(''.join(command)[:-1])
            p.stdin.write('\n')

def plot_tau(f, filename):
    params = print_mregn_log.read_parameter(f)
    expert_num = int(params['expert_num'])
    c_state_size = int(params['c_state_size'])
    for i in xrange(expert_num):
        p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                shell=True)
        p.stdin.write("set nokey;")
        p.stdin.write("set title 'Type=Time-constant(Expert %d)  File=%s';" % \
                (i, filename))
        p.stdin.write("set xlabel 'Learning epoch';")
        p.stdin.write("set ylabel 'Time constant';")
        command = ["plot "]
        for j in xrange(c_state_size):
            command.append("'%s' u 1:%d w l," % (filename, i*expert_num+j+2))
        p.stdin.write(''.join(command)[:-1])
        p.stdin.write('\n')

def plot_sigma(f, filename):
    params = print_mregn_log.read_parameter(f)
    expert_num = int(params['expert_num'])
    p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
            shell=True)
    p.stdin.write("set nokey;")
    p.stdin.write("set title 'Type=Variance  File=%s';" % filename)
    p.stdin.write("set xlabel 'Learning epoch';")
    p.stdin.write("set ylabel 'Variance';")
    command = ["plot "]
    for i in xrange(expert_num):
        command.append("'%s' u 1:%d w l," % (filename, 2*i+3))
    p.stdin.write(''.join(command)[:-1])
    p.stdin.write('\n')

def plot_init(f, filename, epoch):
    params = print_mregn_log.read_parameter(f)
    expert_num = int(params['expert_num'])
    c_state_size = int(params['c_state_size'])
    tmp = tempfile.NamedTemporaryFile()
    sys.stdout = tmp
    print_mregn_log.print_init(f, epoch)
    sys.stdout.flush()
    index = [(2*x,(2*x+1)%c_state_size) for x in xrange(c_state_size) if 2*x <
            c_state_size]
    for i in xrange(expert_num):
        p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                shell=True)
        p.stdin.write("set nokey;")
        p.stdin.write("set title 'Type=Init(Expert %d)  File=%s';" % (i,
            filename))
        p.stdin.write("set xlabel 'Time step';")
        p.stdin.write("set ylabel 'Initial state';")
        p.stdin.write("set pointsize 3;")
        command = ["plot "]
        k = expert_num * i + 2
        for x in index:
            command.append("'%s' u %d:%d w p," % (tmp.name, x[0] + k,
                x[1] + k))
        p.stdin.write(''.join(command)[:-1])
        p.stdin.write('\n')
        p.stdin.write('exit\n')
        p.wait()
    sys.stdout = sys.__stdout__

def plot_adapt_lr(f, filename):
    plot_log.plot_adapt_lr(f, filename)

def plot_error(f, filename):
    params = print_mregn_log.read_parameter(f)
    target_num = int(params['target_num'])
    type = {}
    type['Error'] = ('Error / (Length times Dimension)', lambda x: 3 * x + 2, \
            'set logscale y;')
    type['Joint likelihood'] = ('Joint-likelihood / Length',
            lambda x: 3 * x + 3, '')
    type['Total likelihood'] = ('Total-likelihood / Length',
            lambda x: 3 * x + 4, '')
    for k,v in type.iteritems():
        p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                shell=True)
        p.stdin.write("set nokey;")
        p.stdin.write("set title 'Type=%s  File=%s';" % (k, filename))
        p.stdin.write("set xlabel 'Learning epoch';")
        p.stdin.write("set ylabel '%s';" % v[0])
        p.stdin.write(v[2])
        command = ["plot "]
        for i in xrange(target_num):
            command.append("'%s' u 1:%d w l," % (filename, v[1](i)))
        p.stdin.write(''.join(command)[:-1])
        p.stdin.write('\n')
        p.stdin.write('exit\n')
        p.wait()

def plot_mre_log(files, epoch):
    for file in files:
        f = open(file, 'r')
        line = f.readline()
        if (re.compile(r'^# MRE STATE FILE').match(line)):
            plot_state(f, file, epoch)
        if (re.compile(r'^# MRE GATE FILE').match(line)):
            plot_gate(f, file, epoch)
        elif (re.compile(r'^# MRE WEIGHT FILE').match(line)):
            plot_weight(f, file)
        elif (re.compile(r'^# MRE THRESHOLD FILE').match(line)):
            plot_threshold(f, file)
        elif (re.compile(r'^# MRE TAU FILE').match(line)):
            plot_tau(f, file)
        elif (re.compile(r'^# MRE SIGMA FILE').match(line)):
            plot_sigma(f, file)
        elif (re.compile(r'^# MRE INIT FILE').match(line)):
            plot_init(f, file, epoch)
        elif (re.compile(r'^# MRE ADAPT_LR FILE').match(line)):
            plot_adapt_lr(f, file)
        elif (re.compile(r'^# MRE ERROR FILE').match(line)):
            plot_error(f, file)
        else:
            plot_log.plot_log([file], epoch)
        f.close()


def main():
    epoch = None
    if str.isdigit(sys.argv[1]):
        epoch = int(sys.argv[1])
    plot_mre_log(sys.argv[2:], epoch)


if __name__ == "__main__":
    main()

