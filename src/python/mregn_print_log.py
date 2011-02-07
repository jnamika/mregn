# -*- coding:utf-8 -*-

import sys
import re
import rnn_print_log

def read_parameter(f):
    r = {}
    r['expert_num'] = re.compile(r'^# expert_num')
    r['in_state_size'] = re.compile(r'^# in_state_size')
    r['c_state_size'] = re.compile(r'^# c_state_size')
    r['out_state_size'] = re.compile(r'^# out_state_size')
    r['delay_length'] = re.compile(r'^# delay_length')
    r['target_num'] = re.compile(r'^# target_num')
    r['target'] = re.compile(r'^# target ([0-9]+)')
    r_comment = re.compile(r'^#')
    params = {}
    for line in f:
        for k,v in r.iteritems():
            if (v.match(line)):
                x = int(line.split('=')[1])
                if k == 'target':
                    m = v.match(line).group(1)
                    if (k in params):
                        params[k][m] = x
                    else:
                        params[k] = {m:x}
                else:
                    params[k] = x

        if (r_comment.match(line) == None):
            break
    f.seek(0)
    return params


def print_state(f, epoch=None):
    rnn_print_log.print_state(f, epoch)

def print_gate(f, epoch=None):
    rnn_print_log.print_state(f, epoch)

def print_weight(f, epoch=None):
    params = read_parameter(f)
    expert_num = int(params['expert_num'])
    in_state_size = int(params['in_state_size'])
    c_state_size = int(params['c_state_size'])
    out_state_size = int(params['out_state_size'])
    s = rnn_print_log.current_line(f, epoch)
    if s != None:
        epoch = s[0]
        print 'epoch : %s' % epoch
        s = s[1:]
        for i in xrange(expert_num):
            print 'expert : %s' % i
            w_i2c, w_c2c, w_c2o = [], [], []
            for j in xrange(c_state_size):
                w_i2c.append(s[:in_state_size])
                s = s[in_state_size:]
                w_c2c.append(s[:c_state_size])
                s = s[c_state_size:]
            for j in xrange(out_state_size):
                w_c2o.append(s[:c_state_size])
                s = s[c_state_size:]
            print 'weight (input to context)'
            for w in w_i2c:
                print '\t'.join([str(x) for x in w])
            print 'weight (context to context)'
            for w in w_c2c:
                print '\t'.join([str(x) for x in w])
            print 'weight (context to output)'
            for w in w_c2o:
                print '\t'.join([str(x) for x in w])

def print_threshold(f, epoch=None):
    params = read_parameter(f)
    expert_num = int(params['expert_num'])
    c_state_size = int(params['c_state_size'])
    out_state_size = int(params['out_state_size'])
    s = rnn_print_log.current_line(f, epoch)
    if s != None:
        epoch = s[0]
        print 'epoch : %s' % epoch
        s = s[1:]
        for i in xrange(expert_num):
            print 'expert : %s' % i
            t_c = s[:c_state_size]
            s = s[c_state_size:]
            t_o = s[:out_state_size]
            s = s[out_state_size:]
            print 'threshold (context)'
            print '\t'.join([str(x) for x in t_c])
            print 'threshold (output)'
            print '\t'.join([str(x) for x in t_o])

def print_tau(f, epoch=None):
    params = read_parameter(f)
    expert_num = int(params['expert_num'])
    c_state_size = int(params['c_state_size'])
    s = rnn_print_log.current_line(f, epoch)
    if s != None:
        epoch = s[0]
        print 'epoch : %s' % epoch
        s = s[1:]
        for i in xrange(expert_num):
            print 'expert : %s' % i
            tau = s[:c_state_size]
            s = s[c_state_size:]
            print 'time constant'
            print '\t'.join([str(x) for x in tau])

def print_sigma(f, epoch=None):
    s = rnn_print_log.current_line(f, epoch)
    if s != None:
        epoch = s[0]
        sigma = s[1::2]
        variance = s[2::2]
        print 'epoch : %s' % epoch
        print 'sigma : %s' % '\t'.join([str(x) for x in sigma])
        print 'variance : %s' % '\t'.join([str(x) for x in variance])

def print_init(f, epoch=None):
    rnn_print_log.print_init(f, epoch)

def print_adapt_lr(f, epoch=None):
    rnn_print_log.print_adapt_lr(f, epoch)

def print_error(f, epoch=None):
    s = rnn_print_log.current_line(f, epoch)
    if s != None:
        epoch = s[0]
        error = s[1::3]
        joint_likelihood = s[2::3]
        total_likelihood = s[3::3]
        print 'epoch : %s' % epoch
        print 'error / (length * dimension)'
        print '\t'.join([str(x) for x in error])
        print 'joint_likelihood / length'
        print '\t'.join([str(x) for x in joint_likelihood])
        print 'total_likelihood / length'
        print '\t'.join([str(x) for x in total_likelihood])

def print_log(f, epoch):
    line = f.readline()
    if (re.compile(r'^# MRE STATE FILE').match(line)):
        print_state(f, epoch)
    elif (re.compile(r'^# MRE GATE FILE').match(line)):
        print_gate(f, epoch)
    elif (re.compile(r'^# MRE WEIGHT FILE').match(line)):
        print_weight(f, epoch)
    elif (re.compile(r'^# MRE THRESHOLD FILE').match(line)):
        print_threshold(f, epoch)
    elif (re.compile(r'^# MRE TAU FILE').match(line)):
        print_tau(f, epoch)
    elif (re.compile(r'^# MRE SIGMA FILE').match(line)):
        print_sigma(f, epoch)
    elif (re.compile(r'^# MRE INIT FILE').match(line)):
        print_init(f, epoch)
    elif (re.compile(r'^# MRE ADAPT_LR FILE').match(line)):
        print_adapt_lr(f, epoch)
    elif (re.compile(r'^# MRE ERROR FILE').match(line)):
        print_error(f, epoch)
    else:
        f.seek(0)
        rnn_print_log.print_log(f, epoch)

def main():
    epoch = None
    if str.isdigit(sys.argv[1]):
        epoch = int(sys.argv[1])
    for file in sys.argv[2:]:
        f = open(file, 'r')
        print_log(f, epoch)
        f.close()


if __name__ == '__main__':
    main()

