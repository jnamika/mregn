#!/bin/sh

if [ "$1" = clean ]; then
    rm -f *.log *.dat target.txt
    exit
fi

cat <<EOS | python > target.txt
import sys
sys.path.append('..')
import gen_target
for n in xrange(4):
    gen_target.print_Reber_grammar(500, 100)
    print '\n'
EOS

if [ x`which rnn-learn` == x ]; then
    path1=../../src/rnn-learn/
else
    path1=
fi
if [ x`which rnn-generate` == x ]; then
    path2=../../src/rnn-generate/
else
    path2=
fi
${path1}rnn-learn -e 10000 -l 1000 -n 30 -t 10.0 -k 1 -a -x 0.001 -m 0.9 target.txt
${path2}rnn-generate -n 2000 rnn.dat > orbit.log

