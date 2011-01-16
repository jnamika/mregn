#!/bin/sh

my_path=../../bin

if [ "$1" = clean ]; then
    rm -f *.log *.dat target.txt
    exit
fi

cat <<EOS | python > target.txt
import sys
sys.path.append('$my_path')
import gen_target
for n in xrange(3):
    gen_target.print_henon_map(500, 100)
    print '\n'
EOS

$my_path/rnn-learn -e 2000 -n 20 -t 5.0 -p 0.001 -a target.txt
$my_path/rnn-generate -n 1500 rnn.dat > orbit.log

