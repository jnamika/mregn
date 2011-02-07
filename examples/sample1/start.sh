#!/bin/sh

if [ "$1" = clean ]; then
    rm -f *.log *.dat target.txt
    exit
fi

cat <<EOS | python > target.txt
import sys
sys.path.append('..')
import gen_target
gen_target.print_comp_Lissajous_08curves(500, 25)
EOS

mre-learn -e 30000 -d 3 target.txt
gn-learn -a -e 10000 -n 20 mre.dat
mregn-generate -n 1000 mre.dat gn.dat > orbit.log

