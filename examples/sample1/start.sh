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
gen_target.print_comp_Lissajous_08curves(500, 25)
EOS

$my_path/mre-learn -e 30000 -d 3 target.txt
$my_path/gn-learn -a -e 10000 -n 20 mre.dat
$my_path/mregn-generate -n 1000 mre.dat gn.dat > orbit.log

