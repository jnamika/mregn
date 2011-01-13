#!/bin/sh

my_path=../../bin

if [ "$1" = clean ]; then
    rm -f *.log *.dat target.txt
    exit
fi

echo "
import sys
sys.path.append('$my_path')
import gen_target
gen_target.print_comp_Lissajous_08curves(1000, 32)
" | python > target.txt

$my_path/mre-learn -e 10000 target.txt
$my_path/gn-learn -a -e 10000 mre.dat
$my_path/mregn-generate -n 10000 mre.dat gn.dat > orbit.log

