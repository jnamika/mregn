#!/bin/sh

dir=`echo $(cd $(dirname $0) && pwd)`

openmp=('-O3 -fopenmp' '-O3')
debug=('-DNDEBUG' '')
adapt_lr=('-DENABLE_ADAPTIVE_LEARNING_RATE' '')
attract=('-DENABLE_ATTRACTION_OF_INIT_C' '')


for i in 0 1
do
    for j in 0 1
    do
        for k in 0 1
        do
            for l in 0 1
            do
                OPT=${openmp[$i]}
                MACROS='-DVERSION=$(VERSION)'
                MACROS="$MACROS ${debug[$j]}"
                MACROS="$MACROS ${adapt_lr[$k]}"
                MACROS="$MACROS ${attract[$l]}"
                cd ../mre-learn
                make CC=gcc OBJ_DIR=$dir/obj/mre-learn/omp${i}d${j}ad${k}at${l} BIN_DIR=$dir/bin OPT="$OPT" MACROS="$MACROS" TARGET=mre-learn-omp${i}d${j}ad${k}at${l}
                cd $dir
                cd ../gn-learn
                make CC=gcc OBJ_DIR=$dir/obj/gn-learn/omp${i}d${j}ad${k}at${l} BIN_DIR=$dir/bin OPT="$OPT" MACROS="$MACROS" TARGET=gn-learn-omp${i}d${j}ad${k}at${l}
                cd $dir
                cd ../mregn-generate
                make CC=gcc OBJ_DIR=$dir/obj/mregn-generate/omp${i}d${j}ad${k}at${l} BIN_DIR=$dir/bin OPT="$OPT" MACROS="$MACROS" TARGET=mregn-generate-omp${i}d${j}ad${k}at${l}
                cd $dir
                cd ../mregn-lyapunov
                make CC=gcc OBJ_DIR=$dir/obj/mregn-lyapunov/omp${i}d${j}ad${k}at${l} BIN_DIR=$dir/bin OPT="$OPT" MACROS="$MACROS" TARGET=mregn-lyapunov-omp${i}d${j}ad${k}at${l}
                cd $dir
            done
        done
    done
done

cd ../python
make CC=gcc OBJ_DIR="$dir"/obj/python BIN_DIR="$dir"/bin
cd $dir


