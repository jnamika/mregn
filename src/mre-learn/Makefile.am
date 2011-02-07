bin_PROGRAMS = mre-learn
mre_learn_SOURCES = main.c training.c print.c ../common/mre.c ../../rnn/src/common/rnn.c ../../rnn/src/common/utils.c ../../rnn/src/rnn-learn/parse.c ../../rnn/src/rnn-learn/target.c
AM_CPPFLAGS = -I ../common -I ../../rnn/src/common -I ../../rnn/src/rnn-learn -DENABLE_ADAPTIVE_LEARNING_RATE -DENABLE_ATTRACTION_OF_INIT_C -D_POSIX_C_SOURCE=200112L
AM_CFLAGS = $(OPENMP_CFLAGS)
