bin_PROGRAMS = mregn-generate
mregn_generate_SOURCES = main.c ../common/mre.c ../common/mregn_runner.c ../../rnn/src/common/rnn.c ../../rnn/src/common/utils.c
AM_CPPFLAGS = -I ../common -I ../../rnn/src/common -DENABLE_ADAPTIVE_LEARNING_RATE -DENABLE_ATTRACTION_OF_INIT_C -D_POSIX_C_SOURCE=200112L
AM_CFLAGS = $(OPENMP_CFLAGS)
