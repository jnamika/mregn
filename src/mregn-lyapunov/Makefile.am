bin_PROGRAMS = mregn-lyapunov
mregn_lyapunov_SOURCES = main.c lyapunov.c ../common/mre.c ../common/mregn_runner.c ../common/mregn_lyapunov.c ../../rnn/src/common/rnn.c ../../rnn/src/common/solver.c ../../rnn/src/common/utils.c
AM_CPPFLAGS = -I ../common -I ../../rnn/src/common -DENABLE_ADAPTIVE_LEARNING_RATE -DENABLE_ATTRACTION_OF_INIT_C -D_POSIX_C_SOURCE=200112L
AM_CFLAGS = $(OPENMP_CFLAGS)
