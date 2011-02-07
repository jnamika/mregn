bin_PROGRAMS = mregn-unit-test
mregn_unit_test_SOURCES = main.c test_mre.c test_mregn.c test_mregn_lyapunov.c test_mregn_runner.c ../common/mre.c ../common/mregn.c ../common/mregn_lyapunov.c ../common/mregn_runner.c ../../rnn/src/common/rnn.c ../../rnn/src/common/solver.c ../../rnn/src/common/utils.c ../../rnn/src/unit-test/minunit.c ../../rnn/src/unit-test/test_rnn.c
AM_CPPFLAGS = -I ../common -I ../../rnn/src/common -I ../../rnn/src/unit-test -DENABLE_ADAPTIVE_LEARNING_RATE -DENABLE_ATTRACTION_OF_INIT_C -DMIN_VARIANCE=0.01 -D_POSIX_C_SOURCE=200112L
AM_CFLAGS = $(OPENMP_CFLAGS)
TESTS = mregn-unit-test
TESTS_ENVIRONMENT =
