lib_LTLIBRARIES = libmregnrunner.la
libmregnrunner_la_SOURCES = ../common/mregn_runner.c ../common/mre.c ../../rnn/src/common/rnn.c ../../rnn/src/common/utils.c
AM_CPPFLAGS = -I ../common -I ../../rnn/src/common -DENABLE_ADAPTIVE_LEARNING_RATE -DENABLE_ATTRACTION_OF_INIT_C -D_POSIX_C_SOURCE=200112L
PY_SRCS = mregn_print_log.py mregn_plot_log.py mregn_runner.py mregn_kl_div.py
SH_SRCS = mregn-print-log mregn-plot-log mregn-kl-div
bin_SCRIPTS = $(PY_SRCS) $(SH_SRCS)
