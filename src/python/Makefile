# Makefile for: python scripts
# Data created: 12 Jan 2011
# Whom: Jun Namikawa

TARGET = libmrunner.so
VERSION = 1

CC = gcc
CPP = gcc

OBJ_DIR = ../../obj/python
BIN_DIR = ../../bin
COMMON_DIR = ../common
RNN_COMMON_DIR = ../../rnn/src/common

INCLUDE_DIR := -I. -I$(COMMON_DIR) -I$(RNN_COMMON_DIR)
LIB_DIR :=
WARNFLAG := -Wall -Wextra -Wno-unused-parameter
STD := -std=gnu99 -pedantic
OPT := -O3 -fPIC
MACROS :=
SYSLIBS := -lm
DEBUG :=

CFLAGS = $(WARNFLAG) $(STD) $(OPT) $(MACROS) $(INCLUDE_DIR) $(LIB_DIR) $(DEBUG)


vpath %.c $(COMMON_DIR) $(RNN_COMMON_DIR)
SRCS = mregn_runner.c rnn.c mre.c utils.c mt19937ar.c
PY_SRCS = mregn_runner.py print_mregn_log.py plot_mregn_log.py mregn_kl_div.py
SH_SRCS = print-log plot-log mregn-kl-div
OBJS = $(SRCS:%.c=$(OBJ_DIR)/%.o)


.PHONY: all
all: depend.mk $(OBJ_DIR) $(BIN_DIR) $(BIN_DIR)/$(TARGET) $(PY_SRCS) $(SH_SRCS)
	-@ cp $(PY_SRCS) $(SH_SRCS) $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/$(TARGET): $(OBJS)
	$(CC) -shared -o $@ $(OBJS) $(CFLAGS) $(SYSLIBS)

$(OBJ_DIR)/%.o: %.c
	$(CC) -c $< -o $@ $(CFLAGS)


depend.mk: $(SRCS)
	-@ echo "creating depend.mk"
	-@ rm -f depend.mk
	-@ for i in $^ ; do \
		($(CPP) -MM $(INCLUDE_DIR) $(MACROS) $$i >> depend.mk); \
	done
	-@ cat depend.mk | sed 's:^.*\.o:$(OBJ_DIR)/&:g' > .depend.mk
	-@ mv .depend.mk depend.mk

-include depend.mk



.PHONY: clean
clean:
	-@ rm -rf $(OBJ_DIR) $(BIN_DIR)/$(TARGET) $(PY_SRCS:%=$(BIN_DIR)/%) $(SH_SRCS:%=$(BIN_DIR)/%) depend.mk

.PHONY: help
help: info

info:
	-@ echo
	-@ echo "Makefile for shared library of mixture of RNN experts used by python script"
	-@ echo "Jun Namikawa, 2010"
	-@ echo
	-@ echo "USAGE: make [all | clean | help]"
	-@ echo "all (default)      builds $(TARGET)"
	-@ echo "clean              removes target, depend.mk and .o files"
	-@ echo "help               displays help"
	-@ echo

.PHONY: echo
echo:
	@echo TARGET=$(TARGET)
	@echo CC=$(CC)
	@echo WARNFLAG=$(WARNFLAG)
	@echo STD=$(STD)
	@echo OPT=$(OPT)
	@echo MACROS=$(MACROS)
	@echo INCLUDE_DIR=$(INCLUDE_DIR)
	@echo LIB_DIR=$(LIB_DIR)
	@echo DEBUG=$(DEBUG)

