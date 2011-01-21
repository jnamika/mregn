
.PHONY: all
all: 
	$(MAKE) -C rnn BIN_DIR=`pwd`/bin
	$(MAKE) -C src

.PHONY: clean
clean:
	$(MAKE) clean -C rnn BIN_DIR=`pwd`/bin
	$(MAKE) clean -C src
	rm -rf obj
