
.PHONY: all
all: 
	$(MAKE) all -C mre-learn
	$(MAKE) all -C gn-learn
	$(MAKE) all -C mregn-generate
	$(MAKE) all -C mregn-lyapunov
	$(MAKE) all -C python CC=gcc
	$(MAKE) all -C unit-test CC=gcc

.PHONY: clean
clean:
	$(MAKE) clean -C mre-learn
	$(MAKE) clean -C gn-learn
	$(MAKE) clean -C mregn-generate
	$(MAKE) clean -C mregn-lyapunov
	$(MAKE) clean -C python
	$(MAKE) clean -C unit-test

