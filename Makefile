COMPILER=julia

run:
	$(COMPILER) src/test_cuda.jl

all: run
