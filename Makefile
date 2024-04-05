COMPILER=julia

run:
	$(COMPILER) src/lorenz_cuda.jl

all: run
