COMPILER=julia

run:
	$(COMPILER) src/lorenz.jl

all: run
