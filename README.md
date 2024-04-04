# Cuda + Julia
This project aims to explore the use of Julia and CUDA through a practical case study.

## Project Description

We used JuliaGPU (https://juliagpu.org) to run Julia programs on NVIDIA CUDA platforms. 

This project aims to solve Lorenz using CUDA on graphical processing units (GPUs) with the Julia programming language.

Another interesting reference is [Solving PDEs in parallel on GPUs with Julia](https://pde-on-gpu.vaw.ethz.ch/)
link: https://pde-on-gpu.vaw.ethz.ch



## Code Organization

```src/```



```bin/```
This folder should hold all binary/executable code that is built automatically or manually. Executable code should have use the .exe extension or programming language-specific extension.

```data/```
This folder should hold all example data in any format. If the original data is rather large or can be brought in via scripts, this can be left blank in the respository, so that it doesn't require major downloads when all that is desired is the code/structure.

```lib/```
Any libraries that are not installed via the Operating System-specific package manager should be placed here, so that it is easier for inclusion/linking.



```README.md```
This file should hold the description of the project so that anyone cloning or deciding if they want to clone this repository can understand its purpose to help with their decision.

```INSTALL```
This file should hold the human-readable set of instructions for installing the code so that it can be executed. If possible it should be organized around different operating systems, so that it can be done by as many people as possible with different constraints.

```Makefile or CMAkeLists.txt or build.sh```
There should be some rudimentary scripts for building your project's code in an automatic fashion.

```run.sh```
An optional script used to run your executable code, either with or without command-line arguments.