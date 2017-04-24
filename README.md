Berenger Bramas - MPCDF (berenger.bramas@mpcdf.mpg.de)

## AVX-512 sort functions

This project is a small library that provides fast functions to sort array of int or double using AVX-512.
A paper describes the different strategies, it is currently under review, but drafts are available at:
arxiv
or
https://hal.inria.fr/hal-01512970


The branch `paper` contains some not very clean files that were used for benchmarks,
wherease the current master branch provides an header only library:
- sort512.hpp : the library that can be directly include in any code
- sort512test.cpp : some unit tests (can be used for examples)

##  Functions
- Sort512::Sort(); to sort an array
- Sort512::SortOmp(); to sort in parallel (need openmp)
- Sort512::Partition512(); to partition
- Sort512::SmallSort16V(); to sort a small array (should be less than 16 AVX512 vectors)


## AVX 512 compilation flags
- Gcc : -mavx512f -mavx512pf -mavx512er -mavx512cd
- Intel : -xCOMMON-AVX512 -xMIC-AVX512

In case you want to use the parallel sort, you need to add the flag:
- Gcc :  -fopenmp
- Intel :  -qopenmp

## Using Intel SDE

Anyone can test the code without having a KNL by using the Intel SDE