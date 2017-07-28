Berenger Bramas - MPCDF (berenger.bramas@mpcdf.mpg.de)

## AVX-512 sort functions

This project is a small library that provides fast functions to sort array of int or double or int[2] using AVX-512.
A paper describes the different strategies, it is currently under review, but drafts are available at:
https://arxiv.org/abs/1704.08579
This paper presents results for the KNL architecture.
An appendix that contains results for the Skylake architecture is available at http://berenger.eu/blog/wp-content/uploads/2017/06/avxsort.pdf or https://datashare.rzg.mpg.de/s/QCBTOdc5r0daqNt .


The branch `paper` contains some not very clean files that were used for benchmarks,
wherease the current master branch provides an header only library:
- sort512.hpp : the library that can be directly include in any code to sort integer or double
- sort512kv.hpp : the library that can be directly include in any code to sort key/value pairs of integers
- sort512test.cpp : some unit tests (can be used for examples)

##  Functions
- Sort512::Sort(); to sort an array
- Sort512::SortOmp(); to sort in parallel (need openmp)
- Sort512::Partition512(); to partition
- Sort512::SmallSort16V(); to sort a small array (should be less than 16 AVX512 vectors)


## AVX 512 compilation flags (KNL)
- Gcc : -mavx512f -mavx512pf -mavx512er -mavx512cd
- Intel : -xCOMMON-AVX512 -xMIC-AVX512

## AVX 512 compilation flags (SKL)
- Gcc : -mavx512f -mavx512cd -mavx512vl -mavx512bw -mavx512dq
- Intel : -xCOMMON-AVX512 -xCORE-AVX512

## OpenMP compilation flags
In case you want to use the parallel sort, you need to add the flag:
- Gcc :  -fopenmp
- Intel :  -qopenmp

## Using Intel SDE

Anyone can test the code without having a KNL by using the Intel SDE