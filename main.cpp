////////////////////////////////////////////////////////////
/// Berenger Bramas - 2016
/// berenger.bramas@mpcdf.mpg.de
/// MIT Licence
/// AVX 512 Sorting algorithm
////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
/// Compilation :
/// Gcc : g++ -DNDEBUG -O3 -funroll-loops -faggressive-loop-optimizations -std=c++11 -mavx512f -mavx512pf -mavx512er -mavx512cd -fopenmp main.cpp -o test.gcc.exe
/// Intel : icpc -DNDEBUG -O3 -std=c++11 -xCOMMON-AVX512 -xMIC-AVX512 -qopenmp main.cpp -o test.intel.exe
///
/// Numa:
/// In Flat mode list with : numactl --hardware
/// Then run with : numactl --physcpubind=8 --membind=1  EXEC
////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////
/// Utils Functions
////////////////////////////////////////////////////////////


#include <immintrin.h>
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cassert>
#include <stdexcept>
#include <climits>
#include <cfloat>

class dtimer {
    using double_second_time = std::chrono::duration<double, std::ratio<1, 1>>;

    std::chrono::high_resolution_clock::time_point
    m_start;  ///< m_start time (start)
    std::chrono::high_resolution_clock::time_point m_end;  ///< stop time (stop)
    std::chrono::nanoseconds m_cumulate;  ///< the m_cumulate time

public:
    /// Constructor
    dtimer() { start(); }

    /// Copy constructor
    dtimer(const dtimer& other) = delete;
    /// Copies an other timer
    dtimer& operator=(const dtimer& other) = delete;
    /// Move constructor
    dtimer(dtimer&& other) = delete;
    /// Copies an other timer
    dtimer& operator=(dtimer&& other) = delete;

    /** Rest all the values, and apply start */
    void reset() {
        m_start = std::chrono::high_resolution_clock::time_point();
        m_end = std::chrono::high_resolution_clock::time_point();
        m_cumulate = std::chrono::nanoseconds();
        start();
    }

    /** Start the timer */
    void start() {
        m_start = std::chrono::high_resolution_clock::now();
    }

    /** Stop the current timer */
    void stop() {
        m_end = std::chrono::high_resolution_clock::now();
        m_cumulate += std::chrono::duration_cast<std::chrono::nanoseconds>(m_end - m_start);
    }

    /** Return the elapsed time between start and stop (in second) */
    double getElapsed() const {
        return std::chrono::duration_cast<double_second_time>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(m_end - m_start)).count();
    }

    /** Return the total counted time */
    double getCumulated() const {
        return std::chrono::duration_cast<double_second_time>(m_cumulate).count();
    }

    /** End the current counter (stop) and return the elapsed time */
    double stopAndGetElapsed() {
        stop();
        return getElapsed();
    }
};


void printVec(__m512d input, const std::string endOfLine = "\n"){
    double array[8];

    _mm512_storeu_pd(array, input);
    std::cout << " => ";
    for( int idx = 0 ; idx < 8 ; ++idx){
        std::cout << " [" << idx << "] " << array[idx];
    }
    std::cout << endOfLine;
}

void printVec(__m512i input, const std::string endOfLine = "\n"){
    int array[16];

    _mm512_storeu_si512(array, input);
    std::cout << " => ";
    for( int idx = 0 ; idx < 16 ; ++idx){
        std::cout << " [" << idx << "] " << array[idx];
    }
    std::cout << endOfLine;
}

void printMask(__mmask16 mask, const std::string endOfLine = "\n"){
    std::cout << " => ";
    for( int idx = 0 ; idx < 16 ; ++idx){
        std::cout << ((mask & (1 << (15 - idx))) ? "1" : "0");
    }
    std::cout << endOfLine;
}

template <class ObjectClass>
void printArray(const ObjectClass array[], const int size, const std::string endOfLine = "\n"){
    std::cout << " => ";
    for( int idx = 0 ; idx < size ; ++idx){
        std::cout << " [" << idx << "] " << array[idx];
    }
    std::cout << endOfLine;
}

template <class ObjectClass>
void printHist(const ObjectClass array[], const int size, const ObjectClass maxval, const int nbBuckets = 20){
    int hist[nbBuckets] = {0};
    for(int idx =  0 ; idx < size ; ++idx){
        hist[int((array[idx]*nbBuckets)/maxval)]++;
    }
    std::cout << "hist:\n";
    for(int idx = 0 ; idx < nbBuckets ; ++idx){
        std::cout <<"[" << hist[idx] << "]";
    }
    std::cout << "\n";
}

////////////////////////////////////////////////////////////
/// Sort AVX512 Vec Functions
////////////////////////////////////////////////////////////

#include <immintrin.h>

/// Double


inline __m512d SortVecBit(__m512d input){
    {
        __m512d permNeighOdd = _mm512_permute_pd(input, 0x55);
        __mmask8 compMaskOdd = _mm512_cmp_pd_mask(permNeighOdd, input, _CMP_LT_OQ);
        input = _mm512_mask_mov_pd(input, (compMaskOdd & 0x55) | ((compMaskOdd & 0x55)<<1), permNeighOdd);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(4, 5, 6, 7, 0, 1, 2, 3);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        input = _mm512_mask_mov_pd(permNeighMin, 0xCC, permNeighMax);
    }
    {
        __m512d permNeighOdd = _mm512_permute_pd(input, 0x55);
        __mmask8 compMaskOdd = _mm512_cmp_pd_mask(permNeighOdd, input, _CMP_LT_OQ);
        input = _mm512_mask_mov_pd(input, (compMaskOdd & 0x55) | ((compMaskOdd & 0x55)<<1), permNeighOdd);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        input = _mm512_mask_mov_pd(permNeighMin, 0xF0, permNeighMax);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        input = _mm512_mask_mov_pd(permNeighMin, 0xCC, permNeighMax);
    }
    {
        __m512d permNeighOdd = _mm512_permute_pd(input, 0x55);
        __mmask8 compMaskOdd = _mm512_cmp_pd_mask(permNeighOdd, input, _CMP_LT_OQ);
        input = _mm512_mask_mov_pd(input, (compMaskOdd & 0x55) | ((compMaskOdd & 0x55)<<1), permNeighOdd);
    }

    return input;
}

inline void SortVecBit(double* __restrict__ ptr1){
    _mm512_storeu_pd(ptr1, SortVecBit(_mm512_loadu_pd(ptr1)));
}

inline __m512d SortVecBitFull(__m512d input){
    {
        __m512i idxNoNeigh = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        input = _mm512_mask_mov_pd(permNeighMin, 0xAA, permNeighMax);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(4, 5, 6, 7, 0, 1, 2, 3);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        input = _mm512_mask_mov_pd(permNeighMin, 0xCC, permNeighMax);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        input = _mm512_mask_mov_pd(permNeighMin, 0xAA, permNeighMax);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        input = _mm512_mask_mov_pd(permNeighMin, 0xF0, permNeighMax);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        input = _mm512_mask_mov_pd(permNeighMin, 0xCC, permNeighMax);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        input = _mm512_mask_mov_pd(permNeighMin, 0xAA, permNeighMax);
    }

    return input;
}

inline void SortVecBitFull(double* __restrict__ ptr1){
    _mm512_storeu_pd(ptr1, SortVecBitFull(_mm512_loadu_pd(ptr1)));
}



inline __m512d SortVec(__m512d input){
    __m512i idxNoNeigh = _mm512_set_epi64(7, 5, 6, 3, 4, 1, 2, 0);
    for( int idx = 0 ; idx < 4 ; ++idx){
        __m512d permNeighOdd = _mm512_permute_pd(input, 0x55);
        __mmask8 compMaskOdd = _mm512_cmp_pd_mask(permNeighOdd, input, _CMP_LT_OQ);
        input = _mm512_mask_mov_pd(input, (compMaskOdd & 0x55) | ((compMaskOdd & 0x55)<<1), permNeighOdd);

        __m512d permNeighEven = _mm512_permutexvar_pd(idxNoNeigh, input);
        __mmask8 compMaskEven = _mm512_cmp_pd_mask(permNeighEven, input, _CMP_LT_OQ);
        input = _mm512_mask_mov_pd(input, (compMaskEven & 0x2A) | ((compMaskEven & 0x2A)<<1), permNeighEven);
    }
    return input;
}

inline void SortVec(double* __restrict__ ptr1){
    _mm512_storeu_pd(ptr1, SortVec(_mm512_loadu_pd(ptr1)));
}

inline __m512d SortVecWithTest(__m512d input){
    __m512i idxNoNeigh = _mm512_set_epi64(7, 5, 6, 3, 4, 1, 2, 0);
    for( int idx = 0 ; idx < 4 ; ++idx){
        __m512d permNeighOdd = _mm512_permute_pd(input, 0x55);
        __mmask8 compMaskOdd = _mm512_cmp_pd_mask(permNeighOdd, input, _CMP_LT_OQ);
        input = _mm512_mask_mov_pd(input, (compMaskOdd & 0x55) | ((compMaskOdd & 0x55)<<1), permNeighOdd);

        __m512d permNeighEven = _mm512_permutexvar_pd(idxNoNeigh, input);
        __mmask8 compMaskEven = _mm512_cmp_pd_mask(permNeighEven, input, _CMP_LT_OQ);
        input = _mm512_mask_mov_pd(input, (compMaskEven & 0x2A) | ((compMaskEven & 0x2A)<<1), permNeighEven);

        if(compMaskOdd == 0 && compMaskEven == 0){
            break;
        }
    }
    return input;
}

inline void SortVecWithTest(double* __restrict__ ptr1){
    _mm512_storeu_pd(ptr1, SortVecWithTest(_mm512_loadu_pd(ptr1)));
}

inline void Sort2Vec(__m512d& input1, __m512d& input2 ){

    __m512i idxNoNeigh = _mm512_set_epi64(7, 5, 6, 3, 4, 1, 2, 0);
    __m512i idxAll0 = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, 0);
    __m512i idxAll7 = _mm512_set_epi64(7, 7, 7, 7, 7, 7, 7, 7);

    for( int idx = 0 ; idx < 8 ; ++idx){
        __m512d permNeighOdd1 = _mm512_permute_pd(input1, 0x55);
        __m512d permNeighOdd2 = _mm512_permute_pd(input2, 0x55);

        __mmask8 compMaskOdd1 = _mm512_cmp_pd_mask(permNeighOdd1, input1, _CMP_LT_OQ);
        __mmask8 compMaskOdd2 = _mm512_cmp_pd_mask(permNeighOdd2, input2, _CMP_LT_OQ);

        input1 = _mm512_mask_mov_pd(input1, (compMaskOdd1 & 0x55) | ((compMaskOdd1 & 0x55)<<1), permNeighOdd1);
        input2 = _mm512_mask_mov_pd(input2, (compMaskOdd2 & 0x55) | ((compMaskOdd2 & 0x55)<<1), permNeighOdd2);

        __m512d permNeighEven1 = _mm512_permutexvar_pd(idxNoNeigh, input1);
        __m512d permNeighEven2 = _mm512_permutexvar_pd(idxNoNeigh, input2);

        __mmask8 compMaskEven1 = _mm512_cmp_pd_mask(permNeighEven1, input1, _CMP_LT_OQ);
        __mmask8 compMaskEven2 = _mm512_cmp_pd_mask(permNeighEven2, input2, _CMP_LT_OQ);

        input1 = _mm512_mask_mov_pd(input1, (compMaskEven1 & 0x2A) | ((compMaskEven1 & 0x2A)<<1), permNeighEven1);
        input2 = _mm512_mask_mov_pd(input2, (compMaskEven2 & 0x2A) | ((compMaskEven2 & 0x2A)<<1), permNeighEven2);

        // Exchange border if needed
        __m512d input1last = _mm512_permutexvar_pd(idxAll7, input1);
        __m512d input2first = _mm512_permutexvar_pd(idxAll0, input2);

        __mmask8 compDoExchange = _mm512_cmp_pd_mask(input1last, input2first, _CMP_GT_OQ);

        input1 = _mm512_mask_mov_pd(input1, compDoExchange & 0x80, input2first);
        input2 = _mm512_mask_mov_pd(input2, compDoExchange & 1, input1last);
    }
}


inline void Sort2Vec(double* __restrict__ ptr1, double* __restrict__ ptr2 ){
    __m512d input1 = _mm512_loadu_pd(ptr1);
    __m512d input2 = _mm512_loadu_pd(ptr2);
    Sort2Vec(input1, input2);
    _mm512_storeu_pd(ptr1, input1);
    _mm512_storeu_pd(ptr2, input2);
}


inline void ExchangeAndSort(__m512d& input, __m512d& input2){
    {
        __m512i idxNoNeigh = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        input = _mm512_min_pd(input2, permNeigh);
        input2 = _mm512_max_pd(input2, permNeigh);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        input = _mm512_mask_mov_pd(permNeighMin, 0xF0, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xF0, permNeighMax2);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        input = _mm512_mask_mov_pd(permNeighMin, 0xCC, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xCC, permNeighMax2);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        input = _mm512_mask_mov_pd(permNeighMin, 0xAA, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xAA, permNeighMax2);
    }
}

inline void Sort2VecBitFull(__m512d& input, __m512d& input2){
    {
        __m512i idxNoNeigh = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        input = _mm512_mask_mov_pd(permNeighMin, 0xAA, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xAA, permNeighMax2);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(4, 5, 6, 7, 0, 1, 2, 3);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        input = _mm512_mask_mov_pd(permNeighMin, 0xCC, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xCC, permNeighMax2);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        input = _mm512_mask_mov_pd(permNeighMin, 0xAA, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xAA, permNeighMax2);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        input = _mm512_mask_mov_pd(permNeighMin, 0xF0, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xF0, permNeighMax2);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        input = _mm512_mask_mov_pd(permNeighMin, 0xCC, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xCC, permNeighMax2);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        input = _mm512_mask_mov_pd(permNeighMin, 0xAA, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xAA, permNeighMax2);
    }
    ExchangeAndSort(input, input2);
}

inline void Sort2VecBitFull(double* __restrict__ ptr1, double* __restrict__ ptr2 ){
    __m512d input1 = _mm512_loadu_pd(ptr1);
    __m512d input2 = _mm512_loadu_pd(ptr2);
    Sort2VecBitFull(input1, input2);
    _mm512_storeu_pd(ptr1, input1);
    _mm512_storeu_pd(ptr2, input2);
}


inline void Sort3Vec(__m512d& input1, __m512d& input2, __m512d& input3 ){

    __m512i idxNoNeigh = _mm512_set_epi64(7, 5, 6, 3, 4, 1, 2, 0);
    __m512i idxAll0 = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, 0);
    __m512i idxAll7 = _mm512_set_epi64(7, 7, 7, 7, 7, 7, 7, 7);

    for( int idx = 0 ; idx < 12 ; ++idx){
        __m512d permNeighOdd1 = _mm512_permute_pd(input1, 0x55);
        __m512d permNeighOdd2 = _mm512_permute_pd(input2, 0x55);
        __m512d permNeighOdd3 = _mm512_permute_pd(input3, 0x55);

        __mmask8 compMaskOdd1 = _mm512_cmp_pd_mask(permNeighOdd1, input1, _CMP_LT_OQ);
        __mmask8 compMaskOdd2 = _mm512_cmp_pd_mask(permNeighOdd2, input2, _CMP_LT_OQ);
        __mmask8 compMaskOdd3 = _mm512_cmp_pd_mask(permNeighOdd3, input3, _CMP_LT_OQ);

        input1 = _mm512_mask_mov_pd(input1, (compMaskOdd1 & 0x55) | ((compMaskOdd1 & 0x55)<<1), permNeighOdd1);
        input2 = _mm512_mask_mov_pd(input2, (compMaskOdd2 & 0x55) | ((compMaskOdd2 & 0x55)<<1), permNeighOdd2);
        input3 = _mm512_mask_mov_pd(input3, (compMaskOdd3 & 0x55) | ((compMaskOdd3 & 0x55)<<1), permNeighOdd3);

        __m512d permNeighEven1 = _mm512_permutexvar_pd(idxNoNeigh, input1);
        __m512d permNeighEven2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeighEven3 = _mm512_permutexvar_pd(idxNoNeigh, input3);

        __mmask8 compMaskEven1 = _mm512_cmp_pd_mask(permNeighEven1, input1, _CMP_LT_OQ);
        __mmask8 compMaskEven2 = _mm512_cmp_pd_mask(permNeighEven2, input2, _CMP_LT_OQ);
        __mmask8 compMaskEven3 = _mm512_cmp_pd_mask(permNeighEven3, input3, _CMP_LT_OQ);

        input1 = _mm512_mask_mov_pd(input1, (compMaskEven1 & 0x2A) | ((compMaskEven1 & 0x2A)<<1), permNeighEven1);
        input2 = _mm512_mask_mov_pd(input2, (compMaskEven2 & 0x2A) | ((compMaskEven2 & 0x2A)<<1), permNeighEven2);
        input3 = _mm512_mask_mov_pd(input3, (compMaskEven3 & 0x2A) | ((compMaskEven3 & 0x2A)<<1), permNeighEven3);

        // Exchange border if needed
        __m512d input1last = _mm512_permutexvar_pd(idxAll7, input1);
        __m512d input2first = _mm512_permutexvar_pd(idxAll0, input2);

        __mmask8 compDoExchange = _mm512_cmp_pd_mask(input1last, input2first, _CMP_GT_OQ);

        input1 = _mm512_mask_mov_pd(input1, compDoExchange & 0x80, input2first);
        input2 = _mm512_mask_mov_pd(input2, compDoExchange & 1, input1last);

        __m512d input2last = _mm512_permutexvar_pd(idxAll7, input2);
        __m512d input3first = _mm512_permutexvar_pd(idxAll0, input3);

        __mmask8 compDoExchange23 = _mm512_cmp_pd_mask(input2last, input3first, _CMP_GT_OQ);

        input2 = _mm512_mask_mov_pd(input2, compDoExchange23 & 0x80, input3first);
        input3 = _mm512_mask_mov_pd(input3, compDoExchange23 & 1, input2last);
    }
}

inline void Sort3Vec(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3  ){
    __m512d input1 = _mm512_loadu_pd(ptr1);
    __m512d input2 = _mm512_loadu_pd(ptr2);
    __m512d input3 = _mm512_loadu_pd(ptr3);
    Sort3Vec(input1, input2, input3);
    _mm512_storeu_pd(ptr1, input1);
    _mm512_storeu_pd(ptr2, input2);
    _mm512_storeu_pd(ptr3, input3);
}

inline void Sort3VecBitFull(__m512d& input, __m512d& input2, __m512d& input3 ){
    Sort2VecBitFull(input, input2);
    input3 = SortVecBitFull(input3);
    {
        __m512i idxNoNeigh = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input3);
        input3 = _mm512_max_pd(input2, permNeigh);
        input2 = _mm512_min_pd(input2, permNeigh);
    }
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input2, inputCopy);
        input2 = _mm512_max_pd(input2, inputCopy);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        input = _mm512_mask_mov_pd(permNeighMin, 0xF0, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xF0, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xF0, permNeighMax3);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        input = _mm512_mask_mov_pd(permNeighMin, 0xCC, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xCC, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xCC, permNeighMax3);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        input = _mm512_mask_mov_pd(permNeighMin, 0xAA, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xAA, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xAA, permNeighMax3);
    }
}

inline void Sort3VecBitFull(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3  ){
    __m512d input1 = _mm512_loadu_pd(ptr1);
    __m512d input2 = _mm512_loadu_pd(ptr2);
    __m512d input3 = _mm512_loadu_pd(ptr3);
    Sort3VecBitFull(input1, input2, input3);
    _mm512_storeu_pd(ptr1, input1);
    _mm512_storeu_pd(ptr2, input2);
    _mm512_storeu_pd(ptr3, input3);
}

inline void Sort4Vec(__m512d& input1, __m512d& input2, __m512d& input3, __m512d& input4 ){

    __m512i idxNoNeigh = _mm512_set_epi64(7, 5, 6, 3, 4, 1, 2, 0);
    __m512i idxAll0 = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, 0);
    __m512i idxAll7 = _mm512_set_epi64(7, 7, 7, 7, 7, 7, 7, 7);

    for( int idx = 0 ; idx < 16 ; ++idx){
        __m512d permNeighOdd1 = _mm512_permute_pd(input1, 0x55);
        __m512d permNeighOdd2 = _mm512_permute_pd(input2, 0x55);
        __m512d permNeighOdd3 = _mm512_permute_pd(input3, 0x55);
        __m512d permNeighOdd4 = _mm512_permute_pd(input4, 0x55);

        __mmask8 compMaskOdd1 = _mm512_cmp_pd_mask(permNeighOdd1, input1, _CMP_LT_OQ);
        __mmask8 compMaskOdd2 = _mm512_cmp_pd_mask(permNeighOdd2, input2, _CMP_LT_OQ);
        __mmask8 compMaskOdd3 = _mm512_cmp_pd_mask(permNeighOdd3, input3, _CMP_LT_OQ);
        __mmask8 compMaskOdd4 = _mm512_cmp_pd_mask(permNeighOdd4, input4, _CMP_LT_OQ);

        input1 = _mm512_mask_mov_pd(input1, (compMaskOdd1 & 0x55) | ((compMaskOdd1 & 0x55)<<1), permNeighOdd1);
        input2 = _mm512_mask_mov_pd(input2, (compMaskOdd2 & 0x55) | ((compMaskOdd2 & 0x55)<<1), permNeighOdd2);
        input3 = _mm512_mask_mov_pd(input3, (compMaskOdd3 & 0x55) | ((compMaskOdd3 & 0x55)<<1), permNeighOdd3);
        input4 = _mm512_mask_mov_pd(input4, (compMaskOdd4 & 0x55) | ((compMaskOdd4 & 0x55)<<1), permNeighOdd4);

        __m512d permNeighEven1 = _mm512_permutexvar_pd(idxNoNeigh, input1);
        __m512d permNeighEven2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeighEven3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeighEven4 = _mm512_permutexvar_pd(idxNoNeigh, input4);

        __mmask8 compMaskEven1 = _mm512_cmp_pd_mask(permNeighEven1, input1, _CMP_LT_OQ);
        __mmask8 compMaskEven2 = _mm512_cmp_pd_mask(permNeighEven2, input2, _CMP_LT_OQ);
        __mmask8 compMaskEven3 = _mm512_cmp_pd_mask(permNeighEven3, input3, _CMP_LT_OQ);
        __mmask8 compMaskEven4 = _mm512_cmp_pd_mask(permNeighEven4, input4, _CMP_LT_OQ);

        input1 = _mm512_mask_mov_pd(input1, (compMaskEven1 & 0x2A) | ((compMaskEven1 & 0x2A)<<1), permNeighEven1);
        input2 = _mm512_mask_mov_pd(input2, (compMaskEven2 & 0x2A) | ((compMaskEven2 & 0x2A)<<1), permNeighEven2);
        input3 = _mm512_mask_mov_pd(input3, (compMaskEven3 & 0x2A) | ((compMaskEven3 & 0x2A)<<1), permNeighEven3);
        input4 = _mm512_mask_mov_pd(input4, (compMaskEven4 & 0x2A) | ((compMaskEven4 & 0x2A)<<1), permNeighEven4);

        // Exchange border if needed
        __m512d input1last = _mm512_permutexvar_pd(idxAll7, input1);
        __m512d input2first = _mm512_permutexvar_pd(idxAll0, input2);

        __mmask8 compDoExchange = _mm512_cmp_pd_mask(input1last, input2first, _CMP_GT_OQ);

        input1 = _mm512_mask_mov_pd(input1, compDoExchange & 0x80, input2first);
        input2 = _mm512_mask_mov_pd(input2, compDoExchange & 1, input1last);

        __m512d input2last = _mm512_permutexvar_pd(idxAll7, input2);
        __m512d input3first = _mm512_permutexvar_pd(idxAll0, input3);

        __mmask8 compDoExchange23 = _mm512_cmp_pd_mask(input2last, input3first, _CMP_GT_OQ);

        input2 = _mm512_mask_mov_pd(input2, compDoExchange23 & 0x80, input3first);
        input3 = _mm512_mask_mov_pd(input3, compDoExchange23 & 1, input2last);

        __m512d input3last = _mm512_permutexvar_pd(idxAll7, input3);
        __m512d input4first = _mm512_permutexvar_pd(idxAll0, input4);

        __mmask8 compDoExchange34 = _mm512_cmp_pd_mask(input3last, input4first, _CMP_GT_OQ);

        input3 = _mm512_mask_mov_pd(input3, compDoExchange34 & 0x80, input4first);
        input4 = _mm512_mask_mov_pd(input4, compDoExchange34 & 1, input3last);
    }
}

inline void Sort4Vec(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3, double* __restrict__ ptr4  ){
    __m512d input1 = _mm512_loadu_pd(ptr1);
    __m512d input2 = _mm512_loadu_pd(ptr2);
    __m512d input3 = _mm512_loadu_pd(ptr3);
    __m512d input4 = _mm512_loadu_pd(ptr4);
    Sort4Vec(input1, input2, input3, input4);
    _mm512_storeu_pd(ptr1, input1);
    _mm512_storeu_pd(ptr2, input2);
    _mm512_storeu_pd(ptr3, input3);
    _mm512_storeu_pd(ptr4, input4);
}


inline void Sort4VecBitFull(__m512d& input, __m512d& input2, __m512d& input3, __m512d& input4 ){
    Sort2VecBitFull(input, input2);
    Sort2VecBitFull(input3, input4);
    {
        __m512i idxNoNeigh = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);

        input4 = _mm512_max_pd(input, permNeigh4);
        input = _mm512_min_pd(input, permNeigh4);

        input3 = _mm512_max_pd(input2, permNeigh3);
        input2 = _mm512_min_pd(input2, permNeigh3);
    }
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input2, inputCopy);
        input2 = _mm512_max_pd(input2, inputCopy);
    }
    {
        __m512d inputCopy = input3;
        input3 = _mm512_min_pd(input4, inputCopy);
        input4 = _mm512_max_pd(input4, inputCopy);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        input = _mm512_mask_mov_pd(permNeighMin, 0xF0, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xF0, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xF0, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xF0, permNeighMax4);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        input = _mm512_mask_mov_pd(permNeighMin, 0xCC, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xCC, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xCC, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xCC, permNeighMax4);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        input = _mm512_mask_mov_pd(permNeighMin, 0xAA, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xAA, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xAA, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xAA, permNeighMax4);
    }
}


inline void Sort4VecBitFull(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3, double* __restrict__ ptr4  ){
    __m512d input1 = _mm512_loadu_pd(ptr1);
    __m512d input2 = _mm512_loadu_pd(ptr2);
    __m512d input3 = _mm512_loadu_pd(ptr3);
    __m512d input4 = _mm512_loadu_pd(ptr4);
    Sort4VecBitFull(input1, input2, input3, input4);
    _mm512_storeu_pd(ptr1, input1);
    _mm512_storeu_pd(ptr2, input2);
    _mm512_storeu_pd(ptr3, input3);
    _mm512_storeu_pd(ptr4, input4);
}


inline void Sort5VecBitFull(__m512d& input, __m512d& input2, __m512d& input3, __m512d& input4, __m512d& input5 ){
    Sort4VecBitFull(input, input2, input3, input4);
    input5 = SortVecBitFull(input5);
//    std::cout << "input " ; printVec(input);
//    std::cout << "input2 " ; printVec(input2);
//    std::cout << "input3 " ; printVec(input3);
//    std::cout << "input4 " ; printVec(input4);
//    std::cout << "input5 " ; printVec(input5);
    {
        __m512i idxNoNeigh = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);

        input5 = _mm512_max_pd(input4, permNeigh5);
        input4 = _mm512_min_pd(input4, permNeigh5);
    }
//    std::cout << "input " ; printVec(input);
//    std::cout << "input2 " ; printVec(input2);
//    std::cout << "input3 " ; printVec(input3);
//    std::cout << "input4 " ; printVec(input4);
//    std::cout << "input5 " ; printVec(input5);
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input3, inputCopy);
        input3 = _mm512_max_pd(input3, inputCopy);
    }
//    std::cout << "input " ; printVec(input);
//    std::cout << "input2 " ; printVec(input2);
//    std::cout << "input3 " ; printVec(input3);
//    std::cout << "input4 " ; printVec(input4);
//    std::cout << "input5 " ; printVec(input5);
    {
        __m512d inputCopy = input2;
        input2 = _mm512_min_pd(input4, inputCopy);
        input4 = _mm512_max_pd(input4, inputCopy);
    }
//    std::cout << "input " ; printVec(input);
//    std::cout << "input2 " ; printVec(input2);
//    std::cout << "input3 " ; printVec(input3);
//    std::cout << "input4 " ; printVec(input4);
//    std::cout << "input5 " ; printVec(input5);
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input2, inputCopy);
        input2 = _mm512_max_pd(input2, inputCopy);
    }
//    std::cout << "input " ; printVec(input);
//    std::cout << "input2 " ; printVec(input2);
//    std::cout << "input3 " ; printVec(input3);
//    std::cout << "input4 " ; printVec(input4);
//    std::cout << "input5 " ; printVec(input5);
    {
        __m512d inputCopy = input3;
        input3 = _mm512_min_pd(input4, inputCopy);
        input4 = _mm512_max_pd(input4, inputCopy);
    }
//    std::cout << "input " ; printVec(input);
//    std::cout << "input2 " ; printVec(input2);
//    std::cout << "input3 " ; printVec(input3);
//    std::cout << "input4 " ; printVec(input4);
//    std::cout << "input5 " ; printVec(input5);
    {
        __m512i idxNoNeigh = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMin5 = _mm512_min_pd(permNeigh5, input5);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        __m512d permNeighMax5 = _mm512_max_pd(permNeigh5, input5);
        input = _mm512_mask_mov_pd(permNeighMin, 0xF0, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xF0, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xF0, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xF0, permNeighMax4);
        input5 = _mm512_mask_mov_pd(permNeighMin5, 0xF0, permNeighMax5);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMin5 = _mm512_min_pd(permNeigh5, input5);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        __m512d permNeighMax5 = _mm512_max_pd(permNeigh5, input5);
        input = _mm512_mask_mov_pd(permNeighMin, 0xCC, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xCC, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xCC, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xCC, permNeighMax4);
        input5 = _mm512_mask_mov_pd(permNeighMin5, 0xCC, permNeighMax5);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMin5 = _mm512_min_pd(permNeigh5, input5);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        __m512d permNeighMax5 = _mm512_max_pd(permNeigh5, input5);
        input = _mm512_mask_mov_pd(permNeighMin, 0xAA, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xAA, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xAA, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xAA, permNeighMax4);
        input5 = _mm512_mask_mov_pd(permNeighMin5, 0xAA, permNeighMax5);
    }
}


inline void Sort5VecBitFull(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3,
                            double* __restrict__ ptr4, double* __restrict__ ptr5 ){
    __m512d input1 = _mm512_loadu_pd(ptr1);
    __m512d input2 = _mm512_loadu_pd(ptr2);
    __m512d input3 = _mm512_loadu_pd(ptr3);
    __m512d input4 = _mm512_loadu_pd(ptr4);
    __m512d input5 = _mm512_loadu_pd(ptr5);
    Sort5VecBitFull(input1, input2, input3, input4, input5);
    _mm512_storeu_pd(ptr1, input1);
    _mm512_storeu_pd(ptr2, input2);
    _mm512_storeu_pd(ptr3, input3);
    _mm512_storeu_pd(ptr4, input4);
    _mm512_storeu_pd(ptr5, input5);
}


inline void Sort6VecBitFull(__m512d& input, __m512d& input2, __m512d& input3, __m512d& input4, __m512d& input5, __m512d& input6 ){
    Sort4VecBitFull(input, input2, input3, input4);
    Sort2VecBitFull(input5, input6);
    {
        __m512i idxNoNeigh = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeigh6 = _mm512_permutexvar_pd(idxNoNeigh, input6);

        input5 = _mm512_max_pd(input4, permNeigh5);
        input6 = _mm512_max_pd(input3, permNeigh6);

        input4 = _mm512_min_pd(input4, permNeigh5);
        input3 = _mm512_min_pd(input3, permNeigh6);
    }
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input3, inputCopy);
        input3 = _mm512_max_pd(input3, inputCopy);
    }
    {
        __m512d inputCopy = input2;
        input2 = _mm512_min_pd(input4, inputCopy);
        input4 = _mm512_max_pd(input4, inputCopy);
    }
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input2, inputCopy);
        input2 = _mm512_max_pd(input2, inputCopy);
    }
    {
        __m512d inputCopy = input3;
        input3 = _mm512_min_pd(input4, inputCopy);
        input4 = _mm512_max_pd(input4, inputCopy);
    }
    {
        __m512d inputCopy = input5;
        input5 = _mm512_min_pd(input6, inputCopy);
        input6 = _mm512_max_pd(input6, inputCopy);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeigh6 = _mm512_permutexvar_pd(idxNoNeigh, input6);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMin5 = _mm512_min_pd(permNeigh5, input5);
        __m512d permNeighMin6 = _mm512_min_pd(permNeigh6, input6);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        __m512d permNeighMax5 = _mm512_max_pd(permNeigh5, input5);
        __m512d permNeighMax6 = _mm512_max_pd(permNeigh6, input6);
        input = _mm512_mask_mov_pd(permNeighMin, 0xF0, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xF0, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xF0, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xF0, permNeighMax4);
        input5 = _mm512_mask_mov_pd(permNeighMin5, 0xF0, permNeighMax5);
        input6 = _mm512_mask_mov_pd(permNeighMin6, 0xF0, permNeighMax6);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeigh6 = _mm512_permutexvar_pd(idxNoNeigh, input6);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMin5 = _mm512_min_pd(permNeigh5, input5);
        __m512d permNeighMin6 = _mm512_min_pd(permNeigh6, input6);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        __m512d permNeighMax5 = _mm512_max_pd(permNeigh5, input5);
        __m512d permNeighMax6 = _mm512_max_pd(permNeigh6, input6);
        input = _mm512_mask_mov_pd(permNeighMin, 0xCC, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xCC, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xCC, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xCC, permNeighMax4);
        input5 = _mm512_mask_mov_pd(permNeighMin5, 0xCC, permNeighMax5);
        input6 = _mm512_mask_mov_pd(permNeighMin6, 0xCC, permNeighMax6);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeigh6 = _mm512_permutexvar_pd(idxNoNeigh, input6);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMin5 = _mm512_min_pd(permNeigh5, input5);
        __m512d permNeighMin6 = _mm512_min_pd(permNeigh6, input6);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        __m512d permNeighMax5 = _mm512_max_pd(permNeigh5, input5);
        __m512d permNeighMax6 = _mm512_max_pd(permNeigh6, input6);
        input = _mm512_mask_mov_pd(permNeighMin, 0xAA, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xAA, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xAA, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xAA, permNeighMax4);
        input5 = _mm512_mask_mov_pd(permNeighMin5, 0xAA, permNeighMax5);
        input6 = _mm512_mask_mov_pd(permNeighMin6, 0xAA, permNeighMax6);
    }
}


inline void Sort6VecBitFull(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3,
                            double* __restrict__ ptr4, double* __restrict__ ptr5, double* __restrict__ ptr6 ){
    __m512d input1 = _mm512_loadu_pd(ptr1);
    __m512d input2 = _mm512_loadu_pd(ptr2);
    __m512d input3 = _mm512_loadu_pd(ptr3);
    __m512d input4 = _mm512_loadu_pd(ptr4);
    __m512d input5 = _mm512_loadu_pd(ptr5);
    __m512d input6 = _mm512_loadu_pd(ptr6);
    Sort6VecBitFull(input1, input2, input3, input4, input5, input6);
    _mm512_storeu_pd(ptr1, input1);
    _mm512_storeu_pd(ptr2, input2);
    _mm512_storeu_pd(ptr3, input3);
    _mm512_storeu_pd(ptr4, input4);
    _mm512_storeu_pd(ptr5, input5);
    _mm512_storeu_pd(ptr6, input6);
}


inline void Sort7VecBitFull(__m512d& input, __m512d& input2, __m512d& input3, __m512d& input4,
                            __m512d& input5, __m512d& input6, __m512d& input7 ){
    Sort4VecBitFull(input, input2, input3, input4);
    Sort3VecBitFull(input5, input6, input7);
    {
        __m512i idxNoNeigh = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeigh6 = _mm512_permutexvar_pd(idxNoNeigh, input6);
        __m512d permNeigh7 = _mm512_permutexvar_pd(idxNoNeigh, input7);

        input5 = _mm512_max_pd(input4, permNeigh5);
        input6 = _mm512_max_pd(input3, permNeigh6);
        input7 = _mm512_max_pd(input2, permNeigh7);

        input4 = _mm512_min_pd(input4, permNeigh5);
        input3 = _mm512_min_pd(input3, permNeigh6);
        input2 = _mm512_min_pd(input2, permNeigh7);
    }
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input3, inputCopy);
        input3 = _mm512_max_pd(input3, inputCopy);
    }
    {
        __m512d inputCopy = input2;
        input2 = _mm512_min_pd(input4, inputCopy);
        input4 = _mm512_max_pd(input4, inputCopy);
    }
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input2, inputCopy);
        input2 = _mm512_max_pd(input2, inputCopy);
    }
    {
        __m512d inputCopy = input3;
        input3 = _mm512_min_pd(input4, inputCopy);
        input4 = _mm512_max_pd(input4, inputCopy);
    }
    {
        __m512d inputCopy = input5;
        input5 = _mm512_min_pd(input7, inputCopy);
        input7 = _mm512_max_pd(input7, inputCopy);
    }
    {
        __m512d inputCopy = input5;
        input5 = _mm512_min_pd(input6, inputCopy);
        input6 = _mm512_max_pd(input6, inputCopy);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeigh6 = _mm512_permutexvar_pd(idxNoNeigh, input6);
        __m512d permNeigh7 = _mm512_permutexvar_pd(idxNoNeigh, input7);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMin5 = _mm512_min_pd(permNeigh5, input5);
        __m512d permNeighMin6 = _mm512_min_pd(permNeigh6, input6);
        __m512d permNeighMin7 = _mm512_min_pd(permNeigh7, input7);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        __m512d permNeighMax5 = _mm512_max_pd(permNeigh5, input5);
        __m512d permNeighMax6 = _mm512_max_pd(permNeigh6, input6);
        __m512d permNeighMax7 = _mm512_max_pd(permNeigh7, input7);
        input = _mm512_mask_mov_pd(permNeighMin, 0xF0, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xF0, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xF0, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xF0, permNeighMax4);
        input5 = _mm512_mask_mov_pd(permNeighMin5, 0xF0, permNeighMax5);
        input6 = _mm512_mask_mov_pd(permNeighMin6, 0xF0, permNeighMax6);
        input7 = _mm512_mask_mov_pd(permNeighMin7, 0xF0, permNeighMax7);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeigh6 = _mm512_permutexvar_pd(idxNoNeigh, input6);
        __m512d permNeigh7 = _mm512_permutexvar_pd(idxNoNeigh, input7);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMin5 = _mm512_min_pd(permNeigh5, input5);
        __m512d permNeighMin6 = _mm512_min_pd(permNeigh6, input6);
        __m512d permNeighMin7 = _mm512_min_pd(permNeigh7, input7);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        __m512d permNeighMax5 = _mm512_max_pd(permNeigh5, input5);
        __m512d permNeighMax6 = _mm512_max_pd(permNeigh6, input6);
        __m512d permNeighMax7 = _mm512_max_pd(permNeigh7, input7);
        input = _mm512_mask_mov_pd(permNeighMin, 0xCC, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xCC, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xCC, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xCC, permNeighMax4);
        input5 = _mm512_mask_mov_pd(permNeighMin5, 0xCC, permNeighMax5);
        input6 = _mm512_mask_mov_pd(permNeighMin6, 0xCC, permNeighMax6);
        input7 = _mm512_mask_mov_pd(permNeighMin7, 0xCC, permNeighMax7);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeigh6 = _mm512_permutexvar_pd(idxNoNeigh, input6);
        __m512d permNeigh7 = _mm512_permutexvar_pd(idxNoNeigh, input7);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMin5 = _mm512_min_pd(permNeigh5, input5);
        __m512d permNeighMin6 = _mm512_min_pd(permNeigh6, input6);
        __m512d permNeighMin7 = _mm512_min_pd(permNeigh7, input7);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        __m512d permNeighMax5 = _mm512_max_pd(permNeigh5, input5);
        __m512d permNeighMax6 = _mm512_max_pd(permNeigh6, input6);
        __m512d permNeighMax7 = _mm512_max_pd(permNeigh7, input7);
        input = _mm512_mask_mov_pd(permNeighMin, 0xAA, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xAA, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xAA, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xAA, permNeighMax4);
        input5 = _mm512_mask_mov_pd(permNeighMin5, 0xAA, permNeighMax5);
        input6 = _mm512_mask_mov_pd(permNeighMin6, 0xAA, permNeighMax6);
        input7 = _mm512_mask_mov_pd(permNeighMin7, 0xAA, permNeighMax7);
    }
}


inline void Sort7VecBitFull(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3,
                            double* __restrict__ ptr4, double* __restrict__ ptr5, double* __restrict__ ptr6,
                            double* __restrict__ ptr7){
    __m512d input1 = _mm512_loadu_pd(ptr1);
    __m512d input2 = _mm512_loadu_pd(ptr2);
    __m512d input3 = _mm512_loadu_pd(ptr3);
    __m512d input4 = _mm512_loadu_pd(ptr4);
    __m512d input5 = _mm512_loadu_pd(ptr5);
    __m512d input6 = _mm512_loadu_pd(ptr6);
    __m512d input7 = _mm512_loadu_pd(ptr7);
    Sort7VecBitFull(input1, input2, input3, input4, input5, input6, input7);
    _mm512_storeu_pd(ptr1, input1);
    _mm512_storeu_pd(ptr2, input2);
    _mm512_storeu_pd(ptr3, input3);
    _mm512_storeu_pd(ptr4, input4);
    _mm512_storeu_pd(ptr5, input5);
    _mm512_storeu_pd(ptr6, input6);
    _mm512_storeu_pd(ptr7, input7);
}


inline void Sort8VecBitFull(__m512d& input, __m512d& input2, __m512d& input3, __m512d& input4,
                            __m512d& input5, __m512d& input6, __m512d& input7, __m512d& input8 ){
    Sort4VecBitFull(input, input2, input3, input4);
    Sort4VecBitFull(input5, input6, input7, input8);
    {
        __m512i idxNoNeigh = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeigh6 = _mm512_permutexvar_pd(idxNoNeigh, input6);
        __m512d permNeigh7 = _mm512_permutexvar_pd(idxNoNeigh, input7);
        __m512d permNeigh8 = _mm512_permutexvar_pd(idxNoNeigh, input8);

        input5 = _mm512_max_pd(input4, permNeigh5);
        input6 = _mm512_max_pd(input3, permNeigh6);
        input7 = _mm512_max_pd(input2, permNeigh7);
        input8 = _mm512_max_pd(input, permNeigh8);

        input4 = _mm512_min_pd(input4, permNeigh5);
        input3 = _mm512_min_pd(input3, permNeigh6);
        input2 = _mm512_min_pd(input2, permNeigh7);
        input = _mm512_min_pd(input, permNeigh8);
    }
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input3, inputCopy);
        input3 = _mm512_max_pd(input3, inputCopy);
    }
    {
        __m512d inputCopy = input2;
        input2 = _mm512_min_pd(input4, inputCopy);
        input4 = _mm512_max_pd(input4, inputCopy);
    }
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input2, inputCopy);
        input2 = _mm512_max_pd(input2, inputCopy);
    }
    {
        __m512d inputCopy = input3;
        input3 = _mm512_min_pd(input4, inputCopy);
        input4 = _mm512_max_pd(input4, inputCopy);
    }
    {
        __m512d inputCopy = input5;
        input5 = _mm512_min_pd(input7, inputCopy);
        input7 = _mm512_max_pd(input7, inputCopy);
    }
    {
        __m512d inputCopy = input6;
        input6 = _mm512_min_pd(input8, inputCopy);
        input8 = _mm512_max_pd(input8, inputCopy);
    }
    {
        __m512d inputCopy = input5;
        input5 = _mm512_min_pd(input6, inputCopy);
        input6 = _mm512_max_pd(input6, inputCopy);
    }
    {
        __m512d inputCopy = input7;
        input7 = _mm512_min_pd(input8, inputCopy);
        input8 = _mm512_max_pd(input8, inputCopy);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeigh6 = _mm512_permutexvar_pd(idxNoNeigh, input6);
        __m512d permNeigh7 = _mm512_permutexvar_pd(idxNoNeigh, input7);
        __m512d permNeigh8 = _mm512_permutexvar_pd(idxNoNeigh, input8);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMin5 = _mm512_min_pd(permNeigh5, input5);
        __m512d permNeighMin6 = _mm512_min_pd(permNeigh6, input6);
        __m512d permNeighMin7 = _mm512_min_pd(permNeigh7, input7);
        __m512d permNeighMin8 = _mm512_min_pd(permNeigh8, input8);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        __m512d permNeighMax5 = _mm512_max_pd(permNeigh5, input5);
        __m512d permNeighMax6 = _mm512_max_pd(permNeigh6, input6);
        __m512d permNeighMax7 = _mm512_max_pd(permNeigh7, input7);
        __m512d permNeighMax8 = _mm512_max_pd(permNeigh8, input8);
        input = _mm512_mask_mov_pd(permNeighMin, 0xF0, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xF0, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xF0, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xF0, permNeighMax4);
        input5 = _mm512_mask_mov_pd(permNeighMin5, 0xF0, permNeighMax5);
        input6 = _mm512_mask_mov_pd(permNeighMin6, 0xF0, permNeighMax6);
        input7 = _mm512_mask_mov_pd(permNeighMin7, 0xF0, permNeighMax7);
        input8 = _mm512_mask_mov_pd(permNeighMin8, 0xF0, permNeighMax8);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeigh6 = _mm512_permutexvar_pd(idxNoNeigh, input6);
        __m512d permNeigh7 = _mm512_permutexvar_pd(idxNoNeigh, input7);
        __m512d permNeigh8 = _mm512_permutexvar_pd(idxNoNeigh, input8);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMin5 = _mm512_min_pd(permNeigh5, input5);
        __m512d permNeighMin6 = _mm512_min_pd(permNeigh6, input6);
        __m512d permNeighMin7 = _mm512_min_pd(permNeigh7, input7);
        __m512d permNeighMin8 = _mm512_min_pd(permNeigh8, input8);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        __m512d permNeighMax5 = _mm512_max_pd(permNeigh5, input5);
        __m512d permNeighMax6 = _mm512_max_pd(permNeigh6, input6);
        __m512d permNeighMax7 = _mm512_max_pd(permNeigh7, input7);
        __m512d permNeighMax8 = _mm512_max_pd(permNeigh8, input8);
        input = _mm512_mask_mov_pd(permNeighMin, 0xCC, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xCC, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xCC, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xCC, permNeighMax4);
        input5 = _mm512_mask_mov_pd(permNeighMin5, 0xCC, permNeighMax5);
        input6 = _mm512_mask_mov_pd(permNeighMin6, 0xCC, permNeighMax6);
        input7 = _mm512_mask_mov_pd(permNeighMin7, 0xCC, permNeighMax7);
        input8 = _mm512_mask_mov_pd(permNeighMin8, 0xCC, permNeighMax8);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeigh6 = _mm512_permutexvar_pd(idxNoNeigh, input6);
        __m512d permNeigh7 = _mm512_permutexvar_pd(idxNoNeigh, input7);
        __m512d permNeigh8 = _mm512_permutexvar_pd(idxNoNeigh, input8);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMin5 = _mm512_min_pd(permNeigh5, input5);
        __m512d permNeighMin6 = _mm512_min_pd(permNeigh6, input6);
        __m512d permNeighMin7 = _mm512_min_pd(permNeigh7, input7);
        __m512d permNeighMin8 = _mm512_min_pd(permNeigh8, input8);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        __m512d permNeighMax5 = _mm512_max_pd(permNeigh5, input5);
        __m512d permNeighMax6 = _mm512_max_pd(permNeigh6, input6);
        __m512d permNeighMax7 = _mm512_max_pd(permNeigh7, input7);
        __m512d permNeighMax8 = _mm512_max_pd(permNeigh8, input8);
        input = _mm512_mask_mov_pd(permNeighMin, 0xAA, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xAA, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xAA, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xAA, permNeighMax4);
        input5 = _mm512_mask_mov_pd(permNeighMin5, 0xAA, permNeighMax5);
        input6 = _mm512_mask_mov_pd(permNeighMin6, 0xAA, permNeighMax6);
        input7 = _mm512_mask_mov_pd(permNeighMin7, 0xAA, permNeighMax7);
        input8 = _mm512_mask_mov_pd(permNeighMin8, 0xAA, permNeighMax8);
    }
}



inline void Sort8VecBitFull(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3,
                            double* __restrict__ ptr4, double* __restrict__ ptr5, double* __restrict__ ptr6,
                            double* __restrict__ ptr7, double* __restrict__ ptr8 ){
    __m512d input1 = _mm512_loadu_pd(ptr1);
    __m512d input2 = _mm512_loadu_pd(ptr2);
    __m512d input3 = _mm512_loadu_pd(ptr3);
    __m512d input4 = _mm512_loadu_pd(ptr4);
    __m512d input5 = _mm512_loadu_pd(ptr5);
    __m512d input6 = _mm512_loadu_pd(ptr6);
    __m512d input7 = _mm512_loadu_pd(ptr7);
    __m512d input8 = _mm512_loadu_pd(ptr8);
    Sort8VecBitFull(input1, input2, input3, input4, input5, input6, input7, input8);
    _mm512_storeu_pd(ptr1, input1);
    _mm512_storeu_pd(ptr2, input2);
    _mm512_storeu_pd(ptr3, input3);
    _mm512_storeu_pd(ptr4, input4);
    _mm512_storeu_pd(ptr5, input5);
    _mm512_storeu_pd(ptr6, input6);
    _mm512_storeu_pd(ptr7, input7);
    _mm512_storeu_pd(ptr8, input8);
}


inline void Finish1VecBitFull(__m512d& input){
    {
        __m512i idxNoNeigh = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        input = _mm512_mask_mov_pd(permNeighMin, 0xF0, permNeighMax);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        input = _mm512_mask_mov_pd(permNeighMin, 0xCC, permNeighMax);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        input = _mm512_mask_mov_pd(permNeighMin, 0xAA, permNeighMax);
    }
}

inline void Finish2VecBitFull(__m512d& input, __m512d& input2){
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input2, inputCopy);
        input2 = _mm512_max_pd(input2, inputCopy);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        input = _mm512_mask_mov_pd(permNeighMin, 0xF0, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xF0, permNeighMax2);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        input = _mm512_mask_mov_pd(permNeighMin, 0xCC, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xCC, permNeighMax2);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        input = _mm512_mask_mov_pd(permNeighMin, 0xAA, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xAA, permNeighMax2);
    }
}

inline void Finish3VecBitFull(__m512d& input, __m512d& input2, __m512d& input3){
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input3, inputCopy);
        input3 = _mm512_max_pd(input3, inputCopy);
    }
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input2, inputCopy);
        input2 = _mm512_max_pd(input2, inputCopy);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        input = _mm512_mask_mov_pd(permNeighMin, 0xF0, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xF0, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xF0, permNeighMax3);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        input = _mm512_mask_mov_pd(permNeighMin, 0xCC, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xCC, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xCC, permNeighMax3);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        input = _mm512_mask_mov_pd(permNeighMin, 0xAA, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xAA, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xAA, permNeighMax3);
    }
}

inline void Finish4VecBitFull(__m512d& input, __m512d& input2, __m512d& input3, __m512d& input4){
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input3, inputCopy);
        input3 = _mm512_max_pd(input3, inputCopy);
    }
    {
        __m512d inputCopy = input2;
        input2 = _mm512_min_pd(input4, inputCopy);
        input4 = _mm512_max_pd(input4, inputCopy);
    }
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input2, inputCopy);
        input2 = _mm512_max_pd(input2, inputCopy);
    }
    {
        __m512d inputCopy = input3;
        input3 = _mm512_min_pd(input4, inputCopy);
        input4 = _mm512_max_pd(input4, inputCopy);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        input = _mm512_mask_mov_pd(permNeighMin, 0xF0, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xF0, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xF0, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xF0, permNeighMax4);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        input = _mm512_mask_mov_pd(permNeighMin, 0xCC, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xCC, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xCC, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xCC, permNeighMax4);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        input = _mm512_mask_mov_pd(permNeighMin, 0xAA, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xAA, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xAA, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xAA, permNeighMax4);
    }
}

inline void Finish5VecBitFull(__m512d& input, __m512d& input2, __m512d& input3, __m512d& input4,
                            __m512d& input5){
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input5, inputCopy);
        input5 = _mm512_max_pd(input5, inputCopy);
    }
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input3, inputCopy);
        input3 = _mm512_max_pd(input3, inputCopy);
    }
    {
        __m512d inputCopy = input2;
        input2 = _mm512_min_pd(input4, inputCopy);
        input4 = _mm512_max_pd(input4, inputCopy);
    }
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input2, inputCopy);
        input2 = _mm512_max_pd(input2, inputCopy);
    }
    {
        __m512d inputCopy = input3;
        input3 = _mm512_min_pd(input4, inputCopy);
        input4 = _mm512_max_pd(input4, inputCopy);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMin5 = _mm512_min_pd(permNeigh5, input5);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        __m512d permNeighMax5 = _mm512_max_pd(permNeigh5, input5);
        input = _mm512_mask_mov_pd(permNeighMin, 0xF0, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xF0, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xF0, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xF0, permNeighMax4);
        input5 = _mm512_mask_mov_pd(permNeighMin5, 0xF0, permNeighMax5);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMin5 = _mm512_min_pd(permNeigh5, input5);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        __m512d permNeighMax5 = _mm512_max_pd(permNeigh5, input5);
        input = _mm512_mask_mov_pd(permNeighMin, 0xCC, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xCC, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xCC, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xCC, permNeighMax4);
        input5 = _mm512_mask_mov_pd(permNeighMin5, 0xCC, permNeighMax5);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMin5 = _mm512_min_pd(permNeigh5, input5);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        __m512d permNeighMax5 = _mm512_max_pd(permNeigh5, input5);
        input = _mm512_mask_mov_pd(permNeighMin, 0xAA, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xAA, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xAA, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xAA, permNeighMax4);
        input5 = _mm512_mask_mov_pd(permNeighMin5, 0xAA, permNeighMax5);
    }
}

inline void Finish6VecBitFull(__m512d& input, __m512d& input2, __m512d& input3, __m512d& input4,
                            __m512d& input5, __m512d& input6){
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input5, inputCopy);
        input5 = _mm512_max_pd(input5, inputCopy);
    }
    {
        __m512d inputCopy = input2;
        input2 = _mm512_min_pd(input6, inputCopy);
        input6 = _mm512_max_pd(input6, inputCopy);
    }
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input3, inputCopy);
        input3 = _mm512_max_pd(input3, inputCopy);
    }
    {
        __m512d inputCopy = input2;
        input2 = _mm512_min_pd(input4, inputCopy);
        input4 = _mm512_max_pd(input4, inputCopy);
    }
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input2, inputCopy);
        input2 = _mm512_max_pd(input2, inputCopy);
    }
    {
        __m512d inputCopy = input3;
        input3 = _mm512_min_pd(input4, inputCopy);
        input4 = _mm512_max_pd(input4, inputCopy);
    }
    {
        __m512d inputCopy = input5;
        input5 = _mm512_min_pd(input6, inputCopy);
        input6 = _mm512_max_pd(input6, inputCopy);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeigh6 = _mm512_permutexvar_pd(idxNoNeigh, input6);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMin5 = _mm512_min_pd(permNeigh5, input5);
        __m512d permNeighMin6 = _mm512_min_pd(permNeigh6, input6);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        __m512d permNeighMax5 = _mm512_max_pd(permNeigh5, input5);
        __m512d permNeighMax6 = _mm512_max_pd(permNeigh6, input6);
        input = _mm512_mask_mov_pd(permNeighMin, 0xF0, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xF0, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xF0, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xF0, permNeighMax4);
        input5 = _mm512_mask_mov_pd(permNeighMin5, 0xF0, permNeighMax5);
        input6 = _mm512_mask_mov_pd(permNeighMin6, 0xF0, permNeighMax6);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeigh6 = _mm512_permutexvar_pd(idxNoNeigh, input6);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMin5 = _mm512_min_pd(permNeigh5, input5);
        __m512d permNeighMin6 = _mm512_min_pd(permNeigh6, input6);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        __m512d permNeighMax5 = _mm512_max_pd(permNeigh5, input5);
        __m512d permNeighMax6 = _mm512_max_pd(permNeigh6, input6);
        input = _mm512_mask_mov_pd(permNeighMin, 0xCC, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xCC, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xCC, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xCC, permNeighMax4);
        input5 = _mm512_mask_mov_pd(permNeighMin5, 0xCC, permNeighMax5);
        input6 = _mm512_mask_mov_pd(permNeighMin6, 0xCC, permNeighMax6);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeigh6 = _mm512_permutexvar_pd(idxNoNeigh, input6);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMin5 = _mm512_min_pd(permNeigh5, input5);
        __m512d permNeighMin6 = _mm512_min_pd(permNeigh6, input6);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        __m512d permNeighMax5 = _mm512_max_pd(permNeigh5, input5);
        __m512d permNeighMax6 = _mm512_max_pd(permNeigh6, input6);
        input = _mm512_mask_mov_pd(permNeighMin, 0xAA, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xAA, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xAA, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xAA, permNeighMax4);
        input5 = _mm512_mask_mov_pd(permNeighMin5, 0xAA, permNeighMax5);
        input6 = _mm512_mask_mov_pd(permNeighMin6, 0xAA, permNeighMax6);
    }
}

inline void Finish7VecBitFull(__m512d& input, __m512d& input2, __m512d& input3, __m512d& input4,
                            __m512d& input5, __m512d& input6, __m512d& input7){
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input5, inputCopy);
        input5 = _mm512_max_pd(input5, inputCopy);
    }
    {
        __m512d inputCopy = input2;
        input2 = _mm512_min_pd(input6, inputCopy);
        input6 = _mm512_max_pd(input6, inputCopy);
    }
    {
        __m512d inputCopy = input3;
        input3 = _mm512_min_pd(input7, inputCopy);
        input7 = _mm512_max_pd(input7, inputCopy);
    }
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input3, inputCopy);
        input3 = _mm512_max_pd(input3, inputCopy);
    }
    {
        __m512d inputCopy = input2;
        input2 = _mm512_min_pd(input4, inputCopy);
        input4 = _mm512_max_pd(input4, inputCopy);
    }
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input2, inputCopy);
        input2 = _mm512_max_pd(input2, inputCopy);
    }
    {
        __m512d inputCopy = input3;
        input3 = _mm512_min_pd(input4, inputCopy);
        input4 = _mm512_max_pd(input4, inputCopy);
    }
    {
        __m512d inputCopy = input5;
        input5 = _mm512_min_pd(input7, inputCopy);
        input7 = _mm512_max_pd(input7, inputCopy);
    }
    {
        __m512d inputCopy = input5;
        input5 = _mm512_min_pd(input6, inputCopy);
        input6 = _mm512_max_pd(input6, inputCopy);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeigh6 = _mm512_permutexvar_pd(idxNoNeigh, input6);
        __m512d permNeigh7 = _mm512_permutexvar_pd(idxNoNeigh, input7);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMin5 = _mm512_min_pd(permNeigh5, input5);
        __m512d permNeighMin6 = _mm512_min_pd(permNeigh6, input6);
        __m512d permNeighMin7 = _mm512_min_pd(permNeigh7, input7);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        __m512d permNeighMax5 = _mm512_max_pd(permNeigh5, input5);
        __m512d permNeighMax6 = _mm512_max_pd(permNeigh6, input6);
        __m512d permNeighMax7 = _mm512_max_pd(permNeigh7, input7);
        input = _mm512_mask_mov_pd(permNeighMin, 0xF0, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xF0, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xF0, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xF0, permNeighMax4);
        input5 = _mm512_mask_mov_pd(permNeighMin5, 0xF0, permNeighMax5);
        input6 = _mm512_mask_mov_pd(permNeighMin6, 0xF0, permNeighMax6);
        input7 = _mm512_mask_mov_pd(permNeighMin7, 0xF0, permNeighMax7);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeigh6 = _mm512_permutexvar_pd(idxNoNeigh, input6);
        __m512d permNeigh7 = _mm512_permutexvar_pd(idxNoNeigh, input7);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMin5 = _mm512_min_pd(permNeigh5, input5);
        __m512d permNeighMin6 = _mm512_min_pd(permNeigh6, input6);
        __m512d permNeighMin7 = _mm512_min_pd(permNeigh7, input7);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        __m512d permNeighMax5 = _mm512_max_pd(permNeigh5, input5);
        __m512d permNeighMax6 = _mm512_max_pd(permNeigh6, input6);
        __m512d permNeighMax7 = _mm512_max_pd(permNeigh7, input7);
        input = _mm512_mask_mov_pd(permNeighMin, 0xCC, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xCC, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xCC, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xCC, permNeighMax4);
        input5 = _mm512_mask_mov_pd(permNeighMin5, 0xCC, permNeighMax5);
        input6 = _mm512_mask_mov_pd(permNeighMin6, 0xCC, permNeighMax6);
        input7 = _mm512_mask_mov_pd(permNeighMin7, 0xCC, permNeighMax7);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeigh6 = _mm512_permutexvar_pd(idxNoNeigh, input6);
        __m512d permNeigh7 = _mm512_permutexvar_pd(idxNoNeigh, input7);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMin5 = _mm512_min_pd(permNeigh5, input5);
        __m512d permNeighMin6 = _mm512_min_pd(permNeigh6, input6);
        __m512d permNeighMin7 = _mm512_min_pd(permNeigh7, input7);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        __m512d permNeighMax5 = _mm512_max_pd(permNeigh5, input5);
        __m512d permNeighMax6 = _mm512_max_pd(permNeigh6, input6);
        __m512d permNeighMax7 = _mm512_max_pd(permNeigh7, input7);
        input = _mm512_mask_mov_pd(permNeighMin, 0xAA, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xAA, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xAA, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xAA, permNeighMax4);
        input5 = _mm512_mask_mov_pd(permNeighMin5, 0xAA, permNeighMax5);
        input6 = _mm512_mask_mov_pd(permNeighMin6, 0xAA, permNeighMax6);
        input7 = _mm512_mask_mov_pd(permNeighMin7, 0xAA, permNeighMax7);
    }
}


inline void Finish8VecBitFull(__m512d& input, __m512d& input2, __m512d& input3, __m512d& input4,
                            __m512d& input5, __m512d& input6, __m512d& input7, __m512d& input8 ){
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input5, inputCopy);
        input5 = _mm512_max_pd(input5, inputCopy);
    }
    {
        __m512d inputCopy = input2;
        input2 = _mm512_min_pd(input6, inputCopy);
        input6 = _mm512_max_pd(input6, inputCopy);
    }
    {
        __m512d inputCopy = input3;
        input3 = _mm512_min_pd(input7, inputCopy);
        input7 = _mm512_max_pd(input7, inputCopy);
    }
    {
        __m512d inputCopy = input4;
        input4 = _mm512_min_pd(input8, inputCopy);
        input8 = _mm512_max_pd(input8, inputCopy);
    }
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input3, inputCopy);
        input3 = _mm512_max_pd(input3, inputCopy);
    }
    {
        __m512d inputCopy = input2;
        input2 = _mm512_min_pd(input4, inputCopy);
        input4 = _mm512_max_pd(input4, inputCopy);
    }
    {
        __m512d inputCopy = input;
        input = _mm512_min_pd(input2, inputCopy);
        input2 = _mm512_max_pd(input2, inputCopy);
    }
    {
        __m512d inputCopy = input3;
        input3 = _mm512_min_pd(input4, inputCopy);
        input4 = _mm512_max_pd(input4, inputCopy);
    }
    {
        __m512d inputCopy = input5;
        input5 = _mm512_min_pd(input7, inputCopy);
        input7 = _mm512_max_pd(input7, inputCopy);
    }
    {
        __m512d inputCopy = input6;
        input6 = _mm512_min_pd(input8, inputCopy);
        input8 = _mm512_max_pd(input8, inputCopy);
    }
    {
        __m512d inputCopy = input5;
        input5 = _mm512_min_pd(input6, inputCopy);
        input6 = _mm512_max_pd(input6, inputCopy);
    }
    {
        __m512d inputCopy = input7;
        input7 = _mm512_min_pd(input8, inputCopy);
        input8 = _mm512_max_pd(input8, inputCopy);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeigh6 = _mm512_permutexvar_pd(idxNoNeigh, input6);
        __m512d permNeigh7 = _mm512_permutexvar_pd(idxNoNeigh, input7);
        __m512d permNeigh8 = _mm512_permutexvar_pd(idxNoNeigh, input8);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMin5 = _mm512_min_pd(permNeigh5, input5);
        __m512d permNeighMin6 = _mm512_min_pd(permNeigh6, input6);
        __m512d permNeighMin7 = _mm512_min_pd(permNeigh7, input7);
        __m512d permNeighMin8 = _mm512_min_pd(permNeigh8, input8);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        __m512d permNeighMax5 = _mm512_max_pd(permNeigh5, input5);
        __m512d permNeighMax6 = _mm512_max_pd(permNeigh6, input6);
        __m512d permNeighMax7 = _mm512_max_pd(permNeigh7, input7);
        __m512d permNeighMax8 = _mm512_max_pd(permNeigh8, input8);
        input = _mm512_mask_mov_pd(permNeighMin, 0xF0, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xF0, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xF0, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xF0, permNeighMax4);
        input5 = _mm512_mask_mov_pd(permNeighMin5, 0xF0, permNeighMax5);
        input6 = _mm512_mask_mov_pd(permNeighMin6, 0xF0, permNeighMax6);
        input7 = _mm512_mask_mov_pd(permNeighMin7, 0xF0, permNeighMax7);
        input8 = _mm512_mask_mov_pd(permNeighMin8, 0xF0, permNeighMax8);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeigh6 = _mm512_permutexvar_pd(idxNoNeigh, input6);
        __m512d permNeigh7 = _mm512_permutexvar_pd(idxNoNeigh, input7);
        __m512d permNeigh8 = _mm512_permutexvar_pd(idxNoNeigh, input8);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMin5 = _mm512_min_pd(permNeigh5, input5);
        __m512d permNeighMin6 = _mm512_min_pd(permNeigh6, input6);
        __m512d permNeighMin7 = _mm512_min_pd(permNeigh7, input7);
        __m512d permNeighMin8 = _mm512_min_pd(permNeigh8, input8);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        __m512d permNeighMax5 = _mm512_max_pd(permNeigh5, input5);
        __m512d permNeighMax6 = _mm512_max_pd(permNeigh6, input6);
        __m512d permNeighMax7 = _mm512_max_pd(permNeigh7, input7);
        __m512d permNeighMax8 = _mm512_max_pd(permNeigh8, input8);
        input = _mm512_mask_mov_pd(permNeighMin, 0xCC, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xCC, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xCC, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xCC, permNeighMax4);
        input5 = _mm512_mask_mov_pd(permNeighMin5, 0xCC, permNeighMax5);
        input6 = _mm512_mask_mov_pd(permNeighMin6, 0xCC, permNeighMax6);
        input7 = _mm512_mask_mov_pd(permNeighMin7, 0xCC, permNeighMax7);
        input8 = _mm512_mask_mov_pd(permNeighMin8, 0xCC, permNeighMax8);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
        __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        __m512d permNeigh3 = _mm512_permutexvar_pd(idxNoNeigh, input3);
        __m512d permNeigh4 = _mm512_permutexvar_pd(idxNoNeigh, input4);
        __m512d permNeigh5 = _mm512_permutexvar_pd(idxNoNeigh, input5);
        __m512d permNeigh6 = _mm512_permutexvar_pd(idxNoNeigh, input6);
        __m512d permNeigh7 = _mm512_permutexvar_pd(idxNoNeigh, input7);
        __m512d permNeigh8 = _mm512_permutexvar_pd(idxNoNeigh, input8);
        __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
        __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
        __m512d permNeighMin3 = _mm512_min_pd(permNeigh3, input3);
        __m512d permNeighMin4 = _mm512_min_pd(permNeigh4, input4);
        __m512d permNeighMin5 = _mm512_min_pd(permNeigh5, input5);
        __m512d permNeighMin6 = _mm512_min_pd(permNeigh6, input6);
        __m512d permNeighMin7 = _mm512_min_pd(permNeigh7, input7);
        __m512d permNeighMin8 = _mm512_min_pd(permNeigh8, input8);
        __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
        __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
        __m512d permNeighMax3 = _mm512_max_pd(permNeigh3, input3);
        __m512d permNeighMax4 = _mm512_max_pd(permNeigh4, input4);
        __m512d permNeighMax5 = _mm512_max_pd(permNeigh5, input5);
        __m512d permNeighMax6 = _mm512_max_pd(permNeigh6, input6);
        __m512d permNeighMax7 = _mm512_max_pd(permNeigh7, input7);
        __m512d permNeighMax8 = _mm512_max_pd(permNeigh8, input8);
        input = _mm512_mask_mov_pd(permNeighMin, 0xAA, permNeighMax);
        input2 = _mm512_mask_mov_pd(permNeighMin2, 0xAA, permNeighMax2);
        input3 = _mm512_mask_mov_pd(permNeighMin3, 0xAA, permNeighMax3);
        input4 = _mm512_mask_mov_pd(permNeighMin4, 0xAA, permNeighMax4);
        input5 = _mm512_mask_mov_pd(permNeighMin5, 0xAA, permNeighMax5);
        input6 = _mm512_mask_mov_pd(permNeighMin6, 0xAA, permNeighMax6);
        input7 = _mm512_mask_mov_pd(permNeighMin7, 0xAA, permNeighMax7);
        input8 = _mm512_mask_mov_pd(permNeighMin8, 0xAA, permNeighMax8);
    }
}

inline void Sort9VecBitFull(__m512d& input, __m512d& input2, __m512d& input3, __m512d& input4,
                            __m512d& input5, __m512d& input6, __m512d& input7, __m512d& input8,
                             __m512d& input9){
    Sort8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    input9 = SortVecBitFull(input9);
    {
        __m512i idxNoNeigh = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        __m512d permNeigh9 = _mm512_permutexvar_pd(idxNoNeigh, input9);

        input9 = _mm512_max_pd(input8, permNeigh9);

        input8 = _mm512_min_pd(input8, permNeigh9);
    }
    Finish8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Finish1VecBitFull(input9);
}



inline void Sort9VecBitFull(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3,
                            double* __restrict__ ptr4, double* __restrict__ ptr5, double* __restrict__ ptr6,
                            double* __restrict__ ptr7, double* __restrict__ ptr8,
                             double* __restrict__ ptr9){
    __m512d input1 = _mm512_loadu_pd(ptr1);
    __m512d input2 = _mm512_loadu_pd(ptr2);
    __m512d input3 = _mm512_loadu_pd(ptr3);
    __m512d input4 = _mm512_loadu_pd(ptr4);
    __m512d input5 = _mm512_loadu_pd(ptr5);
    __m512d input6 = _mm512_loadu_pd(ptr6);
    __m512d input7 = _mm512_loadu_pd(ptr7);
    __m512d input8 = _mm512_loadu_pd(ptr8);
    __m512d input9 = _mm512_loadu_pd(ptr9);
    Sort9VecBitFull(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9);
    _mm512_storeu_pd(ptr1, input1);
    _mm512_storeu_pd(ptr2, input2);
    _mm512_storeu_pd(ptr3, input3);
    _mm512_storeu_pd(ptr4, input4);
    _mm512_storeu_pd(ptr5, input5);
    _mm512_storeu_pd(ptr6, input6);
    _mm512_storeu_pd(ptr7, input7);
    _mm512_storeu_pd(ptr8, input8);
    _mm512_storeu_pd(ptr9, input9);
}


inline void Sort10VecBitFull(__m512d& input, __m512d& input2, __m512d& input3, __m512d& input4,
                            __m512d& input5, __m512d& input6, __m512d& input7, __m512d& input8,
                             __m512d& input9, __m512d& input10){
    Sort8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Sort2VecBitFull(input9, input10);
    {
        __m512i idxNoNeigh = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        __m512d permNeigh9 = _mm512_permutexvar_pd(idxNoNeigh, input9);
        __m512d permNeigh10 = _mm512_permutexvar_pd(idxNoNeigh, input10);

        input9 = _mm512_max_pd(input8, permNeigh9);
        input10 = _mm512_max_pd(input7, permNeigh10);

        input8 = _mm512_min_pd(input8, permNeigh9);
        input7 = _mm512_min_pd(input7, permNeigh10);
    }
    Finish8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Finish2VecBitFull(input9, input10);
}



inline void Sort10VecBitFull(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3,
                            double* __restrict__ ptr4, double* __restrict__ ptr5, double* __restrict__ ptr6,
                            double* __restrict__ ptr7, double* __restrict__ ptr8,
                             double* __restrict__ ptr9, double* __restrict__ ptr10){
    __m512d input1 = _mm512_loadu_pd(ptr1);
    __m512d input2 = _mm512_loadu_pd(ptr2);
    __m512d input3 = _mm512_loadu_pd(ptr3);
    __m512d input4 = _mm512_loadu_pd(ptr4);
    __m512d input5 = _mm512_loadu_pd(ptr5);
    __m512d input6 = _mm512_loadu_pd(ptr6);
    __m512d input7 = _mm512_loadu_pd(ptr7);
    __m512d input8 = _mm512_loadu_pd(ptr8);
    __m512d input9 = _mm512_loadu_pd(ptr9);
    __m512d input10 = _mm512_loadu_pd(ptr10);
    Sort10VecBitFull(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10);
    _mm512_storeu_pd(ptr1, input1);
    _mm512_storeu_pd(ptr2, input2);
    _mm512_storeu_pd(ptr3, input3);
    _mm512_storeu_pd(ptr4, input4);
    _mm512_storeu_pd(ptr5, input5);
    _mm512_storeu_pd(ptr6, input6);
    _mm512_storeu_pd(ptr7, input7);
    _mm512_storeu_pd(ptr8, input8);
    _mm512_storeu_pd(ptr9, input9);
    _mm512_storeu_pd(ptr10, input10);
}

inline void Sort11VecBitFull(__m512d& input, __m512d& input2, __m512d& input3, __m512d& input4,
                            __m512d& input5, __m512d& input6, __m512d& input7, __m512d& input8,
                             __m512d& input9, __m512d& input10, __m512d& input11){
    Sort8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Sort3VecBitFull(input9, input10, input11);
    {
        __m512i idxNoNeigh = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        __m512d permNeigh9 = _mm512_permutexvar_pd(idxNoNeigh, input9);
        __m512d permNeigh10 = _mm512_permutexvar_pd(idxNoNeigh, input10);
        __m512d permNeigh11 = _mm512_permutexvar_pd(idxNoNeigh, input11);

        input9 = _mm512_max_pd(input8, permNeigh9);
        input10 = _mm512_max_pd(input7, permNeigh10);
        input11 = _mm512_max_pd(input6, permNeigh11);

        input8 = _mm512_min_pd(input8, permNeigh9);
        input7 = _mm512_min_pd(input7, permNeigh10);
        input6 = _mm512_min_pd(input6, permNeigh11);
    }
    Finish8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Finish3VecBitFull(input9, input10, input11);
}



inline void Sort11VecBitFull(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3,
                            double* __restrict__ ptr4, double* __restrict__ ptr5, double* __restrict__ ptr6,
                            double* __restrict__ ptr7, double* __restrict__ ptr8,
                             double* __restrict__ ptr9, double* __restrict__ ptr10, double* __restrict__ ptr11){
    __m512d input1 = _mm512_loadu_pd(ptr1);
    __m512d input2 = _mm512_loadu_pd(ptr2);
    __m512d input3 = _mm512_loadu_pd(ptr3);
    __m512d input4 = _mm512_loadu_pd(ptr4);
    __m512d input5 = _mm512_loadu_pd(ptr5);
    __m512d input6 = _mm512_loadu_pd(ptr6);
    __m512d input7 = _mm512_loadu_pd(ptr7);
    __m512d input8 = _mm512_loadu_pd(ptr8);
    __m512d input9 = _mm512_loadu_pd(ptr9);
    __m512d input10 = _mm512_loadu_pd(ptr10);
    __m512d input11 = _mm512_loadu_pd(ptr11);
    Sort11VecBitFull(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10, input11);
    _mm512_storeu_pd(ptr1, input1);
    _mm512_storeu_pd(ptr2, input2);
    _mm512_storeu_pd(ptr3, input3);
    _mm512_storeu_pd(ptr4, input4);
    _mm512_storeu_pd(ptr5, input5);
    _mm512_storeu_pd(ptr6, input6);
    _mm512_storeu_pd(ptr7, input7);
    _mm512_storeu_pd(ptr8, input8);
    _mm512_storeu_pd(ptr9, input9);
    _mm512_storeu_pd(ptr10, input10);
    _mm512_storeu_pd(ptr11, input11);
}

inline void Sort12VecBitFull(__m512d& input, __m512d& input2, __m512d& input3, __m512d& input4,
                            __m512d& input5, __m512d& input6, __m512d& input7, __m512d& input8,
                             __m512d& input9, __m512d& input10, __m512d& input11, __m512d& input12){
    Sort8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Sort4VecBitFull(input9, input10, input11, input12);
    {
        __m512i idxNoNeigh = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        __m512d permNeigh9 = _mm512_permutexvar_pd(idxNoNeigh, input9);
        __m512d permNeigh10 = _mm512_permutexvar_pd(idxNoNeigh, input10);
        __m512d permNeigh11 = _mm512_permutexvar_pd(idxNoNeigh, input11);
        __m512d permNeigh12 = _mm512_permutexvar_pd(idxNoNeigh, input12);

        input9 = _mm512_max_pd(input8, permNeigh9);
        input10 = _mm512_max_pd(input7, permNeigh10);
        input11 = _mm512_max_pd(input6, permNeigh11);
        input12 = _mm512_max_pd(input5, permNeigh12);

        input8 = _mm512_min_pd(input8, permNeigh9);
        input7 = _mm512_min_pd(input7, permNeigh10);
        input6 = _mm512_min_pd(input6, permNeigh11);
        input5 = _mm512_min_pd(input5, permNeigh12);
    }
    Finish8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Finish4VecBitFull(input9, input10, input11, input12);
}



inline void Sort12VecBitFull(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3,
                            double* __restrict__ ptr4, double* __restrict__ ptr5, double* __restrict__ ptr6,
                            double* __restrict__ ptr7, double* __restrict__ ptr8,
                             double* __restrict__ ptr9, double* __restrict__ ptr10, double* __restrict__ ptr11,
                             double* __restrict__ ptr12){
    __m512d input1 = _mm512_loadu_pd(ptr1);
    __m512d input2 = _mm512_loadu_pd(ptr2);
    __m512d input3 = _mm512_loadu_pd(ptr3);
    __m512d input4 = _mm512_loadu_pd(ptr4);
    __m512d input5 = _mm512_loadu_pd(ptr5);
    __m512d input6 = _mm512_loadu_pd(ptr6);
    __m512d input7 = _mm512_loadu_pd(ptr7);
    __m512d input8 = _mm512_loadu_pd(ptr8);
    __m512d input9 = _mm512_loadu_pd(ptr9);
    __m512d input10 = _mm512_loadu_pd(ptr10);
    __m512d input11 = _mm512_loadu_pd(ptr11);
    __m512d input12 = _mm512_loadu_pd(ptr12);
    Sort12VecBitFull(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10, input11, input12);
    _mm512_storeu_pd(ptr1, input1);
    _mm512_storeu_pd(ptr2, input2);
    _mm512_storeu_pd(ptr3, input3);
    _mm512_storeu_pd(ptr4, input4);
    _mm512_storeu_pd(ptr5, input5);
    _mm512_storeu_pd(ptr6, input6);
    _mm512_storeu_pd(ptr7, input7);
    _mm512_storeu_pd(ptr8, input8);
    _mm512_storeu_pd(ptr9, input9);
    _mm512_storeu_pd(ptr10, input10);
    _mm512_storeu_pd(ptr11, input11);
    _mm512_storeu_pd(ptr12, input12);
}

inline void Sort13VecBitFull(__m512d& input, __m512d& input2, __m512d& input3, __m512d& input4,
                            __m512d& input5, __m512d& input6, __m512d& input7, __m512d& input8,
                             __m512d& input9, __m512d& input10, __m512d& input11, __m512d& input12,
                             __m512d& input13){
    Sort8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Sort5VecBitFull(input9, input10, input11, input12, input13);
    {
        __m512i idxNoNeigh = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        __m512d permNeigh9 = _mm512_permutexvar_pd(idxNoNeigh, input9);
        __m512d permNeigh10 = _mm512_permutexvar_pd(idxNoNeigh, input10);
        __m512d permNeigh11 = _mm512_permutexvar_pd(idxNoNeigh, input11);
        __m512d permNeigh12 = _mm512_permutexvar_pd(idxNoNeigh, input12);
        __m512d permNeigh13 = _mm512_permutexvar_pd(idxNoNeigh, input13);

        input9 = _mm512_max_pd(input8, permNeigh9);
        input10 = _mm512_max_pd(input7, permNeigh10);
        input11 = _mm512_max_pd(input6, permNeigh11);
        input12 = _mm512_max_pd(input5, permNeigh12);
        input13 = _mm512_max_pd(input4, permNeigh13);

        input8 = _mm512_min_pd(input8, permNeigh9);
        input7 = _mm512_min_pd(input7, permNeigh10);
        input6 = _mm512_min_pd(input6, permNeigh11);
        input5 = _mm512_min_pd(input5, permNeigh12);
        input4 = _mm512_min_pd(input4, permNeigh13);
    }
    Finish8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Finish5VecBitFull(input9, input10, input11, input12, input13);
}



inline void Sort13VecBitFull(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3,
                            double* __restrict__ ptr4, double* __restrict__ ptr5, double* __restrict__ ptr6,
                            double* __restrict__ ptr7, double* __restrict__ ptr8,
                             double* __restrict__ ptr9, double* __restrict__ ptr10, double* __restrict__ ptr11,
                             double* __restrict__ ptr12, double* __restrict__ ptr13){
    __m512d input1 = _mm512_loadu_pd(ptr1);
    __m512d input2 = _mm512_loadu_pd(ptr2);
    __m512d input3 = _mm512_loadu_pd(ptr3);
    __m512d input4 = _mm512_loadu_pd(ptr4);
    __m512d input5 = _mm512_loadu_pd(ptr5);
    __m512d input6 = _mm512_loadu_pd(ptr6);
    __m512d input7 = _mm512_loadu_pd(ptr7);
    __m512d input8 = _mm512_loadu_pd(ptr8);
    __m512d input9 = _mm512_loadu_pd(ptr9);
    __m512d input10 = _mm512_loadu_pd(ptr10);
    __m512d input11 = _mm512_loadu_pd(ptr11);
    __m512d input12 = _mm512_loadu_pd(ptr12);
    __m512d input13 = _mm512_loadu_pd(ptr13);
    Sort13VecBitFull(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10, input11, input12, input13);
    _mm512_storeu_pd(ptr1, input1);
    _mm512_storeu_pd(ptr2, input2);
    _mm512_storeu_pd(ptr3, input3);
    _mm512_storeu_pd(ptr4, input4);
    _mm512_storeu_pd(ptr5, input5);
    _mm512_storeu_pd(ptr6, input6);
    _mm512_storeu_pd(ptr7, input7);
    _mm512_storeu_pd(ptr8, input8);
    _mm512_storeu_pd(ptr9, input9);
    _mm512_storeu_pd(ptr10, input10);
    _mm512_storeu_pd(ptr11, input11);
    _mm512_storeu_pd(ptr12, input12);
    _mm512_storeu_pd(ptr13, input13);
}

inline void Sort14VecBitFull(__m512d& input, __m512d& input2, __m512d& input3, __m512d& input4,
                            __m512d& input5, __m512d& input6, __m512d& input7, __m512d& input8,
                             __m512d& input9, __m512d& input10, __m512d& input11, __m512d& input12,
                             __m512d& input13, __m512d& input14){
    Sort8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Sort6VecBitFull(input9, input10, input11, input12, input13, input14);
    {
        __m512i idxNoNeigh = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        __m512d permNeigh9 = _mm512_permutexvar_pd(idxNoNeigh, input9);
        __m512d permNeigh10 = _mm512_permutexvar_pd(idxNoNeigh, input10);
        __m512d permNeigh11 = _mm512_permutexvar_pd(idxNoNeigh, input11);
        __m512d permNeigh12 = _mm512_permutexvar_pd(idxNoNeigh, input12);
        __m512d permNeigh13 = _mm512_permutexvar_pd(idxNoNeigh, input13);
        __m512d permNeigh14 = _mm512_permutexvar_pd(idxNoNeigh, input14);

        input9 = _mm512_max_pd(input8, permNeigh9);
        input10 = _mm512_max_pd(input7, permNeigh10);
        input11 = _mm512_max_pd(input6, permNeigh11);
        input12 = _mm512_max_pd(input5, permNeigh12);
        input13 = _mm512_max_pd(input4, permNeigh13);
        input14 = _mm512_max_pd(input3, permNeigh14);

        input8 = _mm512_min_pd(input8, permNeigh9);
        input7 = _mm512_min_pd(input7, permNeigh10);
        input6 = _mm512_min_pd(input6, permNeigh11);
        input5 = _mm512_min_pd(input5, permNeigh12);
        input4 = _mm512_min_pd(input4, permNeigh13);
        input3 = _mm512_min_pd(input3, permNeigh14);
    }
    Finish8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Finish6VecBitFull(input9, input10, input11, input12, input13, input14);
}



inline void Sort14VecBitFull(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3,
                            double* __restrict__ ptr4, double* __restrict__ ptr5, double* __restrict__ ptr6,
                            double* __restrict__ ptr7, double* __restrict__ ptr8,
                             double* __restrict__ ptr9, double* __restrict__ ptr10, double* __restrict__ ptr11,
                             double* __restrict__ ptr12, double* __restrict__ ptr13, double* __restrict__ ptr14){
    __m512d input1 = _mm512_loadu_pd(ptr1);
    __m512d input2 = _mm512_loadu_pd(ptr2);
    __m512d input3 = _mm512_loadu_pd(ptr3);
    __m512d input4 = _mm512_loadu_pd(ptr4);
    __m512d input5 = _mm512_loadu_pd(ptr5);
    __m512d input6 = _mm512_loadu_pd(ptr6);
    __m512d input7 = _mm512_loadu_pd(ptr7);
    __m512d input8 = _mm512_loadu_pd(ptr8);
    __m512d input9 = _mm512_loadu_pd(ptr9);
    __m512d input10 = _mm512_loadu_pd(ptr10);
    __m512d input11 = _mm512_loadu_pd(ptr11);
    __m512d input12 = _mm512_loadu_pd(ptr12);
    __m512d input13 = _mm512_loadu_pd(ptr13);
    __m512d input14 = _mm512_loadu_pd(ptr14);
    Sort14VecBitFull(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10, input11, input12, input13, input14);
    _mm512_storeu_pd(ptr1, input1);
    _mm512_storeu_pd(ptr2, input2);
    _mm512_storeu_pd(ptr3, input3);
    _mm512_storeu_pd(ptr4, input4);
    _mm512_storeu_pd(ptr5, input5);
    _mm512_storeu_pd(ptr6, input6);
    _mm512_storeu_pd(ptr7, input7);
    _mm512_storeu_pd(ptr8, input8);
    _mm512_storeu_pd(ptr9, input9);
    _mm512_storeu_pd(ptr10, input10);
    _mm512_storeu_pd(ptr11, input11);
    _mm512_storeu_pd(ptr12, input12);
    _mm512_storeu_pd(ptr13, input13);
    _mm512_storeu_pd(ptr14, input14);
}

inline void Sort15VecBitFull(__m512d& input, __m512d& input2, __m512d& input3, __m512d& input4,
                            __m512d& input5, __m512d& input6, __m512d& input7, __m512d& input8,
                             __m512d& input9, __m512d& input10, __m512d& input11, __m512d& input12,
                             __m512d& input13, __m512d& input14, __m512d& input15){
    Sort8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Sort7VecBitFull(input9, input10, input11, input12, input13, input14, input15);
    {
        __m512i idxNoNeigh = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        __m512d permNeigh9 = _mm512_permutexvar_pd(idxNoNeigh, input9);
        __m512d permNeigh10 = _mm512_permutexvar_pd(idxNoNeigh, input10);
        __m512d permNeigh11 = _mm512_permutexvar_pd(idxNoNeigh, input11);
        __m512d permNeigh12 = _mm512_permutexvar_pd(idxNoNeigh, input12);
        __m512d permNeigh13 = _mm512_permutexvar_pd(idxNoNeigh, input13);
        __m512d permNeigh14 = _mm512_permutexvar_pd(idxNoNeigh, input14);
        __m512d permNeigh15 = _mm512_permutexvar_pd(idxNoNeigh, input15);

        input9 = _mm512_max_pd(input8, permNeigh9);
        input10 = _mm512_max_pd(input7, permNeigh10);
        input11 = _mm512_max_pd(input6, permNeigh11);
        input12 = _mm512_max_pd(input5, permNeigh12);
        input13 = _mm512_max_pd(input4, permNeigh13);
        input14 = _mm512_max_pd(input3, permNeigh14);
        input15 = _mm512_max_pd(input2, permNeigh15);

        input8 = _mm512_min_pd(input8, permNeigh9);
        input7 = _mm512_min_pd(input7, permNeigh10);
        input6 = _mm512_min_pd(input6, permNeigh11);
        input5 = _mm512_min_pd(input5, permNeigh12);
        input4 = _mm512_min_pd(input4, permNeigh13);
        input3 = _mm512_min_pd(input3, permNeigh14);
        input2 = _mm512_min_pd(input2, permNeigh15);
    }
    Finish8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Finish7VecBitFull(input9, input10, input11, input12, input13, input14, input15);
}



inline void Sort15VecBitFull(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3,
                            double* __restrict__ ptr4, double* __restrict__ ptr5, double* __restrict__ ptr6,
                            double* __restrict__ ptr7, double* __restrict__ ptr8,
                             double* __restrict__ ptr9, double* __restrict__ ptr10, double* __restrict__ ptr11,
                             double* __restrict__ ptr12, double* __restrict__ ptr13, double* __restrict__ ptr14,
                             double* __restrict__ ptr15){
    __m512d input1 = _mm512_loadu_pd(ptr1);
    __m512d input2 = _mm512_loadu_pd(ptr2);
    __m512d input3 = _mm512_loadu_pd(ptr3);
    __m512d input4 = _mm512_loadu_pd(ptr4);
    __m512d input5 = _mm512_loadu_pd(ptr5);
    __m512d input6 = _mm512_loadu_pd(ptr6);
    __m512d input7 = _mm512_loadu_pd(ptr7);
    __m512d input8 = _mm512_loadu_pd(ptr8);
    __m512d input9 = _mm512_loadu_pd(ptr9);
    __m512d input10 = _mm512_loadu_pd(ptr10);
    __m512d input11 = _mm512_loadu_pd(ptr11);
    __m512d input12 = _mm512_loadu_pd(ptr12);
    __m512d input13 = _mm512_loadu_pd(ptr13);
    __m512d input14 = _mm512_loadu_pd(ptr14);
    __m512d input15 = _mm512_loadu_pd(ptr15);
    Sort15VecBitFull(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10, input11, input12, input13, input14, input15);
    _mm512_storeu_pd(ptr1, input1);
    _mm512_storeu_pd(ptr2, input2);
    _mm512_storeu_pd(ptr3, input3);
    _mm512_storeu_pd(ptr4, input4);
    _mm512_storeu_pd(ptr5, input5);
    _mm512_storeu_pd(ptr6, input6);
    _mm512_storeu_pd(ptr7, input7);
    _mm512_storeu_pd(ptr8, input8);
    _mm512_storeu_pd(ptr9, input9);
    _mm512_storeu_pd(ptr10, input10);
    _mm512_storeu_pd(ptr11, input11);
    _mm512_storeu_pd(ptr12, input12);
    _mm512_storeu_pd(ptr13, input13);
    _mm512_storeu_pd(ptr14, input14);
    _mm512_storeu_pd(ptr15, input15);
}


inline void Sort16VecBitFull(__m512d& input, __m512d& input2, __m512d& input3, __m512d& input4,
                            __m512d& input5, __m512d& input6, __m512d& input7, __m512d& input8,
                             __m512d& input9, __m512d& input10, __m512d& input11, __m512d& input12,
                             __m512d& input13, __m512d& input14, __m512d& input15, __m512d& input16){
    Sort8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Sort8VecBitFull(input9, input10, input11, input12, input13, input14, input15, input16);
    {
        __m512i idxNoNeigh = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        __m512d permNeigh9 = _mm512_permutexvar_pd(idxNoNeigh, input9);
        __m512d permNeigh10 = _mm512_permutexvar_pd(idxNoNeigh, input10);
        __m512d permNeigh11 = _mm512_permutexvar_pd(idxNoNeigh, input11);
        __m512d permNeigh12 = _mm512_permutexvar_pd(idxNoNeigh, input12);
        __m512d permNeigh13 = _mm512_permutexvar_pd(idxNoNeigh, input13);
        __m512d permNeigh14 = _mm512_permutexvar_pd(idxNoNeigh, input14);
        __m512d permNeigh15 = _mm512_permutexvar_pd(idxNoNeigh, input15);
        __m512d permNeigh16 = _mm512_permutexvar_pd(idxNoNeigh, input16);

        input9 = _mm512_max_pd(input8, permNeigh9);
        input10 = _mm512_max_pd(input7, permNeigh10);
        input11 = _mm512_max_pd(input6, permNeigh11);
        input12 = _mm512_max_pd(input5, permNeigh12);
        input13 = _mm512_max_pd(input4, permNeigh13);
        input14 = _mm512_max_pd(input3, permNeigh14);
        input15 = _mm512_max_pd(input2, permNeigh15);
        input16 = _mm512_max_pd(input, permNeigh16);

        input8 = _mm512_min_pd(input8, permNeigh9);
        input7 = _mm512_min_pd(input7, permNeigh10);
        input6 = _mm512_min_pd(input6, permNeigh11);
        input5 = _mm512_min_pd(input5, permNeigh12);
        input4 = _mm512_min_pd(input4, permNeigh13);
        input3 = _mm512_min_pd(input3, permNeigh14);
        input2 = _mm512_min_pd(input2, permNeigh15);
        input = _mm512_min_pd(input, permNeigh16);
    }
    Finish8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Finish8VecBitFull(input9, input10, input11, input12, input13, input14, input15, input16);
}



inline void Sort16VecBitFull(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3,
                            double* __restrict__ ptr4, double* __restrict__ ptr5, double* __restrict__ ptr6,
                            double* __restrict__ ptr7, double* __restrict__ ptr8,
                             double* __restrict__ ptr9, double* __restrict__ ptr10, double* __restrict__ ptr11,
                             double* __restrict__ ptr12, double* __restrict__ ptr13, double* __restrict__ ptr14,
                             double* __restrict__ ptr15, double* __restrict__ ptr16){
    __m512d input1 = _mm512_loadu_pd(ptr1);
    __m512d input2 = _mm512_loadu_pd(ptr2);
    __m512d input3 = _mm512_loadu_pd(ptr3);
    __m512d input4 = _mm512_loadu_pd(ptr4);
    __m512d input5 = _mm512_loadu_pd(ptr5);
    __m512d input6 = _mm512_loadu_pd(ptr6);
    __m512d input7 = _mm512_loadu_pd(ptr7);
    __m512d input8 = _mm512_loadu_pd(ptr8);
    __m512d input9 = _mm512_loadu_pd(ptr9);
    __m512d input10 = _mm512_loadu_pd(ptr10);
    __m512d input11 = _mm512_loadu_pd(ptr11);
    __m512d input12 = _mm512_loadu_pd(ptr12);
    __m512d input13 = _mm512_loadu_pd(ptr13);
    __m512d input14 = _mm512_loadu_pd(ptr14);
    __m512d input15 = _mm512_loadu_pd(ptr15);
    __m512d input16 = _mm512_loadu_pd(ptr16);
    Sort16VecBitFull(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10, input11, input12, input13, input14, input15, input16);
    _mm512_storeu_pd(ptr1, input1);
    _mm512_storeu_pd(ptr2, input2);
    _mm512_storeu_pd(ptr3, input3);
    _mm512_storeu_pd(ptr4, input4);
    _mm512_storeu_pd(ptr5, input5);
    _mm512_storeu_pd(ptr6, input6);
    _mm512_storeu_pd(ptr7, input7);
    _mm512_storeu_pd(ptr8, input8);
    _mm512_storeu_pd(ptr9, input9);
    _mm512_storeu_pd(ptr10, input10);
    _mm512_storeu_pd(ptr11, input11);
    _mm512_storeu_pd(ptr12, input12);
    _mm512_storeu_pd(ptr13, input13);
    _mm512_storeu_pd(ptr14, input14);
    _mm512_storeu_pd(ptr15, input15);
    _mm512_storeu_pd(ptr16, input16);
}



inline void SortByVec(double* __restrict__ ptr, const size_t length){
    // length is limited to 4 times size of a vec
    const int nbValuesInVec = 8;
    const int nbVecs = (length+nbValuesInVec-1)/nbValuesInVec;
    const int rest = nbVecs*nbValuesInVec-length;
    const int lastVecSize = nbValuesInVec-rest;
    const double temp_DBL_MAX = DBL_MAX;
    const long int double_max = reinterpret_cast<const long int&>(temp_DBL_MAX);
    switch(nbVecs){
    case 1:
    {
        __m512d v1 = _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(_mm512_maskz_loadu_pd(0xFF>>rest, ptr)),
                        _mm512_maskz_set1_epi64(0xFF<<lastVecSize, double_max)));
        v1 = SortVec(v1);
        _mm512_mask_compressstoreu_pd(ptr, 0xFF>>rest, v1);
    }
        break;
    case 2:
    {
        __m512d v1 = _mm512_loadu_pd(ptr);
        __m512d v2 = _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(_mm512_maskz_loadu_pd(0xFF>>rest, ptr+8)),
                                                         _mm512_maskz_set1_epi64(0xFF<<lastVecSize, double_max)));
        Sort2Vec(v1,v2);
        _mm512_storeu_pd(ptr, v1);
        _mm512_mask_compressstoreu_pd(ptr+8, 0xFF>>rest, v2);
    }
        break;
    case 3:
    {
        __m512d v1 = _mm512_loadu_pd(ptr);
        __m512d v2 = _mm512_loadu_pd(ptr+8);
        __m512d v3 = _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(_mm512_maskz_loadu_pd(0xFF>>rest, ptr+16)),
                                                         _mm512_maskz_set1_epi64(0xFF<<lastVecSize, double_max)));
        Sort3Vec(v1,v2,v3);
        _mm512_storeu_pd(ptr, v1);
        _mm512_storeu_pd(ptr+8, v2);
        _mm512_mask_compressstoreu_pd(ptr+16, 0xFF>>rest, v3);
    }
        break;
    default:;
    //case 4:
    {
        __m512d v1 = _mm512_loadu_pd(ptr);
        __m512d v2 = _mm512_loadu_pd(ptr+8);
        __m512d v3 = _mm512_loadu_pd(ptr+16);
        __m512d v4 = _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(_mm512_maskz_loadu_pd(0xFF>>rest, ptr+24)),
                                                         _mm512_maskz_set1_epi64(0xFF<<lastVecSize, double_max)));
        Sort4Vec(v1,v2,v3,v4);
        _mm512_storeu_pd(ptr, v1);
        _mm512_storeu_pd(ptr+8, v2);
        _mm512_storeu_pd(ptr+16, v3);
        _mm512_mask_compressstoreu_pd(ptr+24, 0xFF>>rest, v4);
    }
    }
}

inline void SortByVecBitFull(double* __restrict__ ptr, const size_t length){
    // length is limited to 4 times size of a vec
    const int nbValuesInVec = 8;
    const int nbVecs = (length+nbValuesInVec-1)/nbValuesInVec;
    const int rest = nbVecs*nbValuesInVec-length;
    const int lastVecSize = nbValuesInVec-rest;
    const double temp_DBL_MAX = DBL_MAX;
    const long int double_max = reinterpret_cast<const long int&>(temp_DBL_MAX);
    switch(nbVecs){
    case 1:
    {
        __m512d v1 = _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(_mm512_maskz_loadu_pd(0xFF>>rest, ptr)),
                        _mm512_maskz_set1_epi64(0xFF<<lastVecSize, double_max)));
        v1 = SortVecBitFull(v1);
        _mm512_mask_compressstoreu_pd(ptr, 0xFF>>rest, v1);
    }
        break;
    case 2:
    {
        __m512d v1 = _mm512_loadu_pd(ptr);
        __m512d v2 = _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(_mm512_maskz_loadu_pd(0xFF>>rest, ptr+8)),
                                                         _mm512_maskz_set1_epi64(0xFF<<lastVecSize, double_max)));
        Sort2VecBitFull(v1,v2);
        _mm512_storeu_pd(ptr, v1);
        _mm512_mask_compressstoreu_pd(ptr+8, 0xFF>>rest, v2);
    }
        break;
    case 3:
    {
        __m512d v1 = _mm512_loadu_pd(ptr);
        __m512d v2 = _mm512_loadu_pd(ptr+8);
        __m512d v3 = _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(_mm512_maskz_loadu_pd(0xFF>>rest, ptr+16)),
                                                         _mm512_maskz_set1_epi64(0xFF<<lastVecSize, double_max)));
        Sort3VecBitFull(v1,v2,v3);
        _mm512_storeu_pd(ptr, v1);
        _mm512_storeu_pd(ptr+8, v2);
        _mm512_mask_compressstoreu_pd(ptr+16, 0xFF>>rest, v3);
    }
        break;
    case 4:
    {
        __m512d v1 = _mm512_loadu_pd(ptr);
        __m512d v2 = _mm512_loadu_pd(ptr+8);
        __m512d v3 = _mm512_loadu_pd(ptr+16);
        __m512d v4 = _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(_mm512_maskz_loadu_pd(0xFF>>rest, ptr+24)),
                                                         _mm512_maskz_set1_epi64(0xFF<<lastVecSize, double_max)));
        Sort4VecBitFull(v1,v2,v3,v4);
        _mm512_storeu_pd(ptr, v1);
        _mm512_storeu_pd(ptr+8, v2);
        _mm512_storeu_pd(ptr+16, v3);
        _mm512_mask_compressstoreu_pd(ptr+24, 0xFF>>rest, v4);
    }
        break;
    case 5:
    {
        __m512d v1 = _mm512_loadu_pd(ptr);
        __m512d v2 = _mm512_loadu_pd(ptr+8);
        __m512d v3 = _mm512_loadu_pd(ptr+16);
        __m512d v4 = _mm512_loadu_pd(ptr+24);
        __m512d v5 = _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(_mm512_maskz_loadu_pd(0xFF>>rest, ptr+32)),
                                                         _mm512_maskz_set1_epi64(0xFF<<lastVecSize, double_max)));
        Sort5VecBitFull(v1,v2,v3,v4,v5);
        _mm512_storeu_pd(ptr, v1);
        _mm512_storeu_pd(ptr+8, v2);
        _mm512_storeu_pd(ptr+16, v3);
        _mm512_storeu_pd(ptr+24, v4);
        _mm512_mask_compressstoreu_pd(ptr+32, 0xFF>>rest, v5);
    }
        break;
    case 6:
    {
        __m512d v1 = _mm512_loadu_pd(ptr);
        __m512d v2 = _mm512_loadu_pd(ptr+8);
        __m512d v3 = _mm512_loadu_pd(ptr+16);
        __m512d v4 = _mm512_loadu_pd(ptr+24);
        __m512d v5 = _mm512_loadu_pd(ptr+32);
        __m512d v6 = _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(_mm512_maskz_loadu_pd(0xFF>>rest, ptr+40)),
                                                         _mm512_maskz_set1_epi64(0xFF<<lastVecSize, double_max)));
        Sort6VecBitFull(v1,v2,v3,v4,v5, v6);
        _mm512_storeu_pd(ptr, v1);
        _mm512_storeu_pd(ptr+8, v2);
        _mm512_storeu_pd(ptr+16, v3);
        _mm512_storeu_pd(ptr+24, v4);
        _mm512_storeu_pd(ptr+32, v5);
        _mm512_mask_compressstoreu_pd(ptr+40, 0xFF>>rest, v6);
    }
        break;
    case 7:
    {
        __m512d v1 = _mm512_loadu_pd(ptr);
        __m512d v2 = _mm512_loadu_pd(ptr+8);
        __m512d v3 = _mm512_loadu_pd(ptr+16);
        __m512d v4 = _mm512_loadu_pd(ptr+24);
        __m512d v5 = _mm512_loadu_pd(ptr+32);
        __m512d v6 = _mm512_loadu_pd(ptr+40);
        __m512d v7 = _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(_mm512_maskz_loadu_pd(0xFF>>rest, ptr+48)),
                                                         _mm512_maskz_set1_epi64(0xFF<<lastVecSize, double_max)));
        Sort7VecBitFull(v1,v2,v3,v4,v5,v6,v7);
        _mm512_storeu_pd(ptr, v1);
        _mm512_storeu_pd(ptr+8, v2);
        _mm512_storeu_pd(ptr+16, v3);
        _mm512_storeu_pd(ptr+24, v4);
        _mm512_storeu_pd(ptr+32, v5);
        _mm512_storeu_pd(ptr+40, v6);
        _mm512_mask_compressstoreu_pd(ptr+48, 0xFF>>rest, v7);
    }
        break;
    case 8:
    {
        __m512d v1 = _mm512_loadu_pd(ptr);
        __m512d v2 = _mm512_loadu_pd(ptr+8);
        __m512d v3 = _mm512_loadu_pd(ptr+16);
        __m512d v4 = _mm512_loadu_pd(ptr+24);
        __m512d v5 = _mm512_loadu_pd(ptr+32);
        __m512d v6 = _mm512_loadu_pd(ptr+40);
        __m512d v7 = _mm512_loadu_pd(ptr+48);
        __m512d v8 = _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(_mm512_maskz_loadu_pd(0xFF>>rest, ptr+56)),
                                                         _mm512_maskz_set1_epi64(0xFF<<lastVecSize, double_max)));
        Sort8VecBitFull(v1,v2,v3,v4,v5,v6,v7,v8);
        _mm512_storeu_pd(ptr, v1);
        _mm512_storeu_pd(ptr+8, v2);
        _mm512_storeu_pd(ptr+16, v3);
        _mm512_storeu_pd(ptr+24, v4);
        _mm512_storeu_pd(ptr+32, v5);
        _mm512_storeu_pd(ptr+40, v6);
        _mm512_storeu_pd(ptr+48, v7);
        _mm512_mask_compressstoreu_pd(ptr+56, 0xFF>>rest, v8);
    }
        break;
    case 9:
    {
        __m512d v1 = _mm512_loadu_pd(ptr);
        __m512d v2 = _mm512_loadu_pd(ptr+8);
        __m512d v3 = _mm512_loadu_pd(ptr+16);
        __m512d v4 = _mm512_loadu_pd(ptr+24);
        __m512d v5 = _mm512_loadu_pd(ptr+32);
        __m512d v6 = _mm512_loadu_pd(ptr+40);
        __m512d v7 = _mm512_loadu_pd(ptr+48);
        __m512d v8 = _mm512_loadu_pd(ptr+56);
        __m512d v9 = _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(_mm512_maskz_loadu_pd(0xFF>>rest, ptr+64)),
                                                         _mm512_maskz_set1_epi64(0xFF<<lastVecSize, double_max)));
        Sort9VecBitFull(v1,v2,v3,v4,v5,v6,v7,v8,v9);
        _mm512_storeu_pd(ptr, v1);
        _mm512_storeu_pd(ptr+8, v2);
        _mm512_storeu_pd(ptr+16, v3);
        _mm512_storeu_pd(ptr+24, v4);
        _mm512_storeu_pd(ptr+32, v5);
        _mm512_storeu_pd(ptr+40, v6);
        _mm512_storeu_pd(ptr+48, v7);
        _mm512_storeu_pd(ptr+56, v8);
        _mm512_mask_compressstoreu_pd(ptr+64, 0xFF>>rest, v9);
    }
        break;
    case 10:
    {
        __m512d v1 = _mm512_loadu_pd(ptr);
        __m512d v2 = _mm512_loadu_pd(ptr+8);
        __m512d v3 = _mm512_loadu_pd(ptr+16);
        __m512d v4 = _mm512_loadu_pd(ptr+24);
        __m512d v5 = _mm512_loadu_pd(ptr+32);
        __m512d v6 = _mm512_loadu_pd(ptr+40);
        __m512d v7 = _mm512_loadu_pd(ptr+48);
        __m512d v8 = _mm512_loadu_pd(ptr+56);
        __m512d v9 = _mm512_loadu_pd(ptr+64);
        __m512d v10 = _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(_mm512_maskz_loadu_pd(0xFF>>rest, ptr+72)),
                                                         _mm512_maskz_set1_epi64(0xFF<<lastVecSize, double_max)));
        Sort10VecBitFull(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10);
        _mm512_storeu_pd(ptr, v1);
        _mm512_storeu_pd(ptr+8, v2);
        _mm512_storeu_pd(ptr+16, v3);
        _mm512_storeu_pd(ptr+24, v4);
        _mm512_storeu_pd(ptr+32, v5);
        _mm512_storeu_pd(ptr+40, v6);
        _mm512_storeu_pd(ptr+48, v7);
        _mm512_storeu_pd(ptr+56, v8);
        _mm512_storeu_pd(ptr+64, v9);
        _mm512_mask_compressstoreu_pd(ptr+72, 0xFF>>rest, v10);
    }
        break;
    case 11:
    {
        __m512d v1 = _mm512_loadu_pd(ptr);
        __m512d v2 = _mm512_loadu_pd(ptr+8);
        __m512d v3 = _mm512_loadu_pd(ptr+16);
        __m512d v4 = _mm512_loadu_pd(ptr+24);
        __m512d v5 = _mm512_loadu_pd(ptr+32);
        __m512d v6 = _mm512_loadu_pd(ptr+40);
        __m512d v7 = _mm512_loadu_pd(ptr+48);
        __m512d v8 = _mm512_loadu_pd(ptr+56);
        __m512d v9 = _mm512_loadu_pd(ptr+64);
        __m512d v10 = _mm512_loadu_pd(ptr+72);
        __m512d v11 = _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(_mm512_maskz_loadu_pd(0xFF>>rest, ptr+80)),
                                                         _mm512_maskz_set1_epi64(0xFF<<lastVecSize, double_max)));
        Sort11VecBitFull(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11);
        _mm512_storeu_pd(ptr, v1);
        _mm512_storeu_pd(ptr+8, v2);
        _mm512_storeu_pd(ptr+16, v3);
        _mm512_storeu_pd(ptr+24, v4);
        _mm512_storeu_pd(ptr+32, v5);
        _mm512_storeu_pd(ptr+40, v6);
        _mm512_storeu_pd(ptr+48, v7);
        _mm512_storeu_pd(ptr+56, v8);
        _mm512_storeu_pd(ptr+64, v9);
        _mm512_storeu_pd(ptr+72, v10);
        _mm512_mask_compressstoreu_pd(ptr+80, 0xFF>>rest, v11);
    }
        break;
    case 12:
    {
        __m512d v1 = _mm512_loadu_pd(ptr);
        __m512d v2 = _mm512_loadu_pd(ptr+8);
        __m512d v3 = _mm512_loadu_pd(ptr+16);
        __m512d v4 = _mm512_loadu_pd(ptr+24);
        __m512d v5 = _mm512_loadu_pd(ptr+32);
        __m512d v6 = _mm512_loadu_pd(ptr+40);
        __m512d v7 = _mm512_loadu_pd(ptr+48);
        __m512d v8 = _mm512_loadu_pd(ptr+56);
        __m512d v9 = _mm512_loadu_pd(ptr+64);
        __m512d v10 = _mm512_loadu_pd(ptr+72);
        __m512d v11 = _mm512_loadu_pd(ptr+80);
        __m512d v12 = _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(_mm512_maskz_loadu_pd(0xFF>>rest, ptr+88)),
                                                         _mm512_maskz_set1_epi64(0xFF<<lastVecSize, double_max)));
        Sort12VecBitFull(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12);
        _mm512_storeu_pd(ptr, v1);
        _mm512_storeu_pd(ptr+8, v2);
        _mm512_storeu_pd(ptr+16, v3);
        _mm512_storeu_pd(ptr+24, v4);
        _mm512_storeu_pd(ptr+32, v5);
        _mm512_storeu_pd(ptr+40, v6);
        _mm512_storeu_pd(ptr+48, v7);
        _mm512_storeu_pd(ptr+56, v8);
        _mm512_storeu_pd(ptr+64, v9);
        _mm512_storeu_pd(ptr+72, v10);
        _mm512_storeu_pd(ptr+80, v11);
        _mm512_mask_compressstoreu_pd(ptr+88, 0xFF>>rest, v12);
    }
        break;
    case 13:
    {
        __m512d v1 = _mm512_loadu_pd(ptr);
        __m512d v2 = _mm512_loadu_pd(ptr+8);
        __m512d v3 = _mm512_loadu_pd(ptr+16);
        __m512d v4 = _mm512_loadu_pd(ptr+24);
        __m512d v5 = _mm512_loadu_pd(ptr+32);
        __m512d v6 = _mm512_loadu_pd(ptr+40);
        __m512d v7 = _mm512_loadu_pd(ptr+48);
        __m512d v8 = _mm512_loadu_pd(ptr+56);
        __m512d v9 = _mm512_loadu_pd(ptr+64);
        __m512d v10 = _mm512_loadu_pd(ptr+72);
        __m512d v11 = _mm512_loadu_pd(ptr+80);
        __m512d v12 = _mm512_loadu_pd(ptr+88);
        __m512d v13 = _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(_mm512_maskz_loadu_pd(0xFF>>rest, ptr+96)),
                                                         _mm512_maskz_set1_epi64(0xFF<<lastVecSize, double_max)));
        Sort13VecBitFull(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13);
        _mm512_storeu_pd(ptr, v1);
        _mm512_storeu_pd(ptr+8, v2);
        _mm512_storeu_pd(ptr+16, v3);
        _mm512_storeu_pd(ptr+24, v4);
        _mm512_storeu_pd(ptr+32, v5);
        _mm512_storeu_pd(ptr+40, v6);
        _mm512_storeu_pd(ptr+48, v7);
        _mm512_storeu_pd(ptr+56, v8);
        _mm512_storeu_pd(ptr+64, v9);
        _mm512_storeu_pd(ptr+72, v10);
        _mm512_storeu_pd(ptr+80, v11);
        _mm512_storeu_pd(ptr+88, v12);
        _mm512_mask_compressstoreu_pd(ptr+96, 0xFF>>rest, v13);
    }
        break;
    case 14:
    {
        __m512d v1 = _mm512_loadu_pd(ptr);
        __m512d v2 = _mm512_loadu_pd(ptr+8);
        __m512d v3 = _mm512_loadu_pd(ptr+16);
        __m512d v4 = _mm512_loadu_pd(ptr+24);
        __m512d v5 = _mm512_loadu_pd(ptr+32);
        __m512d v6 = _mm512_loadu_pd(ptr+40);
        __m512d v7 = _mm512_loadu_pd(ptr+48);
        __m512d v8 = _mm512_loadu_pd(ptr+56);
        __m512d v9 = _mm512_loadu_pd(ptr+64);
        __m512d v10 = _mm512_loadu_pd(ptr+72);
        __m512d v11 = _mm512_loadu_pd(ptr+80);
        __m512d v12 = _mm512_loadu_pd(ptr+88);
        __m512d v13 = _mm512_loadu_pd(ptr+96);
        __m512d v14 = _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(_mm512_maskz_loadu_pd(0xFF>>rest, ptr+104)),
                                                         _mm512_maskz_set1_epi64(0xFF<<lastVecSize, double_max)));
        Sort14VecBitFull(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14);
        _mm512_storeu_pd(ptr, v1);
        _mm512_storeu_pd(ptr+8, v2);
        _mm512_storeu_pd(ptr+16, v3);
        _mm512_storeu_pd(ptr+24, v4);
        _mm512_storeu_pd(ptr+32, v5);
        _mm512_storeu_pd(ptr+40, v6);
        _mm512_storeu_pd(ptr+48, v7);
        _mm512_storeu_pd(ptr+56, v8);
        _mm512_storeu_pd(ptr+64, v9);
        _mm512_storeu_pd(ptr+72, v10);
        _mm512_storeu_pd(ptr+80, v11);
        _mm512_storeu_pd(ptr+88, v12);
        _mm512_storeu_pd(ptr+96, v13);
        _mm512_mask_compressstoreu_pd(ptr+104, 0xFF>>rest, v14);
    }
        break;
    case 15:
    {
        __m512d v1 = _mm512_loadu_pd(ptr);
        __m512d v2 = _mm512_loadu_pd(ptr+8);
        __m512d v3 = _mm512_loadu_pd(ptr+16);
        __m512d v4 = _mm512_loadu_pd(ptr+24);
        __m512d v5 = _mm512_loadu_pd(ptr+32);
        __m512d v6 = _mm512_loadu_pd(ptr+40);
        __m512d v7 = _mm512_loadu_pd(ptr+48);
        __m512d v8 = _mm512_loadu_pd(ptr+56);
        __m512d v9 = _mm512_loadu_pd(ptr+64);
        __m512d v10 = _mm512_loadu_pd(ptr+72);
        __m512d v11 = _mm512_loadu_pd(ptr+80);
        __m512d v12 = _mm512_loadu_pd(ptr+88);
        __m512d v13 = _mm512_loadu_pd(ptr+96);
        __m512d v14 = _mm512_loadu_pd(ptr+104);
        __m512d v15 = _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(_mm512_maskz_loadu_pd(0xFF>>rest, ptr+112)),
                                                         _mm512_maskz_set1_epi64(0xFF<<lastVecSize, double_max)));
        Sort15VecBitFull(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15);
        _mm512_storeu_pd(ptr, v1);
        _mm512_storeu_pd(ptr+8, v2);
        _mm512_storeu_pd(ptr+16, v3);
        _mm512_storeu_pd(ptr+24, v4);
        _mm512_storeu_pd(ptr+32, v5);
        _mm512_storeu_pd(ptr+40, v6);
        _mm512_storeu_pd(ptr+48, v7);
        _mm512_storeu_pd(ptr+56, v8);
        _mm512_storeu_pd(ptr+64, v9);
        _mm512_storeu_pd(ptr+72, v10);
        _mm512_storeu_pd(ptr+80, v11);
        _mm512_storeu_pd(ptr+88, v12);
        _mm512_storeu_pd(ptr+96, v13);
        _mm512_storeu_pd(ptr+104, v14);
        _mm512_mask_compressstoreu_pd(ptr+112, 0xFF>>rest, v15);
    }
        break;
    //case 16:
    default:
    {
        __m512d v1 = _mm512_loadu_pd(ptr);
        __m512d v2 = _mm512_loadu_pd(ptr+8);
        __m512d v3 = _mm512_loadu_pd(ptr+16);
        __m512d v4 = _mm512_loadu_pd(ptr+24);
        __m512d v5 = _mm512_loadu_pd(ptr+32);
        __m512d v6 = _mm512_loadu_pd(ptr+40);
        __m512d v7 = _mm512_loadu_pd(ptr+48);
        __m512d v8 = _mm512_loadu_pd(ptr+56);
        __m512d v9 = _mm512_loadu_pd(ptr+64);
        __m512d v10 = _mm512_loadu_pd(ptr+72);
        __m512d v11 = _mm512_loadu_pd(ptr+80);
        __m512d v12 = _mm512_loadu_pd(ptr+88);
        __m512d v13 = _mm512_loadu_pd(ptr+96);
        __m512d v14 = _mm512_loadu_pd(ptr+104);
        __m512d v15 = _mm512_loadu_pd(ptr+112);
        __m512d v16 = _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(_mm512_maskz_loadu_pd(0xFF>>rest, ptr+120)),
                                                         _mm512_maskz_set1_epi64(0xFF<<lastVecSize, double_max)));
        Sort16VecBitFull(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16);
        _mm512_storeu_pd(ptr, v1);
        _mm512_storeu_pd(ptr+8, v2);
        _mm512_storeu_pd(ptr+16, v3);
        _mm512_storeu_pd(ptr+24, v4);
        _mm512_storeu_pd(ptr+32, v5);
        _mm512_storeu_pd(ptr+40, v6);
        _mm512_storeu_pd(ptr+48, v7);
        _mm512_storeu_pd(ptr+56, v8);
        _mm512_storeu_pd(ptr+64, v9);
        _mm512_storeu_pd(ptr+72, v10);
        _mm512_storeu_pd(ptr+80, v11);
        _mm512_storeu_pd(ptr+88, v12);
        _mm512_storeu_pd(ptr+96, v13);
        _mm512_storeu_pd(ptr+104, v14);
        _mm512_storeu_pd(ptr+112, v15);
        _mm512_mask_compressstoreu_pd(ptr+120, 0xFF>>rest, v16);
    }
    }
}


inline void Sort2VecWithTest(__m512d& input1, __m512d& input2 ){
    __m512i idxNoNeigh = _mm512_set_epi64(7, 5, 6, 3, 4, 1, 2, 0);
    __m512i idxAll0 = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, 0);
    __m512i idxAll7 = _mm512_set_epi64(7, 7, 7, 7, 7, 7, 7, 7);

    for( int idx = 0 ; idx < 8 ; ++idx){
        __m512d permNeighOdd1 = _mm512_permute_pd(input1, 0x55);
        __m512d permNeighOdd2 = _mm512_permute_pd(input2, 0x55);

        __mmask8 compMaskOdd1 = _mm512_cmp_pd_mask(permNeighOdd1, input1, _CMP_LT_OQ);
        __mmask8 compMaskOdd2 = _mm512_cmp_pd_mask(permNeighOdd2, input2, _CMP_LT_OQ);

        input1 = _mm512_mask_mov_pd(input1, (compMaskOdd1 & 0x55) | ((compMaskOdd1 & 0x55)<<1), permNeighOdd1);
        input2 = _mm512_mask_mov_pd(input2, (compMaskOdd2 & 0x55) | ((compMaskOdd2 & 0x55)<<1), permNeighOdd2);

        __m512d permNeighEven1 = _mm512_permutexvar_pd(idxNoNeigh, input1);
        __m512d permNeighEven2 = _mm512_permutexvar_pd(idxNoNeigh, input2);

        __mmask8 compMaskEven1 = _mm512_cmp_pd_mask(permNeighEven1, input1, _CMP_LT_OQ);
        __mmask8 compMaskEven2 = _mm512_cmp_pd_mask(permNeighEven2, input2, _CMP_LT_OQ);

        input1 = _mm512_mask_mov_pd(input1, (compMaskEven1 & 0x2A) | ((compMaskEven1 & 0x2A)<<1), permNeighEven1);
        input2 = _mm512_mask_mov_pd(input2, (compMaskEven2 & 0x2A) | ((compMaskEven2 & 0x2A)<<1), permNeighEven2);

        // Exchange border if needed
        __m512d input1last = _mm512_permutexvar_pd(idxAll7, input1);
        __m512d input2first = _mm512_permutexvar_pd(idxAll0, input2);

        __mmask8 compDoExchange = _mm512_cmp_pd_mask(input1last, input2first, _CMP_GT_OQ);

        input1 = _mm512_mask_mov_pd(input1, compDoExchange & 0x80, input2first);
        input2 = _mm512_mask_mov_pd(input2, compDoExchange & 1, input1last);

        if(!(compMaskOdd1 || compMaskOdd2 || compMaskEven1 || compMaskEven2 || compDoExchange)){
            break;
        }
    }
}

inline void Sort2VecWithTest(double* __restrict__ ptr1, double* __restrict__ ptr2 ){
    __m512d input1 = _mm512_loadu_pd(ptr1);
    __m512d input2 = _mm512_loadu_pd(ptr2);
    Sort2VecWithTest(input1, input2);
    _mm512_storeu_pd(ptr1, input1);
    _mm512_storeu_pd(ptr2, input2);
}


inline void Merge2Vec(__m512d& input1, __m512d& input2 ){
    __m512i reverseIdx = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    __m512d input2rev = _mm512_permutexvar_pd(reverseIdx, input2);

    __mmask8 compDoExchange = _mm512_cmp_pd_mask(input1, input2rev, _MM_CMPINT_GT);

    if(compDoExchange == 0){
        return;
    }

    __m512d newInput1 = _mm512_mask_permutexvar_pd(input1, compDoExchange, reverseIdx, input2);
    __m512d newInput2 = _mm512_mask_permutexvar_pd(input1, ~compDoExchange, reverseIdx, input2);

    input1 = SortVecWithTest(newInput1);
    input2 = SortVecWithTest(_mm512_permutexvar_pd(reverseIdx, newInput2));
}

inline void Merge2Vec(double* __restrict__ ptr1, double* __restrict__ ptr2 ){
    __m512d input1 = _mm512_loadu_pd(ptr1);
    __m512d input2 = _mm512_loadu_pd(ptr2);
    Merge2Vec(input1, input2);
    _mm512_storeu_pd(ptr1, input1);
    _mm512_storeu_pd(ptr2, input2);
}


/// Int

inline __m512i SortVec(__m512i input){
    __m512i idxNoNeigh = _mm512_set_epi32(15, 13, 14, 11, 12, 9, 10, 7, 8, 5, 6, 3, 4, 1, 2, 0);
    __m512i idxNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    for( int idx = 0 ; idx < 8 ; ++idx){
        __m512i permNeighOdd = _mm512_permutexvar_epi32(idxNeigh, input);
        __mmask16 compMaskOdd = _mm512_cmp_epi32_mask(permNeighOdd, input, _MM_CMPINT_LT);
        input = _mm512_mask_mov_epi32(input, (compMaskOdd & 0x5555) | ((compMaskOdd & 0x5555)<<1), permNeighOdd);

        __m512i permNeighEven = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __mmask16 compMaskEven = _mm512_cmp_epi32_mask(permNeighEven, input, _MM_CMPINT_LT);
        input = _mm512_mask_mov_epi32(input, (compMaskEven & 0x2AAA) | ((compMaskEven & 0x2AAA)<<1), permNeighEven);
    }
    return input;
}

inline void SortVec(int* __restrict__ ptr1){
    _mm512_storeu_si512(ptr1, SortVec(_mm512_loadu_si512(ptr1)));
}

inline __m512i SortVecBitFull(__m512i input){
    {
        __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,
                                              6, 7, 4, 5, 2, 3, 0, 1);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(12, 13, 14, 15, 8, 9, 10, 11,
                                              4, 5, 6, 7, 0, 1, 2, 3);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,
                                              6, 7, 4, 5, 2, 3, 0, 1);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(8, 9, 10, 11, 12, 13, 14, 15,
                                              0, 1, 2, 3, 4, 5, 6, 7);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,
                                              5, 4, 7, 6, 1, 0, 3, 2);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,
                                              6, 7, 4, 5, 2, 3, 0, 1);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                              8, 9, 10, 11, 12, 13, 14, 15);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,
                                              3, 2, 1, 0, 7, 6, 5, 4);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,
                                              5, 4, 7, 6, 1, 0, 3, 2);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,
                                              6, 7, 4, 5, 2, 3, 0, 1);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);
    }

    return input;
}

inline void SortVecBitFull(int* __restrict__ ptr1){
    _mm512_storeu_si512(ptr1, SortVecBitFull(_mm512_loadu_si512(ptr1)));
}


inline __m512i SortVecWithTest(__m512i input){
    __m512i idxNoNeigh = _mm512_set_epi32(15, 13, 14, 11, 12, 9, 10, 7, 8, 5, 6, 3, 4, 1, 2, 0);
    __m512i idxNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    for( int idx = 0 ; idx < 8 ; ++idx){
        __m512i permNeighOdd = _mm512_permutexvar_epi32(idxNeigh, input);
        __mmask16 compMaskOdd = _mm512_cmp_epi32_mask(permNeighOdd, input, _MM_CMPINT_LT);
        input = _mm512_mask_mov_epi32(input, (compMaskOdd & 0x5555) | ((compMaskOdd & 0x5555)<<1), permNeighOdd);

        __m512i permNeighEven = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __mmask16 compMaskEven = _mm512_cmp_epi32_mask(permNeighEven, input, _MM_CMPINT_LT);
        input = _mm512_mask_mov_epi32(input, (compMaskEven & 0x2AAA) | ((compMaskEven & 0x2AAA)<<1), permNeighEven);

        if(compMaskOdd == 0 && compMaskEven == 0){
            break;
        }
    }
    return input;
}

inline void SortVecWithTest(int* __restrict__ ptr1){
    _mm512_storeu_si512(ptr1, SortVecWithTest(_mm512_loadu_si512(ptr1)));
}

inline void Sort2Vec(__m512i& input1, __m512i& input2 ){

    __m512i idxNoNeigh = _mm512_set_epi32(15, 13, 14, 11, 12, 9, 10, 7, 8, 5, 6, 3, 4, 1, 2, 0);
    __m512i idxNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    __m512i idxAll0 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    __m512i idxAll7 = _mm512_set_epi32(15, 15, 15, 15,15, 15, 15, 15,15, 15, 15, 15,15, 15, 15, 15);

    for( int idx = 0 ; idx < 16 ; ++idx){
        __m512i permNeighOdd1 = _mm512_permutexvar_epi32(idxNeigh, input1);
        __m512i permNeighOdd2 = _mm512_permutexvar_epi32(idxNeigh, input2);

        __mmask16 compMaskOdd1 = _mm512_cmp_epi32_mask(permNeighOdd1, input1, _MM_CMPINT_LT);
        __mmask16 compMaskOdd2 = _mm512_cmp_epi32_mask(permNeighOdd2, input2, _MM_CMPINT_LT);

        input1 = _mm512_mask_mov_epi32(input1, (compMaskOdd1 & 0x5555) | ((compMaskOdd1 & 0x5555)<<1), permNeighOdd1);
        input2 = _mm512_mask_mov_epi32(input2, (compMaskOdd2 & 0x5555) | ((compMaskOdd2 & 0x5555)<<1), permNeighOdd2);

        __m512i permNeighEven1 = _mm512_permutexvar_epi32(idxNoNeigh, input1);
        __m512i permNeighEven2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);

        __mmask16 compMaskEven1 = _mm512_cmp_epi32_mask(permNeighEven1, input1, _MM_CMPINT_LT);
        __mmask16 compMaskEven2 = _mm512_cmp_epi32_mask(permNeighEven2, input2, _MM_CMPINT_LT);

        input1 = _mm512_mask_mov_epi32(input1, (compMaskEven1 & 0x2AAA) | ((compMaskEven1 & 0x2AAA)<<1), permNeighEven1);
        input2 = _mm512_mask_mov_epi32(input2, (compMaskEven2 & 0x2AAA) | ((compMaskEven2 & 0x2AAA)<<1), permNeighEven2);

        // Exchange border if needed
        __m512i input1last = _mm512_permutexvar_epi32(idxAll7, input1);
        __m512i input2first = _mm512_permutexvar_epi32(idxAll0, input2);

        __mmask16 compDoExchange = _mm512_cmp_epi32_mask(input1last, input2first, _MM_CMPINT_GT);

        input1 = _mm512_mask_mov_epi32(input1, compDoExchange & 0x8000, input2first);
        input2 = _mm512_mask_mov_epi32(input2, compDoExchange & 1, input1last);
    }
}

inline void Sort2Vec(int* __restrict__ ptr1, int* __restrict__ ptr2 ){
    __m512i input1 = _mm512_loadu_si512(ptr1);
    __m512i input2 = _mm512_loadu_si512(ptr2);
    Sort2Vec(input1, input2);
    _mm512_storeu_si512(ptr1, input1);
    _mm512_storeu_si512(ptr2, input2);
}

inline void ExchangeAndSort(__m512i& input, __m512i& input2 ){
    {
        __m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                              8, 9, 10, 11, 12, 13, 14, 15);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        input = _mm512_min_epi32(input2, permNeigh);
        input2 = _mm512_max_epi32(input2, permNeigh);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,
                                              15, 14, 13, 12, 11, 10, 9, 8);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xFF00, permNeighMax2);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,
                                              3, 2, 1, 0, 7, 6, 5, 4);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xF0F0, permNeighMax2);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,
                                              5, 4, 7, 6, 1, 0, 3, 2);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,
                                              6, 7, 4, 5, 2, 3, 0, 1);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);
    }
}

inline void Sort2VecBitFull(__m512i& input, __m512i& input2 ){
    {
        __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,
                                              6, 7, 4, 5, 2, 3, 0, 1);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(12, 13, 14, 15, 8, 9, 10, 11,
                                              4, 5, 6, 7, 0, 1, 2, 3);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,
                                              6, 7, 4, 5, 2, 3, 0, 1);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(8, 9, 10, 11, 12, 13, 14, 15,
                                              0, 1, 2, 3, 4, 5, 6, 7);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xF0F0, permNeighMax2);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,
                                              5, 4, 7, 6, 1, 0, 3, 2);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,
                                              6, 7, 4, 5, 2, 3, 0, 1);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                              8, 9, 10, 11, 12, 13, 14, 15);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xFF00, permNeighMax2);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,
                                              3, 2, 1, 0, 7, 6, 5, 4);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xF0F0, permNeighMax2);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,
                                              5, 4, 7, 6, 1, 0, 3, 2);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,
                                              6, 7, 4, 5, 2, 3, 0, 1);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);
    }
    ExchangeAndSort(input,input2);
}

inline void Sort2VecBitFull(int* __restrict__ ptr1, int* __restrict__ ptr2 ){
    __m512i input1 = _mm512_loadu_si512(ptr1);
    __m512i input2 = _mm512_loadu_si512(ptr2);
    Sort2VecBitFull(input1, input2);
    _mm512_storeu_si512(ptr1, input1);
    _mm512_storeu_si512(ptr2, input2);
}


inline void Sort3Vec(__m512i& input1, __m512i& input2, __m512i& input3 ){

    __m512i idxNoNeigh = _mm512_set_epi32(15, 13, 14, 11, 12, 9, 10, 7, 8, 5, 6, 3, 4, 1, 2, 0);
    __m512i idxNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    __m512i idxAll0 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    __m512i idxAll7 = _mm512_set_epi32(15, 15, 15, 15,15, 15, 15, 15,15, 15, 15, 15,15, 15, 15, 15);

    for( int idx = 0 ; idx < 24 ; ++idx){
        __m512i permNeighOdd1 = _mm512_permutexvar_epi32(idxNeigh, input1);
        __m512i permNeighOdd2 = _mm512_permutexvar_epi32(idxNeigh, input2);
        __m512i permNeighOdd3 = _mm512_permutexvar_epi32(idxNeigh, input3);

        __mmask16 compMaskOdd1 = _mm512_cmp_epi32_mask(permNeighOdd1, input1, _MM_CMPINT_LT);
        __mmask16 compMaskOdd2 = _mm512_cmp_epi32_mask(permNeighOdd2, input2, _MM_CMPINT_LT);
        __mmask16 compMaskOdd3 = _mm512_cmp_epi32_mask(permNeighOdd3, input3, _MM_CMPINT_LT);

        input1 = _mm512_mask_mov_epi32(input1, (compMaskOdd1 & 0x5555) | ((compMaskOdd1 & 0x5555)<<1), permNeighOdd1);
        input2 = _mm512_mask_mov_epi32(input2, (compMaskOdd2 & 0x5555) | ((compMaskOdd2 & 0x5555)<<1), permNeighOdd2);
        input3 = _mm512_mask_mov_epi32(input3, (compMaskOdd3 & 0x5555) | ((compMaskOdd3 & 0x5555)<<1), permNeighOdd3);

        __m512i permNeighEven1 = _mm512_permutexvar_epi32(idxNoNeigh, input1);
        __m512i permNeighEven2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeighEven3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);

        __mmask16 compMaskEven1 = _mm512_cmp_epi32_mask(permNeighEven1, input1, _MM_CMPINT_LT);
        __mmask16 compMaskEven2 = _mm512_cmp_epi32_mask(permNeighEven2, input2, _MM_CMPINT_LT);
        __mmask16 compMaskEven3 = _mm512_cmp_epi32_mask(permNeighEven3, input3, _MM_CMPINT_LT);

        input1 = _mm512_mask_mov_epi32(input1, (compMaskEven1 & 0x2AAA) | ((compMaskEven1 & 0x2AAA)<<1), permNeighEven1);
        input2 = _mm512_mask_mov_epi32(input2, (compMaskEven2 & 0x2AAA) | ((compMaskEven2 & 0x2AAA)<<1), permNeighEven2);
        input3 = _mm512_mask_mov_epi32(input3, (compMaskEven3 & 0x2AAA) | ((compMaskEven3 & 0x2AAA)<<1), permNeighEven3);

        // Exchange border if needed
        __m512i input1last = _mm512_permutexvar_epi32(idxAll7, input1);
        __m512i input2first = _mm512_permutexvar_epi32(idxAll0, input2);

        __mmask16 compDoExchange = _mm512_cmp_epi32_mask(input1last, input2first, _MM_CMPINT_GT);

        input1 = _mm512_mask_mov_epi32(input1, compDoExchange & 0x8000, input2first);
        input2 = _mm512_mask_mov_epi32(input2, compDoExchange & 1, input1last);

        __m512i input2last = _mm512_permutexvar_epi32(idxAll7, input2);
        __m512i input3first = _mm512_permutexvar_epi32(idxAll0, input3);

        __mmask16 compDoExchange23 = _mm512_cmp_epi32_mask(input2last, input3first, _MM_CMPINT_GT);

        input2 = _mm512_mask_mov_epi32(input2, compDoExchange23 & 0x8000, input3first);
        input3 = _mm512_mask_mov_epi32(input3, compDoExchange23 & 1, input2last);
    }
}

inline void Sort3Vec(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3 ){
    __m512i input1 = _mm512_loadu_si512(ptr1);
    __m512i input2 = _mm512_loadu_si512(ptr2);
    __m512i input3 = _mm512_loadu_si512(ptr3);
    Sort3Vec(input1, input2, input3);
    _mm512_storeu_si512(ptr1, input1);
    _mm512_storeu_si512(ptr2, input2);
    _mm512_storeu_si512(ptr3, input3);
}


inline void Sort3VecBitFull(__m512i& input, __m512i& input2, __m512i& input3 ){
    Sort2VecBitFull(input, input2);
    input3 = SortVecBitFull(input3);
    {
        __m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                              8, 9, 10, 11, 12, 13, 14, 15);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        input3 = _mm512_max_epi32(input2, permNeigh);
        input2 = _mm512_min_epi32(input2, permNeigh);
    }
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input2, inputCopy);
        input2 = _mm512_max_epi32(input2, inputCopy);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,
                                              15, 14, 13, 12, 11, 10, 9, 8);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xFF00, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xFF00, permNeighMax3);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,
                                              3, 2, 1, 0, 7, 6, 5, 4);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xF0F0, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xF0F0, permNeighMax3);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,
                                              5, 4, 7, 6, 1, 0, 3, 2);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xCCCC, permNeighMax3);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,
                                              6, 7, 4, 5, 2, 3, 0, 1);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xAAAA, permNeighMax3);
    }
}

inline void Sort3VecBitFull(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3 ){
    __m512i input1 = _mm512_loadu_si512(ptr1);
    __m512i input2 = _mm512_loadu_si512(ptr2);
    __m512i input3 = _mm512_loadu_si512(ptr3);
    Sort3VecBitFull(input1, input2, input3);
    _mm512_storeu_si512(ptr1, input1);
    _mm512_storeu_si512(ptr2, input2);
    _mm512_storeu_si512(ptr3, input3);
}


inline void Sort4Vec(__m512i& input1, __m512i& input2, __m512i& input3, __m512i& input4 ){

    __m512i idxNoNeigh = _mm512_set_epi32(15, 13, 14, 11, 12, 9, 10, 7, 8, 5, 6, 3, 4, 1, 2, 0);
    __m512i idxNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    __m512i idxAll0 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    __m512i idxAll7 = _mm512_set_epi32(15, 15, 15, 15,15, 15, 15, 15,15, 15, 15, 15,15, 15, 15, 15);

    for( int idx = 0 ; idx < 32 ; ++idx){
        __m512i permNeighOdd1 = _mm512_permutexvar_epi32(idxNeigh, input1);
        __m512i permNeighOdd2 = _mm512_permutexvar_epi32(idxNeigh, input2);
        __m512i permNeighOdd3 = _mm512_permutexvar_epi32(idxNeigh, input3);
        __m512i permNeighOdd4 = _mm512_permutexvar_epi32(idxNeigh, input4);

        __mmask16 compMaskOdd1 = _mm512_cmp_epi32_mask(permNeighOdd1, input1, _MM_CMPINT_LT);
        __mmask16 compMaskOdd2 = _mm512_cmp_epi32_mask(permNeighOdd2, input2, _MM_CMPINT_LT);
        __mmask16 compMaskOdd3 = _mm512_cmp_epi32_mask(permNeighOdd3, input3, _MM_CMPINT_LT);
        __mmask16 compMaskOdd4 = _mm512_cmp_epi32_mask(permNeighOdd4, input4, _MM_CMPINT_LT);

        input1 = _mm512_mask_mov_epi32(input1, (compMaskOdd1 & 0x5555) | ((compMaskOdd1 & 0x5555)<<1), permNeighOdd1);
        input2 = _mm512_mask_mov_epi32(input2, (compMaskOdd2 & 0x5555) | ((compMaskOdd2 & 0x5555)<<1), permNeighOdd2);
        input3 = _mm512_mask_mov_epi32(input3, (compMaskOdd3 & 0x5555) | ((compMaskOdd3 & 0x5555)<<1), permNeighOdd3);
        input4 = _mm512_mask_mov_epi32(input4, (compMaskOdd4 & 0x5555) | ((compMaskOdd4 & 0x5555)<<1), permNeighOdd4);

        __m512i permNeighEven1 = _mm512_permutexvar_epi32(idxNoNeigh, input1);
        __m512i permNeighEven2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeighEven3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeighEven4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);

        __mmask16 compMaskEven1 = _mm512_cmp_epi32_mask(permNeighEven1, input1, _MM_CMPINT_LT);
        __mmask16 compMaskEven2 = _mm512_cmp_epi32_mask(permNeighEven2, input2, _MM_CMPINT_LT);
        __mmask16 compMaskEven3 = _mm512_cmp_epi32_mask(permNeighEven3, input3, _MM_CMPINT_LT);
        __mmask16 compMaskEven4 = _mm512_cmp_epi32_mask(permNeighEven4, input4, _MM_CMPINT_LT);

        input1 = _mm512_mask_mov_epi32(input1, (compMaskEven1 & 0x2AAA) | ((compMaskEven1 & 0x2AAA)<<1), permNeighEven1);
        input2 = _mm512_mask_mov_epi32(input2, (compMaskEven2 & 0x2AAA) | ((compMaskEven2 & 0x2AAA)<<1), permNeighEven2);
        input3 = _mm512_mask_mov_epi32(input3, (compMaskEven3 & 0x2AAA) | ((compMaskEven3 & 0x2AAA)<<1), permNeighEven3);
        input4 = _mm512_mask_mov_epi32(input4, (compMaskEven4 & 0x2AAA) | ((compMaskEven4 & 0x2AAA)<<1), permNeighEven4);

        // Exchange border if needed
        __m512i input1last = _mm512_permutexvar_epi32(idxAll7, input1);
        __m512i input2first = _mm512_permutexvar_epi32(idxAll0, input2);

        __mmask16 compDoExchange = _mm512_cmp_epi32_mask(input1last, input2first, _MM_CMPINT_GT);

        input1 = _mm512_mask_mov_epi32(input1, compDoExchange & 0x8000, input2first);
        input2 = _mm512_mask_mov_epi32(input2, compDoExchange & 1, input1last);

        __m512i input2last = _mm512_permutexvar_epi32(idxAll7, input2);
        __m512i input3first = _mm512_permutexvar_epi32(idxAll0, input3);

        __mmask16 compDoExchange23 = _mm512_cmp_epi32_mask(input2last, input3first, _MM_CMPINT_GT);

        input2 = _mm512_mask_mov_epi32(input2, compDoExchange23 & 0x8000, input3first);
        input3 = _mm512_mask_mov_epi32(input3, compDoExchange23 & 1, input2last);

        __m512i input3last = _mm512_permutexvar_epi32(idxAll7, input3);
        __m512i input4first = _mm512_permutexvar_epi32(idxAll0, input4);

        __mmask16 compDoExchange34 = _mm512_cmp_epi32_mask(input3last, input4first, _MM_CMPINT_GT);

        input3 = _mm512_mask_mov_epi32(input3, compDoExchange34 & 0x8000, input4first);
        input4 = _mm512_mask_mov_epi32(input4, compDoExchange34 & 1, input3last);
    }
}

inline void Sort4Vec(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4 ){
    __m512i input1 = _mm512_loadu_si512(ptr1);
    __m512i input2 = _mm512_loadu_si512(ptr2);
    __m512i input3 = _mm512_loadu_si512(ptr3);
    __m512i input4 = _mm512_loadu_si512(ptr4);
    Sort4Vec(input1, input2, input3, input4);
    _mm512_storeu_si512(ptr1, input1);
    _mm512_storeu_si512(ptr2, input2);
    _mm512_storeu_si512(ptr3, input3);
    _mm512_storeu_si512(ptr4, input4);
}

inline void Sort4VecBitFull(__m512i& input, __m512i& input2, __m512i& input3, __m512i& input4 ){
    Sort2VecBitFull(input, input2);
    Sort2VecBitFull(input3, input4);
    {
        __m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                              8, 9, 10, 11, 12, 13, 14, 15);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);

        input4 = _mm512_max_epi32(input, permNeigh4);
        input = _mm512_min_epi32(input, permNeigh4);

        input3 = _mm512_max_epi32(input2, permNeigh3);
        input2 = _mm512_min_epi32(input2, permNeigh3);
    }
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input2, inputCopy);
        input2 = _mm512_max_epi32(input2, inputCopy);
    }
    {
        __m512i inputCopy = input3;
        input3 = _mm512_min_epi32(input4, inputCopy);
        input4 = _mm512_max_epi32(input4, inputCopy);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,
                                              15, 14, 13, 12, 11, 10, 9, 8);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xFF00, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xFF00, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xFF00, permNeighMax4);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,
                                              3, 2, 1, 0, 7, 6, 5, 4);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xF0F0, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xF0F0, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xF0F0, permNeighMax4);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,
                                              5, 4, 7, 6, 1, 0, 3, 2);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xCCCC, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xCCCC, permNeighMax4);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,
                                              6, 7, 4, 5, 2, 3, 0, 1);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xAAAA, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xAAAA, permNeighMax4);
    }
}

inline void Sort4VecBitFull(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4 ){
    __m512i input1 = _mm512_loadu_si512(ptr1);
    __m512i input2 = _mm512_loadu_si512(ptr2);
    __m512i input3 = _mm512_loadu_si512(ptr3);
    __m512i input4 = _mm512_loadu_si512(ptr4);
    Sort4VecBitFull(input1, input2, input3, input4);
    _mm512_storeu_si512(ptr1, input1);
    _mm512_storeu_si512(ptr2, input2);
    _mm512_storeu_si512(ptr3, input3);
    _mm512_storeu_si512(ptr4, input4);
}


inline void Sort5VecBitFull(__m512i& input, __m512i& input2, __m512i& input3, __m512i& input4, __m512i& input5 ){
    Sort4VecBitFull(input, input2, input3, input4);
    input5 = SortVecBitFull(input5);
    {
        __m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                              8, 9, 10, 11, 12, 13, 14, 15);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);

        input5 = _mm512_max_epi32(input4, permNeigh5);
        input4 = _mm512_min_epi32(input4, permNeigh5);
    }
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input3, inputCopy);
        input3 = _mm512_max_epi32(input3, inputCopy);
    }
    {
        __m512i inputCopy = input2;
        input2 = _mm512_min_epi32(input4, inputCopy);
        input4 = _mm512_max_epi32(input4, inputCopy);
    }
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input2, inputCopy);
        input2 = _mm512_max_epi32(input2, inputCopy);
    }
    {
        __m512i inputCopy = input3;
        input3 = _mm512_min_epi32(input4, inputCopy);
        input4 = _mm512_max_epi32(input4, inputCopy);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,
                                              15, 14, 13, 12, 11, 10, 9, 8);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xFF00, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xFF00, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xFF00, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xFF00, permNeighMax5);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,
                                              3, 2, 1, 0, 7, 6, 5, 4);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xF0F0, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xF0F0, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xF0F0, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xF0F0, permNeighMax5);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,
                                              5, 4, 7, 6, 1, 0, 3, 2);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xCCCC, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xCCCC, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xCCCC, permNeighMax5);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,
                                              6, 7, 4, 5, 2, 3, 0, 1);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xAAAA, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xAAAA, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xAAAA, permNeighMax5);
    }
}

inline void Sort5VecBitFull(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4, int* __restrict__ ptr5 ){
    __m512i input1 = _mm512_loadu_si512(ptr1);
    __m512i input2 = _mm512_loadu_si512(ptr2);
    __m512i input3 = _mm512_loadu_si512(ptr3);
    __m512i input4 = _mm512_loadu_si512(ptr4);
    __m512i input5 = _mm512_loadu_si512(ptr5);
    Sort5VecBitFull(input1, input2, input3, input4, input5);
    _mm512_storeu_si512(ptr1, input1);
    _mm512_storeu_si512(ptr2, input2);
    _mm512_storeu_si512(ptr3, input3);
    _mm512_storeu_si512(ptr4, input4);
    _mm512_storeu_si512(ptr5, input5);
}


inline void Sort6VecBitFull(__m512i& input, __m512i& input2, __m512i& input3, __m512i& input4,
                            __m512i& input5, __m512i& input6){
    Sort4VecBitFull(input, input2, input3, input4);
    Sort2VecBitFull(input5, input6);
    {
        __m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                              8, 9, 10, 11, 12, 13, 14, 15);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);

        input5 = _mm512_max_epi32(input4, permNeigh5);
        input6 = _mm512_max_epi32(input3, permNeigh6);

        input4 = _mm512_min_epi32(input4, permNeigh5);
        input3 = _mm512_min_epi32(input3, permNeigh6);
    }
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input3, inputCopy);
        input3 = _mm512_max_epi32(input3, inputCopy);
    }
    {
        __m512i inputCopy = input2;
        input2 = _mm512_min_epi32(input4, inputCopy);
        input4 = _mm512_max_epi32(input4, inputCopy);
    }
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input2, inputCopy);
        input2 = _mm512_max_epi32(input2, inputCopy);
    }
    {
        __m512i inputCopy = input3;
        input3 = _mm512_min_epi32(input4, inputCopy);
        input4 = _mm512_max_epi32(input4, inputCopy);
    }
    {
        __m512i inputCopy = input5;
        input5 = _mm512_min_epi32(input6, inputCopy);
        input6 = _mm512_max_epi32(input6, inputCopy);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,
                                              15, 14, 13, 12, 11, 10, 9, 8);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMin6 = _mm512_min_epi32(permNeigh6, input6);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        __m512i permNeighMax6 = _mm512_max_epi32(permNeigh6, input6);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xFF00, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xFF00, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xFF00, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xFF00, permNeighMax5);
        input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xFF00, permNeighMax6);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,
                                              3, 2, 1, 0, 7, 6, 5, 4);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMin6 = _mm512_min_epi32(permNeigh6, input6);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        __m512i permNeighMax6 = _mm512_max_epi32(permNeigh6, input6);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xF0F0, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xF0F0, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xF0F0, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xF0F0, permNeighMax5);
        input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xF0F0, permNeighMax6);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,
                                              5, 4, 7, 6, 1, 0, 3, 2);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMin6 = _mm512_min_epi32(permNeigh6, input6);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        __m512i permNeighMax6 = _mm512_max_epi32(permNeigh6, input6);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xCCCC, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xCCCC, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xCCCC, permNeighMax5);
        input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xCCCC, permNeighMax6);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,
                                              6, 7, 4, 5, 2, 3, 0, 1);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMin6 = _mm512_min_epi32(permNeigh6, input6);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        __m512i permNeighMax6 = _mm512_max_epi32(permNeigh6, input6);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xAAAA, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xAAAA, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xAAAA, permNeighMax5);
        input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xAAAA, permNeighMax6);
    }
}

inline void Sort6VecBitFull(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4,
            int* __restrict__ ptr5, int* __restrict__ ptr6 ){
    __m512i input1 = _mm512_loadu_si512(ptr1);
    __m512i input2 = _mm512_loadu_si512(ptr2);
    __m512i input3 = _mm512_loadu_si512(ptr3);
    __m512i input4 = _mm512_loadu_si512(ptr4);
    __m512i input5 = _mm512_loadu_si512(ptr5);
    __m512i input6 = _mm512_loadu_si512(ptr6);
    Sort6VecBitFull(input1, input2, input3, input4, input5, input6);
    _mm512_storeu_si512(ptr1, input1);
    _mm512_storeu_si512(ptr2, input2);
    _mm512_storeu_si512(ptr3, input3);
    _mm512_storeu_si512(ptr4, input4);
    _mm512_storeu_si512(ptr5, input5);
    _mm512_storeu_si512(ptr6, input6);
}


inline void Sort7VecBitFull(__m512i& input, __m512i& input2, __m512i& input3, __m512i& input4,
                            __m512i& input5, __m512i& input6, __m512i& input7){
    Sort4VecBitFull(input, input2, input3, input4);
    Sort3VecBitFull(input5, input6, input7);
    {
        __m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                              8, 9, 10, 11, 12, 13, 14, 15);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeigh7 = _mm512_permutexvar_epi32(idxNoNeigh, input7);

        input5 = _mm512_max_epi32(input4, permNeigh5);
        input6 = _mm512_max_epi32(input3, permNeigh6);
        input7 = _mm512_max_epi32(input2, permNeigh7);

        input4 = _mm512_min_epi32(input4, permNeigh5);
        input3 = _mm512_min_epi32(input3, permNeigh6);
        input2 = _mm512_min_epi32(input2, permNeigh7);
    }
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input3, inputCopy);
        input3 = _mm512_max_epi32(input3, inputCopy);
    }
    {
        __m512i inputCopy = input2;
        input2 = _mm512_min_epi32(input4, inputCopy);
        input4 = _mm512_max_epi32(input4, inputCopy);
    }
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input2, inputCopy);
        input2 = _mm512_max_epi32(input2, inputCopy);
    }
    {
        __m512i inputCopy = input3;
        input3 = _mm512_min_epi32(input4, inputCopy);
        input4 = _mm512_max_epi32(input4, inputCopy);
    }
    {
        __m512i inputCopy = input5;
        input5 = _mm512_min_epi32(input7, inputCopy);
        input7 = _mm512_max_epi32(input7, inputCopy);
    }
    {
        __m512i inputCopy = input5;
        input5 = _mm512_min_epi32(input6, inputCopy);
        input6 = _mm512_max_epi32(input6, inputCopy);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,
                                              15, 14, 13, 12, 11, 10, 9, 8);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeigh7 = _mm512_permutexvar_epi32(idxNoNeigh, input7);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMin6 = _mm512_min_epi32(permNeigh6, input6);
        __m512i permNeighMin7 = _mm512_min_epi32(permNeigh7, input7);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        __m512i permNeighMax6 = _mm512_max_epi32(permNeigh6, input6);
        __m512i permNeighMax7 = _mm512_max_epi32(permNeigh7, input7);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xFF00, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xFF00, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xFF00, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xFF00, permNeighMax5);
        input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xFF00, permNeighMax6);
        input7 = _mm512_mask_mov_epi32(permNeighMin7, 0xFF00, permNeighMax7);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,
                                              3, 2, 1, 0, 7, 6, 5, 4);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeigh7 = _mm512_permutexvar_epi32(idxNoNeigh, input7);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMin6 = _mm512_min_epi32(permNeigh6, input6);
        __m512i permNeighMin7 = _mm512_min_epi32(permNeigh7, input7);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        __m512i permNeighMax6 = _mm512_max_epi32(permNeigh6, input6);
        __m512i permNeighMax7 = _mm512_max_epi32(permNeigh7, input7);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xF0F0, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xF0F0, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xF0F0, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xF0F0, permNeighMax5);
        input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xF0F0, permNeighMax6);
        input7 = _mm512_mask_mov_epi32(permNeighMin7, 0xF0F0, permNeighMax7);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,
                                              5, 4, 7, 6, 1, 0, 3, 2);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeigh7 = _mm512_permutexvar_epi32(idxNoNeigh, input7);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMin6 = _mm512_min_epi32(permNeigh6, input6);
        __m512i permNeighMin7 = _mm512_min_epi32(permNeigh7, input7);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        __m512i permNeighMax6 = _mm512_max_epi32(permNeigh6, input6);
        __m512i permNeighMax7 = _mm512_max_epi32(permNeigh7, input7);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xCCCC, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xCCCC, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xCCCC, permNeighMax5);
        input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xCCCC, permNeighMax6);
        input7 = _mm512_mask_mov_epi32(permNeighMin7, 0xCCCC, permNeighMax7);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,
                                              6, 7, 4, 5, 2, 3, 0, 1);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeigh7 = _mm512_permutexvar_epi32(idxNoNeigh, input7);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMin6 = _mm512_min_epi32(permNeigh6, input6);
        __m512i permNeighMin7 = _mm512_min_epi32(permNeigh7, input7);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        __m512i permNeighMax6 = _mm512_max_epi32(permNeigh6, input6);
        __m512i permNeighMax7 = _mm512_max_epi32(permNeigh7, input7);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xAAAA, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xAAAA, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xAAAA, permNeighMax5);
        input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xAAAA, permNeighMax6);
        input7 = _mm512_mask_mov_epi32(permNeighMin7, 0xAAAA, permNeighMax7);
    }
}

inline void Sort7VecBitFull(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4,
            int* __restrict__ ptr5, int* __restrict__ ptr6, int* __restrict__ ptr7 ){
    __m512i input1 = _mm512_loadu_si512(ptr1);
    __m512i input2 = _mm512_loadu_si512(ptr2);
    __m512i input3 = _mm512_loadu_si512(ptr3);
    __m512i input4 = _mm512_loadu_si512(ptr4);
    __m512i input5 = _mm512_loadu_si512(ptr5);
    __m512i input6 = _mm512_loadu_si512(ptr6);
    __m512i input7 = _mm512_loadu_si512(ptr7);
    Sort7VecBitFull(input1, input2, input3, input4, input5, input6, input7);
    _mm512_storeu_si512(ptr1, input1);
    _mm512_storeu_si512(ptr2, input2);
    _mm512_storeu_si512(ptr3, input3);
    _mm512_storeu_si512(ptr4, input4);
    _mm512_storeu_si512(ptr5, input5);
    _mm512_storeu_si512(ptr6, input6);
    _mm512_storeu_si512(ptr7, input7);
}

inline void Sort8VecBitFull(__m512i& input, __m512i& input2, __m512i& input3, __m512i& input4,
                            __m512i& input5, __m512i& input6, __m512i& input7, __m512i& input8 ){
    Sort4VecBitFull(input, input2, input3, input4);
    Sort4VecBitFull(input5, input6, input7, input8);
    {
        __m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                              8, 9, 10, 11, 12, 13, 14, 15);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeigh7 = _mm512_permutexvar_epi32(idxNoNeigh, input7);
        __m512i permNeigh8 = _mm512_permutexvar_epi32(idxNoNeigh, input8);

        input5 = _mm512_max_epi32(input4, permNeigh5);
        input6 = _mm512_max_epi32(input3, permNeigh6);
        input7 = _mm512_max_epi32(input2, permNeigh7);
        input8 = _mm512_max_epi32(input, permNeigh8);

        input4 = _mm512_min_epi32(input4, permNeigh5);
        input3 = _mm512_min_epi32(input3, permNeigh6);
        input2 = _mm512_min_epi32(input2, permNeigh7);
        input = _mm512_min_epi32(input, permNeigh8);
    }
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input3, inputCopy);
        input3 = _mm512_max_epi32(input3, inputCopy);
    }
    {
        __m512i inputCopy = input2;
        input2 = _mm512_min_epi32(input4, inputCopy);
        input4 = _mm512_max_epi32(input4, inputCopy);
    }
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input2, inputCopy);
        input2 = _mm512_max_epi32(input2, inputCopy);
    }
    {
        __m512i inputCopy = input3;
        input3 = _mm512_min_epi32(input4, inputCopy);
        input4 = _mm512_max_epi32(input4, inputCopy);
    }
    {
        __m512i inputCopy = input5;
        input5 = _mm512_min_epi32(input7, inputCopy);
        input7 = _mm512_max_epi32(input7, inputCopy);
    }
    {
        __m512i inputCopy = input6;
        input6 = _mm512_min_epi32(input8, inputCopy);
        input8 = _mm512_max_epi32(input8, inputCopy);
    }
    {
        __m512i inputCopy = input5;
        input5 = _mm512_min_epi32(input6, inputCopy);
        input6 = _mm512_max_epi32(input6, inputCopy);
    }
    {
        __m512i inputCopy = input7;
        input7 = _mm512_min_epi32(input8, inputCopy);
        input8 = _mm512_max_epi32(input8, inputCopy);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,
                                              15, 14, 13, 12, 11, 10, 9, 8);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeigh7 = _mm512_permutexvar_epi32(idxNoNeigh, input7);
        __m512i permNeigh8 = _mm512_permutexvar_epi32(idxNoNeigh, input8);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMin6 = _mm512_min_epi32(permNeigh6, input6);
        __m512i permNeighMin7 = _mm512_min_epi32(permNeigh7, input7);
        __m512i permNeighMin8 = _mm512_min_epi32(permNeigh8, input8);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        __m512i permNeighMax6 = _mm512_max_epi32(permNeigh6, input6);
        __m512i permNeighMax7 = _mm512_max_epi32(permNeigh7, input7);
        __m512i permNeighMax8 = _mm512_max_epi32(permNeigh8, input8);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xFF00, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xFF00, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xFF00, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xFF00, permNeighMax5);
        input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xFF00, permNeighMax6);
        input7 = _mm512_mask_mov_epi32(permNeighMin7, 0xFF00, permNeighMax7);
        input8 = _mm512_mask_mov_epi32(permNeighMin8, 0xFF00, permNeighMax8);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,
                                              3, 2, 1, 0, 7, 6, 5, 4);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeigh7 = _mm512_permutexvar_epi32(idxNoNeigh, input7);
        __m512i permNeigh8 = _mm512_permutexvar_epi32(idxNoNeigh, input8);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMin6 = _mm512_min_epi32(permNeigh6, input6);
        __m512i permNeighMin7 = _mm512_min_epi32(permNeigh7, input7);
        __m512i permNeighMin8 = _mm512_min_epi32(permNeigh8, input8);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        __m512i permNeighMax6 = _mm512_max_epi32(permNeigh6, input6);
        __m512i permNeighMax7 = _mm512_max_epi32(permNeigh7, input7);
        __m512i permNeighMax8 = _mm512_max_epi32(permNeigh8, input8);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xF0F0, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xF0F0, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xF0F0, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xF0F0, permNeighMax5);
        input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xF0F0, permNeighMax6);
        input7 = _mm512_mask_mov_epi32(permNeighMin7, 0xF0F0, permNeighMax7);
        input8 = _mm512_mask_mov_epi32(permNeighMin8, 0xF0F0, permNeighMax8);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,
                                              5, 4, 7, 6, 1, 0, 3, 2);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeigh7 = _mm512_permutexvar_epi32(idxNoNeigh, input7);
        __m512i permNeigh8 = _mm512_permutexvar_epi32(idxNoNeigh, input8);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMin6 = _mm512_min_epi32(permNeigh6, input6);
        __m512i permNeighMin7 = _mm512_min_epi32(permNeigh7, input7);
        __m512i permNeighMin8 = _mm512_min_epi32(permNeigh8, input8);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        __m512i permNeighMax6 = _mm512_max_epi32(permNeigh6, input6);
        __m512i permNeighMax7 = _mm512_max_epi32(permNeigh7, input7);
        __m512i permNeighMax8 = _mm512_max_epi32(permNeigh8, input8);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xCCCC, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xCCCC, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xCCCC, permNeighMax5);
        input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xCCCC, permNeighMax6);
        input7 = _mm512_mask_mov_epi32(permNeighMin7, 0xCCCC, permNeighMax7);
        input8 = _mm512_mask_mov_epi32(permNeighMin8, 0xCCCC, permNeighMax8);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,
                                              6, 7, 4, 5, 2, 3, 0, 1);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeigh7 = _mm512_permutexvar_epi32(idxNoNeigh, input7);
        __m512i permNeigh8 = _mm512_permutexvar_epi32(idxNoNeigh, input8);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMin6 = _mm512_min_epi32(permNeigh6, input6);
        __m512i permNeighMin7 = _mm512_min_epi32(permNeigh7, input7);
        __m512i permNeighMin8 = _mm512_min_epi32(permNeigh8, input8);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        __m512i permNeighMax6 = _mm512_max_epi32(permNeigh6, input6);
        __m512i permNeighMax7 = _mm512_max_epi32(permNeigh7, input7);
        __m512i permNeighMax8 = _mm512_max_epi32(permNeigh8, input8);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xAAAA, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xAAAA, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xAAAA, permNeighMax5);
        input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xAAAA, permNeighMax6);
        input7 = _mm512_mask_mov_epi32(permNeighMin7, 0xAAAA, permNeighMax7);
        input8 = _mm512_mask_mov_epi32(permNeighMin8, 0xAAAA, permNeighMax8);
    }
}

inline void Sort8VecBitFull(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4,
            int* __restrict__ ptr5, int* __restrict__ ptr6, int* __restrict__ ptr7, int* __restrict__ ptr8 ){
    __m512i input1 = _mm512_loadu_si512(ptr1);
    __m512i input2 = _mm512_loadu_si512(ptr2);
    __m512i input3 = _mm512_loadu_si512(ptr3);
    __m512i input4 = _mm512_loadu_si512(ptr4);
    __m512i input5 = _mm512_loadu_si512(ptr5);
    __m512i input6 = _mm512_loadu_si512(ptr6);
    __m512i input7 = _mm512_loadu_si512(ptr7);
    __m512i input8 = _mm512_loadu_si512(ptr8);
    Sort8VecBitFull(input1, input2, input3, input4, input5, input6, input7, input8);
    _mm512_storeu_si512(ptr1, input1);
    _mm512_storeu_si512(ptr2, input2);
    _mm512_storeu_si512(ptr3, input3);
    _mm512_storeu_si512(ptr4, input4);
    _mm512_storeu_si512(ptr5, input5);
    _mm512_storeu_si512(ptr6, input6);
    _mm512_storeu_si512(ptr7, input7);
    _mm512_storeu_si512(ptr8, input8);
}

inline void Finish1VecBitFull(__m512i& input){
    {
        __m512i idxNoNeigh = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,
                                              15, 14, 13, 12, 11, 10, 9, 8);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,
                                              3, 2, 1, 0, 7, 6, 5, 4);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,
                                              5, 4, 7, 6, 1, 0, 3, 2);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,
                                              6, 7, 4, 5, 2, 3, 0, 1);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);
    }
}

inline void Finish2VecBitFull(__m512i& input, __m512i& input2){
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input2, inputCopy);
        input2 = _mm512_max_epi32(input2, inputCopy);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,
                                              15, 14, 13, 12, 11, 10, 9, 8);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xFF00, permNeighMax2);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,
                                              3, 2, 1, 0, 7, 6, 5, 4);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xF0F0, permNeighMax2);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,
                                              5, 4, 7, 6, 1, 0, 3, 2);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,
                                              6, 7, 4, 5, 2, 3, 0, 1);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);
    }
}

inline void Finish3VecBitFull(__m512i& input, __m512i& input2, __m512i& input3){
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input3, inputCopy);
        input3 = _mm512_max_epi32(input3, inputCopy);
    }
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input2, inputCopy);
        input2 = _mm512_max_epi32(input2, inputCopy);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,
                                              15, 14, 13, 12, 11, 10, 9, 8);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xFF00, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xFF00, permNeighMax3);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,
                                              3, 2, 1, 0, 7, 6, 5, 4);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xF0F0, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xF0F0, permNeighMax3);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,
                                              5, 4, 7, 6, 1, 0, 3, 2);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xCCCC, permNeighMax3);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,
                                              6, 7, 4, 5, 2, 3, 0, 1);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xAAAA, permNeighMax3);
    }
}

inline void Finish4VecBitFull(__m512i& input, __m512i& input2, __m512i& input3, __m512i& input4){
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input3, inputCopy);
        input3 = _mm512_max_epi32(input3, inputCopy);
    }
    {
        __m512i inputCopy = input2;
        input2 = _mm512_min_epi32(input4, inputCopy);
        input4 = _mm512_max_epi32(input4, inputCopy);
    }
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input2, inputCopy);
        input2 = _mm512_max_epi32(input2, inputCopy);
    }
    {
        __m512i inputCopy = input3;
        input3 = _mm512_min_epi32(input4, inputCopy);
        input4 = _mm512_max_epi32(input4, inputCopy);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,
                                              15, 14, 13, 12, 11, 10, 9, 8);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xFF00, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xFF00, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xFF00, permNeighMax4);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,
                                              3, 2, 1, 0, 7, 6, 5, 4);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xF0F0, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xF0F0, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xF0F0, permNeighMax4);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,
                                              5, 4, 7, 6, 1, 0, 3, 2);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xCCCC, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xCCCC, permNeighMax4);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,
                                              6, 7, 4, 5, 2, 3, 0, 1);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xAAAA, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xAAAA, permNeighMax4);
    }
}

inline void Finish5VecBitFull(__m512i& input, __m512i& input2, __m512i& input3, __m512i& input4,
                            __m512i& input5){
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input5, inputCopy);
        input5 = _mm512_max_epi32(input5, inputCopy);
    }
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input3, inputCopy);
        input3 = _mm512_max_epi32(input3, inputCopy);
    }
    {
        __m512i inputCopy = input2;
        input2 = _mm512_min_epi32(input4, inputCopy);
        input4 = _mm512_max_epi32(input4, inputCopy);
    }
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input2, inputCopy);
        input2 = _mm512_max_epi32(input2, inputCopy);
    }
    {
        __m512i inputCopy = input3;
        input3 = _mm512_min_epi32(input4, inputCopy);
        input4 = _mm512_max_epi32(input4, inputCopy);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,
                                              15, 14, 13, 12, 11, 10, 9, 8);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xFF00, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xFF00, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xFF00, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xFF00, permNeighMax5);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,
                                              3, 2, 1, 0, 7, 6, 5, 4);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xF0F0, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xF0F0, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xF0F0, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xF0F0, permNeighMax5);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,
                                              5, 4, 7, 6, 1, 0, 3, 2);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xCCCC, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xCCCC, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xCCCC, permNeighMax5);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,
                                              6, 7, 4, 5, 2, 3, 0, 1);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xAAAA, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xAAAA, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xAAAA, permNeighMax5);
    }
}

inline void Finish6VecBitFull(__m512i& input, __m512i& input2, __m512i& input3, __m512i& input4,
                            __m512i& input5, __m512i& input6){
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input5, inputCopy);
        input5 = _mm512_max_epi32(input5, inputCopy);
    }
    {
        __m512i inputCopy = input2;
        input2 = _mm512_min_epi32(input6, inputCopy);
        input6 = _mm512_max_epi32(input6, inputCopy);
    }
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input3, inputCopy);
        input3 = _mm512_max_epi32(input3, inputCopy);
    }
    {
        __m512i inputCopy = input2;
        input2 = _mm512_min_epi32(input4, inputCopy);
        input4 = _mm512_max_epi32(input4, inputCopy);
    }
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input2, inputCopy);
        input2 = _mm512_max_epi32(input2, inputCopy);
    }
    {
        __m512i inputCopy = input3;
        input3 = _mm512_min_epi32(input4, inputCopy);
        input4 = _mm512_max_epi32(input4, inputCopy);
    }
    {
        __m512i inputCopy = input5;
        input5 = _mm512_min_epi32(input6, inputCopy);
        input6 = _mm512_max_epi32(input6, inputCopy);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,
                                              15, 14, 13, 12, 11, 10, 9, 8);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMin6 = _mm512_min_epi32(permNeigh6, input6);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        __m512i permNeighMax6 = _mm512_max_epi32(permNeigh6, input6);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xFF00, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xFF00, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xFF00, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xFF00, permNeighMax5);
        input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xFF00, permNeighMax6);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,
                                              3, 2, 1, 0, 7, 6, 5, 4);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMin6 = _mm512_min_epi32(permNeigh6, input6);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        __m512i permNeighMax6 = _mm512_max_epi32(permNeigh6, input6);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xF0F0, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xF0F0, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xF0F0, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xF0F0, permNeighMax5);
        input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xF0F0, permNeighMax6);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,
                                              5, 4, 7, 6, 1, 0, 3, 2);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMin6 = _mm512_min_epi32(permNeigh6, input6);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        __m512i permNeighMax6 = _mm512_max_epi32(permNeigh6, input6);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xCCCC, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xCCCC, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xCCCC, permNeighMax5);
        input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xCCCC, permNeighMax6);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,
                                              6, 7, 4, 5, 2, 3, 0, 1);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMin6 = _mm512_min_epi32(permNeigh6, input6);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        __m512i permNeighMax6 = _mm512_max_epi32(permNeigh6, input6);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xAAAA, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xAAAA, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xAAAA, permNeighMax5);
        input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xAAAA, permNeighMax6);
    }
}

inline void Finish7VecBitFull(__m512i& input, __m512i& input2, __m512i& input3, __m512i& input4,
                            __m512i& input5, __m512i& input6, __m512i& input7){
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input5, inputCopy);
        input5 = _mm512_max_epi32(input5, inputCopy);
    }
    {
        __m512i inputCopy = input2;
        input2 = _mm512_min_epi32(input6, inputCopy);
        input6 = _mm512_max_epi32(input6, inputCopy);
    }
    {
        __m512i inputCopy = input3;
        input3 = _mm512_min_epi32(input7, inputCopy);
        input7 = _mm512_max_epi32(input7, inputCopy);
    }
    {
        __m512i inputCopy = input5;
        input5 = _mm512_min_epi32(input7, inputCopy);
        input7 = _mm512_max_epi32(input7, inputCopy);
    }
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input3, inputCopy);
        input3 = _mm512_max_epi32(input3, inputCopy);
    }
    {
        __m512i inputCopy = input2;
        input2 = _mm512_min_epi32(input4, inputCopy);
        input4 = _mm512_max_epi32(input4, inputCopy);
    }
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input2, inputCopy);
        input2 = _mm512_max_epi32(input2, inputCopy);
    }
    {
        __m512i inputCopy = input3;
        input3 = _mm512_min_epi32(input4, inputCopy);
        input4 = _mm512_max_epi32(input4, inputCopy);
    }
    {
        __m512i inputCopy = input5;
        input5 = _mm512_min_epi32(input7, inputCopy);
        input7 = _mm512_max_epi32(input7, inputCopy);
    }
    {
        __m512i inputCopy = input5;
        input5 = _mm512_min_epi32(input6, inputCopy);
        input6 = _mm512_max_epi32(input6, inputCopy);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,
                                              15, 14, 13, 12, 11, 10, 9, 8);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeigh7 = _mm512_permutexvar_epi32(idxNoNeigh, input7);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMin6 = _mm512_min_epi32(permNeigh6, input6);
        __m512i permNeighMin7 = _mm512_min_epi32(permNeigh7, input7);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        __m512i permNeighMax6 = _mm512_max_epi32(permNeigh6, input6);
        __m512i permNeighMax7 = _mm512_max_epi32(permNeigh7, input7);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xFF00, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xFF00, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xFF00, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xFF00, permNeighMax5);
        input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xFF00, permNeighMax6);
        input7 = _mm512_mask_mov_epi32(permNeighMin7, 0xFF00, permNeighMax7);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,
                                              3, 2, 1, 0, 7, 6, 5, 4);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeigh7 = _mm512_permutexvar_epi32(idxNoNeigh, input7);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMin6 = _mm512_min_epi32(permNeigh6, input6);
        __m512i permNeighMin7 = _mm512_min_epi32(permNeigh7, input7);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        __m512i permNeighMax6 = _mm512_max_epi32(permNeigh6, input6);
        __m512i permNeighMax7 = _mm512_max_epi32(permNeigh7, input7);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xF0F0, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xF0F0, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xF0F0, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xF0F0, permNeighMax5);
        input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xF0F0, permNeighMax6);
        input7 = _mm512_mask_mov_epi32(permNeighMin7, 0xF0F0, permNeighMax7);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,
                                              5, 4, 7, 6, 1, 0, 3, 2);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeigh7 = _mm512_permutexvar_epi32(idxNoNeigh, input7);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMin6 = _mm512_min_epi32(permNeigh6, input6);
        __m512i permNeighMin7 = _mm512_min_epi32(permNeigh7, input7);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        __m512i permNeighMax6 = _mm512_max_epi32(permNeigh6, input6);
        __m512i permNeighMax7 = _mm512_max_epi32(permNeigh7, input7);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xCCCC, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xCCCC, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xCCCC, permNeighMax5);
        input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xCCCC, permNeighMax6);
        input7 = _mm512_mask_mov_epi32(permNeighMin7, 0xCCCC, permNeighMax7);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,
                                              6, 7, 4, 5, 2, 3, 0, 1);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeigh7 = _mm512_permutexvar_epi32(idxNoNeigh, input7);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMin6 = _mm512_min_epi32(permNeigh6, input6);
        __m512i permNeighMin7 = _mm512_min_epi32(permNeigh7, input7);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        __m512i permNeighMax6 = _mm512_max_epi32(permNeigh6, input6);
        __m512i permNeighMax7 = _mm512_max_epi32(permNeigh7, input7);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xAAAA, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xAAAA, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xAAAA, permNeighMax5);
        input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xAAAA, permNeighMax6);
        input7 = _mm512_mask_mov_epi32(permNeighMin7, 0xAAAA, permNeighMax7);
    }
}

inline void Finish8VecBitFull(__m512i& input, __m512i& input2, __m512i& input3, __m512i& input4,
                            __m512i& input5, __m512i& input6, __m512i& input7, __m512i& input8 ){
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input5, inputCopy);
        input5 = _mm512_max_epi32(input5, inputCopy);
    }
    {
        __m512i inputCopy = input2;
        input2 = _mm512_min_epi32(input6, inputCopy);
        input6 = _mm512_max_epi32(input6, inputCopy);
    }
    {
        __m512i inputCopy = input3;
        input3 = _mm512_min_epi32(input7, inputCopy);
        input7 = _mm512_max_epi32(input7, inputCopy);
    }
    {
        __m512i inputCopy = input4;
        input4 = _mm512_min_epi32(input8, inputCopy);
        input8 = _mm512_max_epi32(input8, inputCopy);
    }
    {
        __m512i inputCopy = input5;
        input5 = _mm512_min_epi32(input7, inputCopy);
        input7 = _mm512_max_epi32(input7, inputCopy);
    }
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input3, inputCopy);
        input3 = _mm512_max_epi32(input3, inputCopy);
    }
    {
        __m512i inputCopy = input2;
        input2 = _mm512_min_epi32(input4, inputCopy);
        input4 = _mm512_max_epi32(input4, inputCopy);
    }
    {
        __m512i inputCopy = input;
        input = _mm512_min_epi32(input2, inputCopy);
        input2 = _mm512_max_epi32(input2, inputCopy);
    }
    {
        __m512i inputCopy = input3;
        input3 = _mm512_min_epi32(input4, inputCopy);
        input4 = _mm512_max_epi32(input4, inputCopy);
    }
    {
        __m512i inputCopy = input5;
        input5 = _mm512_min_epi32(input7, inputCopy);
        input7 = _mm512_max_epi32(input7, inputCopy);
    }
    {
        __m512i inputCopy = input6;
        input6 = _mm512_min_epi32(input8, inputCopy);
        input8 = _mm512_max_epi32(input8, inputCopy);
    }
    {
        __m512i inputCopy = input5;
        input5 = _mm512_min_epi32(input6, inputCopy);
        input6 = _mm512_max_epi32(input6, inputCopy);
    }
    {
        __m512i inputCopy = input7;
        input7 = _mm512_min_epi32(input8, inputCopy);
        input8 = _mm512_max_epi32(input8, inputCopy);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,
                                              15, 14, 13, 12, 11, 10, 9, 8);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeigh7 = _mm512_permutexvar_epi32(idxNoNeigh, input7);
        __m512i permNeigh8 = _mm512_permutexvar_epi32(idxNoNeigh, input8);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMin6 = _mm512_min_epi32(permNeigh6, input6);
        __m512i permNeighMin7 = _mm512_min_epi32(permNeigh7, input7);
        __m512i permNeighMin8 = _mm512_min_epi32(permNeigh8, input8);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        __m512i permNeighMax6 = _mm512_max_epi32(permNeigh6, input6);
        __m512i permNeighMax7 = _mm512_max_epi32(permNeigh7, input7);
        __m512i permNeighMax8 = _mm512_max_epi32(permNeigh8, input8);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xFF00, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xFF00, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xFF00, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xFF00, permNeighMax5);
        input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xFF00, permNeighMax6);
        input7 = _mm512_mask_mov_epi32(permNeighMin7, 0xFF00, permNeighMax7);
        input8 = _mm512_mask_mov_epi32(permNeighMin8, 0xFF00, permNeighMax8);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,
                                              3, 2, 1, 0, 7, 6, 5, 4);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeigh7 = _mm512_permutexvar_epi32(idxNoNeigh, input7);
        __m512i permNeigh8 = _mm512_permutexvar_epi32(idxNoNeigh, input8);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMin6 = _mm512_min_epi32(permNeigh6, input6);
        __m512i permNeighMin7 = _mm512_min_epi32(permNeigh7, input7);
        __m512i permNeighMin8 = _mm512_min_epi32(permNeigh8, input8);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        __m512i permNeighMax6 = _mm512_max_epi32(permNeigh6, input6);
        __m512i permNeighMax7 = _mm512_max_epi32(permNeigh7, input7);
        __m512i permNeighMax8 = _mm512_max_epi32(permNeigh8, input8);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xF0F0, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xF0F0, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xF0F0, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xF0F0, permNeighMax5);
        input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xF0F0, permNeighMax6);
        input7 = _mm512_mask_mov_epi32(permNeighMin7, 0xF0F0, permNeighMax7);
        input8 = _mm512_mask_mov_epi32(permNeighMin8, 0xF0F0, permNeighMax8);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,
                                              5, 4, 7, 6, 1, 0, 3, 2);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeigh7 = _mm512_permutexvar_epi32(idxNoNeigh, input7);
        __m512i permNeigh8 = _mm512_permutexvar_epi32(idxNoNeigh, input8);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMin6 = _mm512_min_epi32(permNeigh6, input6);
        __m512i permNeighMin7 = _mm512_min_epi32(permNeigh7, input7);
        __m512i permNeighMin8 = _mm512_min_epi32(permNeigh8, input8);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        __m512i permNeighMax6 = _mm512_max_epi32(permNeigh6, input6);
        __m512i permNeighMax7 = _mm512_max_epi32(permNeigh7, input7);
        __m512i permNeighMax8 = _mm512_max_epi32(permNeigh8, input8);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xCCCC, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xCCCC, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xCCCC, permNeighMax5);
        input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xCCCC, permNeighMax6);
        input7 = _mm512_mask_mov_epi32(permNeighMin7, 0xCCCC, permNeighMax7);
        input8 = _mm512_mask_mov_epi32(permNeighMin8, 0xCCCC, permNeighMax8);
    }
    {
        __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,
                                              6, 7, 4, 5, 2, 3, 0, 1);
        __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);
        __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);
        __m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);
        __m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);
        __m512i permNeigh7 = _mm512_permutexvar_epi32(idxNoNeigh, input7);
        __m512i permNeigh8 = _mm512_permutexvar_epi32(idxNoNeigh, input8);
        __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
        __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
        __m512i permNeighMin3 = _mm512_min_epi32(permNeigh3, input3);
        __m512i permNeighMin4 = _mm512_min_epi32(permNeigh4, input4);
        __m512i permNeighMin5 = _mm512_min_epi32(permNeigh5, input5);
        __m512i permNeighMin6 = _mm512_min_epi32(permNeigh6, input6);
        __m512i permNeighMin7 = _mm512_min_epi32(permNeigh7, input7);
        __m512i permNeighMin8 = _mm512_min_epi32(permNeigh8, input8);
        __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
        __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
        __m512i permNeighMax3 = _mm512_max_epi32(permNeigh3, input3);
        __m512i permNeighMax4 = _mm512_max_epi32(permNeigh4, input4);
        __m512i permNeighMax5 = _mm512_max_epi32(permNeigh5, input5);
        __m512i permNeighMax6 = _mm512_max_epi32(permNeigh6, input6);
        __m512i permNeighMax7 = _mm512_max_epi32(permNeigh7, input7);
        __m512i permNeighMax8 = _mm512_max_epi32(permNeigh8, input8);
        input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);
        input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);
        input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xAAAA, permNeighMax3);
        input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xAAAA, permNeighMax4);
        input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xAAAA, permNeighMax5);
        input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xAAAA, permNeighMax6);
        input7 = _mm512_mask_mov_epi32(permNeighMin7, 0xAAAA, permNeighMax7);
        input8 = _mm512_mask_mov_epi32(permNeighMin8, 0xAAAA, permNeighMax8);
    }
}

inline void Sort9VecBitFull(__m512i& input, __m512i& input2, __m512i& input3, __m512i& input4,
                            __m512i& input5, __m512i& input6, __m512i& input7, __m512i& input8,
                            __m512i& input9){
    Sort8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    input9 = SortVecBitFull(input9);
    {
        __m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                              8, 9, 10, 11, 12, 13, 14, 15);
        __m512i permNeigh9 = _mm512_permutexvar_epi32(idxNoNeigh, input9);

        input9 = _mm512_max_epi32(input8, permNeigh9);

        input8 = _mm512_min_epi32(input8, permNeigh9);
    }
    Finish8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Finish1VecBitFull(input9);
}

inline void Sort9VecBitFull(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4,
            int* __restrict__ ptr5, int* __restrict__ ptr6, int* __restrict__ ptr7, int* __restrict__ ptr8,
                             int* __restrict__ ptr9){
    __m512i input1 = _mm512_loadu_si512(ptr1);
    __m512i input2 = _mm512_loadu_si512(ptr2);
    __m512i input3 = _mm512_loadu_si512(ptr3);
    __m512i input4 = _mm512_loadu_si512(ptr4);
    __m512i input5 = _mm512_loadu_si512(ptr5);
    __m512i input6 = _mm512_loadu_si512(ptr6);
    __m512i input7 = _mm512_loadu_si512(ptr7);
    __m512i input8 = _mm512_loadu_si512(ptr8);
    __m512i input9 = _mm512_loadu_si512(ptr9);
    Sort9VecBitFull(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9);
    _mm512_storeu_si512(ptr1, input1);
    _mm512_storeu_si512(ptr2, input2);
    _mm512_storeu_si512(ptr3, input3);
    _mm512_storeu_si512(ptr4, input4);
    _mm512_storeu_si512(ptr5, input5);
    _mm512_storeu_si512(ptr6, input6);
    _mm512_storeu_si512(ptr7, input7);
    _mm512_storeu_si512(ptr8, input8);
    _mm512_storeu_si512(ptr9, input9);
}

inline void Sort10VecBitFull(__m512i& input, __m512i& input2, __m512i& input3, __m512i& input4,
                            __m512i& input5, __m512i& input6, __m512i& input7, __m512i& input8,
                            __m512i& input9, __m512i& input10){
    Sort8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Sort2VecBitFull(input9, input10);
    {
        __m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                              8, 9, 10, 11, 12, 13, 14, 15);
        __m512i permNeigh9 = _mm512_permutexvar_epi32(idxNoNeigh, input9);
        __m512i permNeigh10 = _mm512_permutexvar_epi32(idxNoNeigh, input10);

        input9 = _mm512_max_epi32(input8, permNeigh9);
        input10 = _mm512_max_epi32(input7, permNeigh10);

        input8 = _mm512_min_epi32(input8, permNeigh9);
        input7 = _mm512_min_epi32(input7, permNeigh10);
    }
    Finish8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Finish2VecBitFull(input9, input10);
}

inline void Sort10VecBitFull(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4,
            int* __restrict__ ptr5, int* __restrict__ ptr6, int* __restrict__ ptr7, int* __restrict__ ptr8,
                             int* __restrict__ ptr9, int* __restrict__ ptr10){
    __m512i input1 = _mm512_loadu_si512(ptr1);
    __m512i input2 = _mm512_loadu_si512(ptr2);
    __m512i input3 = _mm512_loadu_si512(ptr3);
    __m512i input4 = _mm512_loadu_si512(ptr4);
    __m512i input5 = _mm512_loadu_si512(ptr5);
    __m512i input6 = _mm512_loadu_si512(ptr6);
    __m512i input7 = _mm512_loadu_si512(ptr7);
    __m512i input8 = _mm512_loadu_si512(ptr8);
    __m512i input9 = _mm512_loadu_si512(ptr9);
    __m512i input10 = _mm512_loadu_si512(ptr10);
    Sort10VecBitFull(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10);
    _mm512_storeu_si512(ptr1, input1);
    _mm512_storeu_si512(ptr2, input2);
    _mm512_storeu_si512(ptr3, input3);
    _mm512_storeu_si512(ptr4, input4);
    _mm512_storeu_si512(ptr5, input5);
    _mm512_storeu_si512(ptr6, input6);
    _mm512_storeu_si512(ptr7, input7);
    _mm512_storeu_si512(ptr8, input8);
    _mm512_storeu_si512(ptr9, input9);
    _mm512_storeu_si512(ptr10, input10);
}

inline void Sort11VecBitFull(__m512i& input, __m512i& input2, __m512i& input3, __m512i& input4,
                            __m512i& input5, __m512i& input6, __m512i& input7, __m512i& input8,
                            __m512i& input9, __m512i& input10, __m512i& input11){
    Sort8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Sort3VecBitFull(input9, input10, input11);
    {
        __m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                              8, 9, 10, 11, 12, 13, 14, 15);
        __m512i permNeigh9 = _mm512_permutexvar_epi32(idxNoNeigh, input9);
        __m512i permNeigh10 = _mm512_permutexvar_epi32(idxNoNeigh, input10);
        __m512i permNeigh11 = _mm512_permutexvar_epi32(idxNoNeigh, input11);

        input9 = _mm512_max_epi32(input8, permNeigh9);
        input10 = _mm512_max_epi32(input7, permNeigh10);
        input11 = _mm512_max_epi32(input6, permNeigh11);

        input8 = _mm512_min_epi32(input8, permNeigh9);
        input7 = _mm512_min_epi32(input7, permNeigh10);
        input6 = _mm512_min_epi32(input6, permNeigh11);
    }
    Finish8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Finish3VecBitFull(input9, input10, input11);
}

inline void Sort11VecBitFull(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4,
            int* __restrict__ ptr5, int* __restrict__ ptr6, int* __restrict__ ptr7, int* __restrict__ ptr8,
                             int* __restrict__ ptr9, int* __restrict__ ptr10, int* __restrict__ ptr11){
    __m512i input1 = _mm512_loadu_si512(ptr1);
    __m512i input2 = _mm512_loadu_si512(ptr2);
    __m512i input3 = _mm512_loadu_si512(ptr3);
    __m512i input4 = _mm512_loadu_si512(ptr4);
    __m512i input5 = _mm512_loadu_si512(ptr5);
    __m512i input6 = _mm512_loadu_si512(ptr6);
    __m512i input7 = _mm512_loadu_si512(ptr7);
    __m512i input8 = _mm512_loadu_si512(ptr8);
    __m512i input9 = _mm512_loadu_si512(ptr9);
    __m512i input10 = _mm512_loadu_si512(ptr10);
    __m512i input11 = _mm512_loadu_si512(ptr11);
    Sort11VecBitFull(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10, input11);
    _mm512_storeu_si512(ptr1, input1);
    _mm512_storeu_si512(ptr2, input2);
    _mm512_storeu_si512(ptr3, input3);
    _mm512_storeu_si512(ptr4, input4);
    _mm512_storeu_si512(ptr5, input5);
    _mm512_storeu_si512(ptr6, input6);
    _mm512_storeu_si512(ptr7, input7);
    _mm512_storeu_si512(ptr8, input8);
    _mm512_storeu_si512(ptr9, input9);
    _mm512_storeu_si512(ptr10, input10);
    _mm512_storeu_si512(ptr11, input11);
}

inline void Sort12VecBitFull(__m512i& input, __m512i& input2, __m512i& input3, __m512i& input4,
                            __m512i& input5, __m512i& input6, __m512i& input7, __m512i& input8,
                            __m512i& input9, __m512i& input10, __m512i& input11, __m512i& input12 ){
    Sort8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Sort4VecBitFull(input9, input10, input11, input12);
    {
        __m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                              8, 9, 10, 11, 12, 13, 14, 15);
        __m512i permNeigh9 = _mm512_permutexvar_epi32(idxNoNeigh, input9);
        __m512i permNeigh10 = _mm512_permutexvar_epi32(idxNoNeigh, input10);
        __m512i permNeigh11 = _mm512_permutexvar_epi32(idxNoNeigh, input11);
        __m512i permNeigh12 = _mm512_permutexvar_epi32(idxNoNeigh, input12);

        input9 = _mm512_max_epi32(input8, permNeigh9);
        input10 = _mm512_max_epi32(input7, permNeigh10);
        input11 = _mm512_max_epi32(input6, permNeigh11);
        input12 = _mm512_max_epi32(input5, permNeigh12);

        input8 = _mm512_min_epi32(input8, permNeigh9);
        input7 = _mm512_min_epi32(input7, permNeigh10);
        input6 = _mm512_min_epi32(input6, permNeigh11);
        input5 = _mm512_min_epi32(input5, permNeigh12);
    }
    Finish8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Finish4VecBitFull(input9, input10, input11, input12);
}

inline void Sort12VecBitFull(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4,
            int* __restrict__ ptr5, int* __restrict__ ptr6, int* __restrict__ ptr7, int* __restrict__ ptr8,
                             int* __restrict__ ptr9, int* __restrict__ ptr10, int* __restrict__ ptr11, int* __restrict__ ptr12){
    __m512i input1 = _mm512_loadu_si512(ptr1);
    __m512i input2 = _mm512_loadu_si512(ptr2);
    __m512i input3 = _mm512_loadu_si512(ptr3);
    __m512i input4 = _mm512_loadu_si512(ptr4);
    __m512i input5 = _mm512_loadu_si512(ptr5);
    __m512i input6 = _mm512_loadu_si512(ptr6);
    __m512i input7 = _mm512_loadu_si512(ptr7);
    __m512i input8 = _mm512_loadu_si512(ptr8);
    __m512i input9 = _mm512_loadu_si512(ptr9);
    __m512i input10 = _mm512_loadu_si512(ptr10);
    __m512i input11 = _mm512_loadu_si512(ptr11);
    __m512i input12 = _mm512_loadu_si512(ptr12);
    Sort12VecBitFull(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10, input11, input12);
    _mm512_storeu_si512(ptr1, input1);
    _mm512_storeu_si512(ptr2, input2);
    _mm512_storeu_si512(ptr3, input3);
    _mm512_storeu_si512(ptr4, input4);
    _mm512_storeu_si512(ptr5, input5);
    _mm512_storeu_si512(ptr6, input6);
    _mm512_storeu_si512(ptr7, input7);
    _mm512_storeu_si512(ptr8, input8);
    _mm512_storeu_si512(ptr9, input9);
    _mm512_storeu_si512(ptr10, input10);
    _mm512_storeu_si512(ptr11, input11);
    _mm512_storeu_si512(ptr12, input12);
}

inline void Sort13VecBitFull(__m512i& input, __m512i& input2, __m512i& input3, __m512i& input4,
                            __m512i& input5, __m512i& input6, __m512i& input7, __m512i& input8,
                            __m512i& input9, __m512i& input10, __m512i& input11, __m512i& input12,
                            __m512i& input13 ){
    Sort8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Sort5VecBitFull(input9, input10, input11, input12, input13);
    {
        __m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                              8, 9, 10, 11, 12, 13, 14, 15);
        __m512i permNeigh9 = _mm512_permutexvar_epi32(idxNoNeigh, input9);
        __m512i permNeigh10 = _mm512_permutexvar_epi32(idxNoNeigh, input10);
        __m512i permNeigh11 = _mm512_permutexvar_epi32(idxNoNeigh, input11);
        __m512i permNeigh12 = _mm512_permutexvar_epi32(idxNoNeigh, input12);
        __m512i permNeigh13 = _mm512_permutexvar_epi32(idxNoNeigh, input13);

        input9 = _mm512_max_epi32(input8, permNeigh9);
        input10 = _mm512_max_epi32(input7, permNeigh10);
        input11 = _mm512_max_epi32(input6, permNeigh11);
        input12 = _mm512_max_epi32(input5, permNeigh12);
        input13 = _mm512_max_epi32(input4, permNeigh13);

        input8 = _mm512_min_epi32(input8, permNeigh9);
        input7 = _mm512_min_epi32(input7, permNeigh10);
        input6 = _mm512_min_epi32(input6, permNeigh11);
        input5 = _mm512_min_epi32(input5, permNeigh12);
        input4 = _mm512_min_epi32(input4, permNeigh13);
    }
    Finish8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Finish5VecBitFull(input9, input10, input11, input12, input13);
}

inline void Sort13VecBitFull(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4,
            int* __restrict__ ptr5, int* __restrict__ ptr6, int* __restrict__ ptr7, int* __restrict__ ptr8,
                             int* __restrict__ ptr9, int* __restrict__ ptr10, int* __restrict__ ptr11, int* __restrict__ ptr12,
                                         int* __restrict__ ptr13){
    __m512i input1 = _mm512_loadu_si512(ptr1);
    __m512i input2 = _mm512_loadu_si512(ptr2);
    __m512i input3 = _mm512_loadu_si512(ptr3);
    __m512i input4 = _mm512_loadu_si512(ptr4);
    __m512i input5 = _mm512_loadu_si512(ptr5);
    __m512i input6 = _mm512_loadu_si512(ptr6);
    __m512i input7 = _mm512_loadu_si512(ptr7);
    __m512i input8 = _mm512_loadu_si512(ptr8);
    __m512i input9 = _mm512_loadu_si512(ptr9);
    __m512i input10 = _mm512_loadu_si512(ptr10);
    __m512i input11 = _mm512_loadu_si512(ptr11);
    __m512i input12 = _mm512_loadu_si512(ptr12);
    __m512i input13 = _mm512_loadu_si512(ptr13);
    Sort13VecBitFull(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10, input11, input12, input13);
    _mm512_storeu_si512(ptr1, input1);
    _mm512_storeu_si512(ptr2, input2);
    _mm512_storeu_si512(ptr3, input3);
    _mm512_storeu_si512(ptr4, input4);
    _mm512_storeu_si512(ptr5, input5);
    _mm512_storeu_si512(ptr6, input6);
    _mm512_storeu_si512(ptr7, input7);
    _mm512_storeu_si512(ptr8, input8);
    _mm512_storeu_si512(ptr9, input9);
    _mm512_storeu_si512(ptr10, input10);
    _mm512_storeu_si512(ptr11, input11);
    _mm512_storeu_si512(ptr12, input12);
    _mm512_storeu_si512(ptr13, input13);
}


inline void Sort14VecBitFull(__m512i& input, __m512i& input2, __m512i& input3, __m512i& input4,
                            __m512i& input5, __m512i& input6, __m512i& input7, __m512i& input8,
                            __m512i& input9, __m512i& input10, __m512i& input11, __m512i& input12,
                            __m512i& input13, __m512i& input14 ){
    Sort8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Sort6VecBitFull(input9, input10, input11, input12, input13, input14);
    {
        __m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                              8, 9, 10, 11, 12, 13, 14, 15);
        __m512i permNeigh9 = _mm512_permutexvar_epi32(idxNoNeigh, input9);
        __m512i permNeigh10 = _mm512_permutexvar_epi32(idxNoNeigh, input10);
        __m512i permNeigh11 = _mm512_permutexvar_epi32(idxNoNeigh, input11);
        __m512i permNeigh12 = _mm512_permutexvar_epi32(idxNoNeigh, input12);
        __m512i permNeigh13 = _mm512_permutexvar_epi32(idxNoNeigh, input13);
        __m512i permNeigh14 = _mm512_permutexvar_epi32(idxNoNeigh, input14);

        input9 = _mm512_max_epi32(input8, permNeigh9);
        input10 = _mm512_max_epi32(input7, permNeigh10);
        input11 = _mm512_max_epi32(input6, permNeigh11);
        input12 = _mm512_max_epi32(input5, permNeigh12);
        input13 = _mm512_max_epi32(input4, permNeigh13);
        input14 = _mm512_max_epi32(input3, permNeigh14);

        input8 = _mm512_min_epi32(input8, permNeigh9);
        input7 = _mm512_min_epi32(input7, permNeigh10);
        input6 = _mm512_min_epi32(input6, permNeigh11);
        input5 = _mm512_min_epi32(input5, permNeigh12);
        input4 = _mm512_min_epi32(input4, permNeigh13);
        input3 = _mm512_min_epi32(input3, permNeigh14);
    }
    Finish8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Finish6VecBitFull(input9, input10, input11, input12, input13, input14);
}

inline void Sort14VecBitFull(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4,
            int* __restrict__ ptr5, int* __restrict__ ptr6, int* __restrict__ ptr7, int* __restrict__ ptr8,
                             int* __restrict__ ptr9, int* __restrict__ ptr10, int* __restrict__ ptr11, int* __restrict__ ptr12,
                                         int* __restrict__ ptr13, int* __restrict__ ptr14){
    __m512i input1 = _mm512_loadu_si512(ptr1);
    __m512i input2 = _mm512_loadu_si512(ptr2);
    __m512i input3 = _mm512_loadu_si512(ptr3);
    __m512i input4 = _mm512_loadu_si512(ptr4);
    __m512i input5 = _mm512_loadu_si512(ptr5);
    __m512i input6 = _mm512_loadu_si512(ptr6);
    __m512i input7 = _mm512_loadu_si512(ptr7);
    __m512i input8 = _mm512_loadu_si512(ptr8);
    __m512i input9 = _mm512_loadu_si512(ptr9);
    __m512i input10 = _mm512_loadu_si512(ptr10);
    __m512i input11 = _mm512_loadu_si512(ptr11);
    __m512i input12 = _mm512_loadu_si512(ptr12);
    __m512i input13 = _mm512_loadu_si512(ptr13);
    __m512i input14 = _mm512_loadu_si512(ptr14);
    Sort14VecBitFull(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10, input11, input12, input13, input14);
    _mm512_storeu_si512(ptr1, input1);
    _mm512_storeu_si512(ptr2, input2);
    _mm512_storeu_si512(ptr3, input3);
    _mm512_storeu_si512(ptr4, input4);
    _mm512_storeu_si512(ptr5, input5);
    _mm512_storeu_si512(ptr6, input6);
    _mm512_storeu_si512(ptr7, input7);
    _mm512_storeu_si512(ptr8, input8);
    _mm512_storeu_si512(ptr9, input9);
    _mm512_storeu_si512(ptr10, input10);
    _mm512_storeu_si512(ptr11, input11);
    _mm512_storeu_si512(ptr12, input12);
    _mm512_storeu_si512(ptr13, input13);
    _mm512_storeu_si512(ptr14, input14);
}


inline void Sort15VecBitFull(__m512i& input, __m512i& input2, __m512i& input3, __m512i& input4,
                            __m512i& input5, __m512i& input6, __m512i& input7, __m512i& input8,
                            __m512i& input9, __m512i& input10, __m512i& input11, __m512i& input12,
                            __m512i& input13, __m512i& input14, __m512i& input15 ){
    Sort8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Sort7VecBitFull(input9, input10, input11, input12, input13, input14, input15);
    {
        __m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                              8, 9, 10, 11, 12, 13, 14, 15);
        __m512i permNeigh9 = _mm512_permutexvar_epi32(idxNoNeigh, input9);
        __m512i permNeigh10 = _mm512_permutexvar_epi32(idxNoNeigh, input10);
        __m512i permNeigh11 = _mm512_permutexvar_epi32(idxNoNeigh, input11);
        __m512i permNeigh12 = _mm512_permutexvar_epi32(idxNoNeigh, input12);
        __m512i permNeigh13 = _mm512_permutexvar_epi32(idxNoNeigh, input13);
        __m512i permNeigh14 = _mm512_permutexvar_epi32(idxNoNeigh, input14);
        __m512i permNeigh15 = _mm512_permutexvar_epi32(idxNoNeigh, input15);

        input9 = _mm512_max_epi32(input8, permNeigh9);
        input10 = _mm512_max_epi32(input7, permNeigh10);
        input11 = _mm512_max_epi32(input6, permNeigh11);
        input12 = _mm512_max_epi32(input5, permNeigh12);
        input13 = _mm512_max_epi32(input4, permNeigh13);
        input14 = _mm512_max_epi32(input3, permNeigh14);
        input15 = _mm512_max_epi32(input2, permNeigh15);

        input8 = _mm512_min_epi32(input8, permNeigh9);
        input7 = _mm512_min_epi32(input7, permNeigh10);
        input6 = _mm512_min_epi32(input6, permNeigh11);
        input5 = _mm512_min_epi32(input5, permNeigh12);
        input4 = _mm512_min_epi32(input4, permNeigh13);
        input3 = _mm512_min_epi32(input3, permNeigh14);
        input2 = _mm512_min_epi32(input2, permNeigh15);
    }
    Finish8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Finish7VecBitFull(input9, input10, input11, input12, input13, input14, input15);
}

inline void Sort15VecBitFull(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4,
            int* __restrict__ ptr5, int* __restrict__ ptr6, int* __restrict__ ptr7, int* __restrict__ ptr8,
                             int* __restrict__ ptr9, int* __restrict__ ptr10, int* __restrict__ ptr11, int* __restrict__ ptr12,
                                         int* __restrict__ ptr13, int* __restrict__ ptr14, int* __restrict__ ptr15){
    __m512i input1 = _mm512_loadu_si512(ptr1);
    __m512i input2 = _mm512_loadu_si512(ptr2);
    __m512i input3 = _mm512_loadu_si512(ptr3);
    __m512i input4 = _mm512_loadu_si512(ptr4);
    __m512i input5 = _mm512_loadu_si512(ptr5);
    __m512i input6 = _mm512_loadu_si512(ptr6);
    __m512i input7 = _mm512_loadu_si512(ptr7);
    __m512i input8 = _mm512_loadu_si512(ptr8);
    __m512i input9 = _mm512_loadu_si512(ptr9);
    __m512i input10 = _mm512_loadu_si512(ptr10);
    __m512i input11 = _mm512_loadu_si512(ptr11);
    __m512i input12 = _mm512_loadu_si512(ptr12);
    __m512i input13 = _mm512_loadu_si512(ptr13);
    __m512i input14 = _mm512_loadu_si512(ptr14);
    __m512i input15 = _mm512_loadu_si512(ptr15);
    Sort15VecBitFull(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10, input11, input12, input13, input14, input15);
    _mm512_storeu_si512(ptr1, input1);
    _mm512_storeu_si512(ptr2, input2);
    _mm512_storeu_si512(ptr3, input3);
    _mm512_storeu_si512(ptr4, input4);
    _mm512_storeu_si512(ptr5, input5);
    _mm512_storeu_si512(ptr6, input6);
    _mm512_storeu_si512(ptr7, input7);
    _mm512_storeu_si512(ptr8, input8);
    _mm512_storeu_si512(ptr9, input9);
    _mm512_storeu_si512(ptr10, input10);
    _mm512_storeu_si512(ptr11, input11);
    _mm512_storeu_si512(ptr12, input12);
    _mm512_storeu_si512(ptr13, input13);
    _mm512_storeu_si512(ptr14, input14);
    _mm512_storeu_si512(ptr15, input15);
}


inline void Sort16VecBitFull(__m512i& input, __m512i& input2, __m512i& input3, __m512i& input4,
                            __m512i& input5, __m512i& input6, __m512i& input7, __m512i& input8,
                            __m512i& input9, __m512i& input10, __m512i& input11, __m512i& input12,
                            __m512i& input13, __m512i& input14, __m512i& input15, __m512i& input16 ){
    Sort8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Sort8VecBitFull(input9, input10, input11, input12, input13, input14, input15, input16);
    {
        __m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                              8, 9, 10, 11, 12, 13, 14, 15);
        __m512i permNeigh9 = _mm512_permutexvar_epi32(idxNoNeigh, input9);
        __m512i permNeigh10 = _mm512_permutexvar_epi32(idxNoNeigh, input10);
        __m512i permNeigh11 = _mm512_permutexvar_epi32(idxNoNeigh, input11);
        __m512i permNeigh12 = _mm512_permutexvar_epi32(idxNoNeigh, input12);
        __m512i permNeigh13 = _mm512_permutexvar_epi32(idxNoNeigh, input13);
        __m512i permNeigh14 = _mm512_permutexvar_epi32(idxNoNeigh, input14);
        __m512i permNeigh15 = _mm512_permutexvar_epi32(idxNoNeigh, input15);
        __m512i permNeigh16 = _mm512_permutexvar_epi32(idxNoNeigh, input16);

        input9 = _mm512_max_epi32(input8, permNeigh9);
        input10 = _mm512_max_epi32(input7, permNeigh10);
        input11 = _mm512_max_epi32(input6, permNeigh11);
        input12 = _mm512_max_epi32(input5, permNeigh12);
        input13 = _mm512_max_epi32(input4, permNeigh13);
        input14 = _mm512_max_epi32(input3, permNeigh14);
        input15 = _mm512_max_epi32(input2, permNeigh15);
        input16 = _mm512_max_epi32(input, permNeigh16);

        input8 = _mm512_min_epi32(input8, permNeigh9);
        input7 = _mm512_min_epi32(input7, permNeigh10);
        input6 = _mm512_min_epi32(input6, permNeigh11);
        input5 = _mm512_min_epi32(input5, permNeigh12);
        input4 = _mm512_min_epi32(input4, permNeigh13);
        input3 = _mm512_min_epi32(input3, permNeigh14);
        input2 = _mm512_min_epi32(input2, permNeigh15);
        input = _mm512_min_epi32(input, permNeigh16);
    }
    Finish8VecBitFull(input, input2, input3, input4, input5, input6, input7, input8);
    Finish8VecBitFull(input9, input10, input11, input12, input13, input14, input15, input16);
}

inline void Sort16VecBitFull(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4,
            int* __restrict__ ptr5, int* __restrict__ ptr6, int* __restrict__ ptr7, int* __restrict__ ptr8,
                             int* __restrict__ ptr9, int* __restrict__ ptr10, int* __restrict__ ptr11, int* __restrict__ ptr12,
                                         int* __restrict__ ptr13, int* __restrict__ ptr14, int* __restrict__ ptr15, int* __restrict__ ptr16){
    __m512i input1 = _mm512_loadu_si512(ptr1);
    __m512i input2 = _mm512_loadu_si512(ptr2);
    __m512i input3 = _mm512_loadu_si512(ptr3);
    __m512i input4 = _mm512_loadu_si512(ptr4);
    __m512i input5 = _mm512_loadu_si512(ptr5);
    __m512i input6 = _mm512_loadu_si512(ptr6);
    __m512i input7 = _mm512_loadu_si512(ptr7);
    __m512i input8 = _mm512_loadu_si512(ptr8);
    __m512i input9 = _mm512_loadu_si512(ptr9);
    __m512i input10 = _mm512_loadu_si512(ptr10);
    __m512i input11 = _mm512_loadu_si512(ptr11);
    __m512i input12 = _mm512_loadu_si512(ptr12);
    __m512i input13 = _mm512_loadu_si512(ptr13);
    __m512i input14 = _mm512_loadu_si512(ptr14);
    __m512i input15 = _mm512_loadu_si512(ptr15);
    __m512i input16 = _mm512_loadu_si512(ptr16);
    Sort16VecBitFull(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10, input11, input12, input13, input14, input15, input16);
    _mm512_storeu_si512(ptr1, input1);
    _mm512_storeu_si512(ptr2, input2);
    _mm512_storeu_si512(ptr3, input3);
    _mm512_storeu_si512(ptr4, input4);
    _mm512_storeu_si512(ptr5, input5);
    _mm512_storeu_si512(ptr6, input6);
    _mm512_storeu_si512(ptr7, input7);
    _mm512_storeu_si512(ptr8, input8);
    _mm512_storeu_si512(ptr9, input9);
    _mm512_storeu_si512(ptr10, input10);
    _mm512_storeu_si512(ptr11, input11);
    _mm512_storeu_si512(ptr12, input12);
    _mm512_storeu_si512(ptr13, input13);
    _mm512_storeu_si512(ptr14, input14);
    _mm512_storeu_si512(ptr15, input15);
    _mm512_storeu_si512(ptr16, input16);
}



inline void SortByVec(int* __restrict__ ptr, const size_t length){
    // length is limited to 4 times size of a vec
    const int nbValuesInVec = 16;
    const int nbVecs = (length+nbValuesInVec-1)/nbValuesInVec;
    const int rest = nbVecs*nbValuesInVec-length;
    const int lastVecSize = nbValuesInVec-rest;
    switch(nbVecs){
    case 1:
    {
        __m512i v1 = _mm512_or_si512(_mm512_maskz_loadu_epi32(0xFFFF>>rest, ptr),
                        _mm512_maskz_set1_epi32(0xFFFF<<lastVecSize, INT_MAX));
        v1 = SortVec(v1);
        _mm512_mask_compressstoreu_epi32(ptr, 0xFFFF>>rest, v1);
    }
        break;
    case 2:
    {
        __m512i v1 = _mm512_loadu_si512(ptr);
        __m512i v2 = _mm512_or_si512(_mm512_maskz_loadu_epi32(0xFFFF>>rest, ptr+16),
                                     _mm512_maskz_set1_epi32(0xFFFF<<lastVecSize, INT_MAX));
        Sort2Vec(v1,v2);
        _mm512_storeu_si512(ptr, v1);
        _mm512_mask_compressstoreu_epi32(ptr+16, 0xFFFF>>rest, v2);
    }
        break;
    case 3:
    {
        __m512i v1 = _mm512_loadu_si512(ptr);
        __m512i v2 = _mm512_loadu_si512(ptr+16);
        __m512i v3 = _mm512_or_si512(_mm512_maskz_loadu_epi32(0xFFFF>>rest, ptr+32),
                                     _mm512_maskz_set1_epi32(0xFFFF<<lastVecSize, INT_MAX));
        Sort3Vec(v1,v2,v3);
        _mm512_storeu_si512(ptr, v1);
        _mm512_storeu_si512(ptr+16, v2);
        _mm512_mask_compressstoreu_epi32(ptr+32, 0xFFFF>>rest, v3);
    }
        break;
    default:;
    //case 4:
    {
        __m512i v1 = _mm512_loadu_si512(ptr);
        __m512i v2 = _mm512_loadu_si512(ptr+16);
        __m512i v3 = _mm512_loadu_si512(ptr+32);
        __m512i v4 = _mm512_or_si512(_mm512_maskz_loadu_epi32(0xFFFF>>rest, ptr+48),
                                     _mm512_maskz_set1_epi32(0xFFFF<<lastVecSize, INT_MAX));
        Sort4Vec(v1,v2,v3,v4);
        _mm512_storeu_si512(ptr, v1);
        _mm512_storeu_si512(ptr+16, v2);
        _mm512_storeu_si512(ptr+32, v3);
        _mm512_mask_compressstoreu_epi32(ptr+48, 0xFFFF>>rest, v4);
    }
    }
}

inline void SortByVecBitFull(int* __restrict__ ptr, const size_t length){
    // length is limited to 4 times size of a vec
    const int nbValuesInVec = 16;
    const int nbVecs = (length+nbValuesInVec-1)/nbValuesInVec;
    const int rest = nbVecs*nbValuesInVec-length;
    const int lastVecSize = nbValuesInVec-rest;
    switch(nbVecs){
    case 1:
    {
        __m512i v1 = _mm512_or_si512(_mm512_maskz_loadu_epi32(0xFFFF>>rest, ptr),
                        _mm512_maskz_set1_epi32(0xFFFF<<lastVecSize, INT_MAX));
        v1 = SortVecBitFull(v1);
        _mm512_mask_compressstoreu_epi32(ptr, 0xFFFF>>rest, v1);
    }
        break;
    case 2:
    {
        __m512i v1 = _mm512_loadu_si512(ptr);
        __m512i v2 = _mm512_or_si512(_mm512_maskz_loadu_epi32(0xFFFF>>rest, ptr+16),
                                     _mm512_maskz_set1_epi32(0xFFFF<<lastVecSize, INT_MAX));
        Sort2VecBitFull(v1,v2);
        _mm512_storeu_si512(ptr, v1);
        _mm512_mask_compressstoreu_epi32(ptr+16, 0xFFFF>>rest, v2);
    }
        break;
    case 3:
    {
        __m512i v1 = _mm512_loadu_si512(ptr);
        __m512i v2 = _mm512_loadu_si512(ptr+16);
        __m512i v3 = _mm512_or_si512(_mm512_maskz_loadu_epi32(0xFFFF>>rest, ptr+32),
                                     _mm512_maskz_set1_epi32(0xFFFF<<lastVecSize, INT_MAX));
        Sort3VecBitFull(v1,v2,v3);
        _mm512_storeu_si512(ptr, v1);
        _mm512_storeu_si512(ptr+16, v2);
        _mm512_mask_compressstoreu_epi32(ptr+32, 0xFFFF>>rest, v3);
    }
        break;
    case 4:
    {
        __m512i v1 = _mm512_loadu_si512(ptr);
        __m512i v2 = _mm512_loadu_si512(ptr+16);
        __m512i v3 = _mm512_loadu_si512(ptr+32);
        __m512i v4 = _mm512_or_si512(_mm512_maskz_loadu_epi32(0xFFFF>>rest, ptr+48),
                                     _mm512_maskz_set1_epi32(0xFFFF<<lastVecSize, INT_MAX));
        Sort4VecBitFull(v1,v2,v3,v4);
        _mm512_storeu_si512(ptr, v1);
        _mm512_storeu_si512(ptr+16, v2);
        _mm512_storeu_si512(ptr+32, v3);
        _mm512_mask_compressstoreu_epi32(ptr+48, 0xFFFF>>rest, v4);
    }
        break;
    case 5:
    {
        __m512i v1 = _mm512_loadu_si512(ptr);
        __m512i v2 = _mm512_loadu_si512(ptr+16);
        __m512i v3 = _mm512_loadu_si512(ptr+32);
        __m512i v4 = _mm512_loadu_si512(ptr+48);
        __m512i v5 = _mm512_or_si512(_mm512_maskz_loadu_epi32(0xFFFF>>rest, ptr+64),
                                     _mm512_maskz_set1_epi32(0xFFFF<<lastVecSize, INT_MAX));
        Sort5VecBitFull(v1,v2,v3,v4,v5);
        _mm512_storeu_si512(ptr, v1);
        _mm512_storeu_si512(ptr+16, v2);
        _mm512_storeu_si512(ptr+32, v3);
        _mm512_storeu_si512(ptr+48, v4);
        _mm512_mask_compressstoreu_epi32(ptr+64, 0xFFFF>>rest, v5);
    }
        break;
    case 6:
    {
        __m512i v1 = _mm512_loadu_si512(ptr);
        __m512i v2 = _mm512_loadu_si512(ptr+16);
        __m512i v3 = _mm512_loadu_si512(ptr+32);
        __m512i v4 = _mm512_loadu_si512(ptr+48);
        __m512i v5 = _mm512_loadu_si512(ptr+64);
        __m512i v6 = _mm512_or_si512(_mm512_maskz_loadu_epi32(0xFFFF>>rest, ptr+80),
                                     _mm512_maskz_set1_epi32(0xFFFF<<lastVecSize, INT_MAX));
        Sort6VecBitFull(v1,v2,v3,v4,v5,v6);
        _mm512_storeu_si512(ptr, v1);
        _mm512_storeu_si512(ptr+16, v2);
        _mm512_storeu_si512(ptr+32, v3);
        _mm512_storeu_si512(ptr+48, v4);
        _mm512_storeu_si512(ptr+64, v5);
        _mm512_mask_compressstoreu_epi32(ptr+80, 0xFFFF>>rest, v6);
    }
        break;
    case 7:
    {
        __m512i v1 = _mm512_loadu_si512(ptr);
        __m512i v2 = _mm512_loadu_si512(ptr+16);
        __m512i v3 = _mm512_loadu_si512(ptr+32);
        __m512i v4 = _mm512_loadu_si512(ptr+48);
        __m512i v5 = _mm512_loadu_si512(ptr+64);
        __m512i v6 = _mm512_loadu_si512(ptr+80);
        __m512i v7 = _mm512_or_si512(_mm512_maskz_loadu_epi32(0xFFFF>>rest, ptr+96),
                                     _mm512_maskz_set1_epi32(0xFFFF<<lastVecSize, INT_MAX));
        Sort7VecBitFull(v1,v2,v3,v4,v5,v6,v7);
        _mm512_storeu_si512(ptr, v1);
        _mm512_storeu_si512(ptr+16, v2);
        _mm512_storeu_si512(ptr+32, v3);
        _mm512_storeu_si512(ptr+48, v4);
        _mm512_storeu_si512(ptr+64, v5);
        _mm512_storeu_si512(ptr+80, v6);
        _mm512_mask_compressstoreu_epi32(ptr+96, 0xFFFF>>rest, v7);
    }
        break;
    case 8:
    {
        __m512i v1 = _mm512_loadu_si512(ptr);
        __m512i v2 = _mm512_loadu_si512(ptr+16);
        __m512i v3 = _mm512_loadu_si512(ptr+32);
        __m512i v4 = _mm512_loadu_si512(ptr+48);
        __m512i v5 = _mm512_loadu_si512(ptr+64);
        __m512i v6 = _mm512_loadu_si512(ptr+80);
        __m512i v7 = _mm512_loadu_si512(ptr+96);
        __m512i v8 = _mm512_or_si512(_mm512_maskz_loadu_epi32(0xFFFF>>rest, ptr+112),
                                     _mm512_maskz_set1_epi32(0xFFFF<<lastVecSize, INT_MAX));
        Sort8VecBitFull(v1,v2,v3,v4,v5,v6,v7,v8);
        _mm512_storeu_si512(ptr, v1);
        _mm512_storeu_si512(ptr+16, v2);
        _mm512_storeu_si512(ptr+32, v3);
        _mm512_storeu_si512(ptr+48, v4);
        _mm512_storeu_si512(ptr+64, v5);
        _mm512_storeu_si512(ptr+80, v6);
        _mm512_storeu_si512(ptr+96, v7);
        _mm512_mask_compressstoreu_epi32(ptr+112, 0xFFFF>>rest, v8);
    }
        break;
    case 9:
    {
        __m512i v1 = _mm512_loadu_si512(ptr);
        __m512i v2 = _mm512_loadu_si512(ptr+16);
        __m512i v3 = _mm512_loadu_si512(ptr+32);
        __m512i v4 = _mm512_loadu_si512(ptr+48);
        __m512i v5 = _mm512_loadu_si512(ptr+64);
        __m512i v6 = _mm512_loadu_si512(ptr+80);
        __m512i v7 = _mm512_loadu_si512(ptr+96);
        __m512i v8 = _mm512_loadu_si512(ptr+112);
        __m512i v9 = _mm512_or_si512(_mm512_maskz_loadu_epi32(0xFFFF>>rest, ptr+128),
                                     _mm512_maskz_set1_epi32(0xFFFF<<lastVecSize, INT_MAX));
        Sort9VecBitFull(v1,v2,v3,v4,v5,v6,v7,v8,v9);
        _mm512_storeu_si512(ptr, v1);
        _mm512_storeu_si512(ptr+16, v2);
        _mm512_storeu_si512(ptr+32, v3);
        _mm512_storeu_si512(ptr+48, v4);
        _mm512_storeu_si512(ptr+64, v5);
        _mm512_storeu_si512(ptr+80, v6);
        _mm512_storeu_si512(ptr+96, v7);
        _mm512_storeu_si512(ptr+112, v8);
        _mm512_mask_compressstoreu_epi32(ptr+128, 0xFFFF>>rest, v9);
    }
        break;
    case 10:
    {
        __m512i v1 = _mm512_loadu_si512(ptr);
        __m512i v2 = _mm512_loadu_si512(ptr+16);
        __m512i v3 = _mm512_loadu_si512(ptr+32);
        __m512i v4 = _mm512_loadu_si512(ptr+48);
        __m512i v5 = _mm512_loadu_si512(ptr+64);
        __m512i v6 = _mm512_loadu_si512(ptr+80);
        __m512i v7 = _mm512_loadu_si512(ptr+96);
        __m512i v8 = _mm512_loadu_si512(ptr+112);
        __m512i v9 = _mm512_loadu_si512(ptr+128);
        __m512i v10 = _mm512_or_si512(_mm512_maskz_loadu_epi32(0xFFFF>>rest, ptr+144),
                                     _mm512_maskz_set1_epi32(0xFFFF<<lastVecSize, INT_MAX));
        Sort10VecBitFull(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10);
        _mm512_storeu_si512(ptr, v1);
        _mm512_storeu_si512(ptr+16, v2);
        _mm512_storeu_si512(ptr+32, v3);
        _mm512_storeu_si512(ptr+48, v4);
        _mm512_storeu_si512(ptr+64, v5);
        _mm512_storeu_si512(ptr+80, v6);
        _mm512_storeu_si512(ptr+96, v7);
        _mm512_storeu_si512(ptr+112, v8);
        _mm512_storeu_si512(ptr+128, v9);
        _mm512_mask_compressstoreu_epi32(ptr+144, 0xFFFF>>rest, v10);
    }
        break;
    case 11:
    {
        __m512i v1 = _mm512_loadu_si512(ptr);
        __m512i v2 = _mm512_loadu_si512(ptr+16);
        __m512i v3 = _mm512_loadu_si512(ptr+32);
        __m512i v4 = _mm512_loadu_si512(ptr+48);
        __m512i v5 = _mm512_loadu_si512(ptr+64);
        __m512i v6 = _mm512_loadu_si512(ptr+80);
        __m512i v7 = _mm512_loadu_si512(ptr+96);
        __m512i v8 = _mm512_loadu_si512(ptr+112);
        __m512i v9 = _mm512_loadu_si512(ptr+128);
        __m512i v10 = _mm512_loadu_si512(ptr+144);
        __m512i v11 = _mm512_or_si512(_mm512_maskz_loadu_epi32(0xFFFF>>rest, ptr+160),
                                     _mm512_maskz_set1_epi32(0xFFFF<<lastVecSize, INT_MAX));
        Sort11VecBitFull(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11);
        _mm512_storeu_si512(ptr, v1);
        _mm512_storeu_si512(ptr+16, v2);
        _mm512_storeu_si512(ptr+32, v3);
        _mm512_storeu_si512(ptr+48, v4);
        _mm512_storeu_si512(ptr+64, v5);
        _mm512_storeu_si512(ptr+80, v6);
        _mm512_storeu_si512(ptr+96, v7);
        _mm512_storeu_si512(ptr+112, v8);
        _mm512_storeu_si512(ptr+128, v9);
        _mm512_storeu_si512(ptr+144, v10);
        _mm512_mask_compressstoreu_epi32(ptr+160, 0xFFFF>>rest, v11);
    }
        break;
    case 12:
    {
        __m512i v1 = _mm512_loadu_si512(ptr);
        __m512i v2 = _mm512_loadu_si512(ptr+16);
        __m512i v3 = _mm512_loadu_si512(ptr+32);
        __m512i v4 = _mm512_loadu_si512(ptr+48);
        __m512i v5 = _mm512_loadu_si512(ptr+64);
        __m512i v6 = _mm512_loadu_si512(ptr+80);
        __m512i v7 = _mm512_loadu_si512(ptr+96);
        __m512i v8 = _mm512_loadu_si512(ptr+112);
        __m512i v9 = _mm512_loadu_si512(ptr+128);
        __m512i v10 = _mm512_loadu_si512(ptr+144);
        __m512i v11 = _mm512_loadu_si512(ptr+160);
        __m512i v12 = _mm512_or_si512(_mm512_maskz_loadu_epi32(0xFFFF>>rest, ptr+176),
                                     _mm512_maskz_set1_epi32(0xFFFF<<lastVecSize, INT_MAX));
        Sort12VecBitFull(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12);
        _mm512_storeu_si512(ptr, v1);
        _mm512_storeu_si512(ptr+16, v2);
        _mm512_storeu_si512(ptr+32, v3);
        _mm512_storeu_si512(ptr+48, v4);
        _mm512_storeu_si512(ptr+64, v5);
        _mm512_storeu_si512(ptr+80, v6);
        _mm512_storeu_si512(ptr+96, v7);
        _mm512_storeu_si512(ptr+112, v8);
        _mm512_storeu_si512(ptr+128, v9);
        _mm512_storeu_si512(ptr+144, v10);
        _mm512_storeu_si512(ptr+160, v11);
        _mm512_mask_compressstoreu_epi32(ptr+176, 0xFFFF>>rest, v12);
    }
        break;
    case 13:
    {
        __m512i v1 = _mm512_loadu_si512(ptr);
        __m512i v2 = _mm512_loadu_si512(ptr+16);
        __m512i v3 = _mm512_loadu_si512(ptr+32);
        __m512i v4 = _mm512_loadu_si512(ptr+48);
        __m512i v5 = _mm512_loadu_si512(ptr+64);
        __m512i v6 = _mm512_loadu_si512(ptr+80);
        __m512i v7 = _mm512_loadu_si512(ptr+96);
        __m512i v8 = _mm512_loadu_si512(ptr+112);
        __m512i v9 = _mm512_loadu_si512(ptr+128);
        __m512i v10 = _mm512_loadu_si512(ptr+144);
        __m512i v11 = _mm512_loadu_si512(ptr+160);
        __m512i v12 = _mm512_loadu_si512(ptr+176);
        __m512i v13 = _mm512_or_si512(_mm512_maskz_loadu_epi32(0xFFFF>>rest, ptr+192),
                                     _mm512_maskz_set1_epi32(0xFFFF<<lastVecSize, INT_MAX));
        Sort13VecBitFull(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13);
        _mm512_storeu_si512(ptr, v1);
        _mm512_storeu_si512(ptr+16, v2);
        _mm512_storeu_si512(ptr+32, v3);
        _mm512_storeu_si512(ptr+48, v4);
        _mm512_storeu_si512(ptr+64, v5);
        _mm512_storeu_si512(ptr+80, v6);
        _mm512_storeu_si512(ptr+96, v7);
        _mm512_storeu_si512(ptr+112, v8);
        _mm512_storeu_si512(ptr+128, v9);
        _mm512_storeu_si512(ptr+144, v10);
        _mm512_storeu_si512(ptr+160, v11);
        _mm512_storeu_si512(ptr+176, v12);
        _mm512_mask_compressstoreu_epi32(ptr+192, 0xFFFF>>rest, v13);
    }
        break;
    case 14:
    {
        __m512i v1 = _mm512_loadu_si512(ptr);
        __m512i v2 = _mm512_loadu_si512(ptr+16);
        __m512i v3 = _mm512_loadu_si512(ptr+32);
        __m512i v4 = _mm512_loadu_si512(ptr+48);
        __m512i v5 = _mm512_loadu_si512(ptr+64);
        __m512i v6 = _mm512_loadu_si512(ptr+80);
        __m512i v7 = _mm512_loadu_si512(ptr+96);
        __m512i v8 = _mm512_loadu_si512(ptr+112);
        __m512i v9 = _mm512_loadu_si512(ptr+128);
        __m512i v10 = _mm512_loadu_si512(ptr+144);
        __m512i v11 = _mm512_loadu_si512(ptr+160);
        __m512i v12 = _mm512_loadu_si512(ptr+176);
        __m512i v13 = _mm512_loadu_si512(ptr+192);
        __m512i v14 = _mm512_or_si512(_mm512_maskz_loadu_epi32(0xFFFF>>rest, ptr+208),
                                     _mm512_maskz_set1_epi32(0xFFFF<<lastVecSize, INT_MAX));
        Sort14VecBitFull(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14);
        _mm512_storeu_si512(ptr, v1);
        _mm512_storeu_si512(ptr+16, v2);
        _mm512_storeu_si512(ptr+32, v3);
        _mm512_storeu_si512(ptr+48, v4);
        _mm512_storeu_si512(ptr+64, v5);
        _mm512_storeu_si512(ptr+80, v6);
        _mm512_storeu_si512(ptr+96, v7);
        _mm512_storeu_si512(ptr+112, v8);
        _mm512_storeu_si512(ptr+128, v9);
        _mm512_storeu_si512(ptr+144, v10);
        _mm512_storeu_si512(ptr+160, v11);
        _mm512_storeu_si512(ptr+176, v12);
        _mm512_storeu_si512(ptr+192, v13);
        _mm512_mask_compressstoreu_epi32(ptr+208, 0xFFFF>>rest, v14);
    }
        break;
    case 15:
    {
        __m512i v1 = _mm512_loadu_si512(ptr);
        __m512i v2 = _mm512_loadu_si512(ptr+16);
        __m512i v3 = _mm512_loadu_si512(ptr+32);
        __m512i v4 = _mm512_loadu_si512(ptr+48);
        __m512i v5 = _mm512_loadu_si512(ptr+64);
        __m512i v6 = _mm512_loadu_si512(ptr+80);
        __m512i v7 = _mm512_loadu_si512(ptr+96);
        __m512i v8 = _mm512_loadu_si512(ptr+112);
        __m512i v9 = _mm512_loadu_si512(ptr+128);
        __m512i v10 = _mm512_loadu_si512(ptr+144);
        __m512i v11 = _mm512_loadu_si512(ptr+160);
        __m512i v12 = _mm512_loadu_si512(ptr+176);
        __m512i v13 = _mm512_loadu_si512(ptr+192);
        __m512i v14 = _mm512_loadu_si512(ptr+208);
        __m512i v15 = _mm512_or_si512(_mm512_maskz_loadu_epi32(0xFFFF>>rest, ptr+224),
                                     _mm512_maskz_set1_epi32(0xFFFF<<lastVecSize, INT_MAX));
        Sort15VecBitFull(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15);
        _mm512_storeu_si512(ptr, v1);
        _mm512_storeu_si512(ptr+16, v2);
        _mm512_storeu_si512(ptr+32, v3);
        _mm512_storeu_si512(ptr+48, v4);
        _mm512_storeu_si512(ptr+64, v5);
        _mm512_storeu_si512(ptr+80, v6);
        _mm512_storeu_si512(ptr+96, v7);
        _mm512_storeu_si512(ptr+112, v8);
        _mm512_storeu_si512(ptr+128, v9);
        _mm512_storeu_si512(ptr+144, v10);
        _mm512_storeu_si512(ptr+160, v11);
        _mm512_storeu_si512(ptr+176, v12);
        _mm512_storeu_si512(ptr+192, v13);
        _mm512_storeu_si512(ptr+208, v14);
        _mm512_mask_compressstoreu_epi32(ptr+224, 0xFFFF>>rest, v15);
    }
        break;
    //case 16:
    default:
    {
        __m512i v1 = _mm512_loadu_si512(ptr);
        __m512i v2 = _mm512_loadu_si512(ptr+16);
        __m512i v3 = _mm512_loadu_si512(ptr+32);
        __m512i v4 = _mm512_loadu_si512(ptr+48);
        __m512i v5 = _mm512_loadu_si512(ptr+64);
        __m512i v6 = _mm512_loadu_si512(ptr+80);
        __m512i v7 = _mm512_loadu_si512(ptr+96);
        __m512i v8 = _mm512_loadu_si512(ptr+112);
        __m512i v9 = _mm512_loadu_si512(ptr+128);
        __m512i v10 = _mm512_loadu_si512(ptr+144);
        __m512i v11 = _mm512_loadu_si512(ptr+160);
        __m512i v12 = _mm512_loadu_si512(ptr+176);
        __m512i v13 = _mm512_loadu_si512(ptr+192);
        __m512i v14 = _mm512_loadu_si512(ptr+208);
        __m512i v15 = _mm512_loadu_si512(ptr+224);
        __m512i v16 = _mm512_or_si512(_mm512_maskz_loadu_epi32(0xFFFF>>rest, ptr+240),
                                     _mm512_maskz_set1_epi32(0xFFFF<<lastVecSize, INT_MAX));
        Sort16VecBitFull(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16);
        _mm512_storeu_si512(ptr, v1);
        _mm512_storeu_si512(ptr+16, v2);
        _mm512_storeu_si512(ptr+32, v3);
        _mm512_storeu_si512(ptr+48, v4);
        _mm512_storeu_si512(ptr+64, v5);
        _mm512_storeu_si512(ptr+80, v6);
        _mm512_storeu_si512(ptr+96, v7);
        _mm512_storeu_si512(ptr+112, v8);
        _mm512_storeu_si512(ptr+128, v9);
        _mm512_storeu_si512(ptr+144, v10);
        _mm512_storeu_si512(ptr+160, v11);
        _mm512_storeu_si512(ptr+176, v12);
        _mm512_storeu_si512(ptr+192, v13);
        _mm512_storeu_si512(ptr+208, v14);
        _mm512_storeu_si512(ptr+224, v15);
        _mm512_mask_compressstoreu_epi32(ptr+240, 0xFFFF>>rest, v16);
    }
    }
}

inline void Sort2VecWithTest(__m512i& input1, __m512i& input2 ){
    __m512i idxNoNeigh = _mm512_set_epi32(15, 13, 14, 11, 12, 9, 10, 7, 8, 5, 6, 3, 4, 1, 2, 0);
    __m512i idxNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    __m512i idxAll0 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    __m512i idxAll7 = _mm512_set_epi32(15, 15, 15, 15,15, 15, 15, 15,15, 15, 15, 15,15, 15, 15, 15);

    for( int idx = 0 ; idx < 16 ; ++idx){
        __m512i permNeighOdd1 = _mm512_permutexvar_epi32(idxNeigh, input1);
        __m512i permNeighOdd2 = _mm512_permutexvar_epi32(idxNeigh, input2);

        __mmask16 compMaskOdd1 = _mm512_cmp_epi32_mask(permNeighOdd1, input1, _MM_CMPINT_LT);
        __mmask16 compMaskOdd2 = _mm512_cmp_epi32_mask(permNeighOdd2, input2, _MM_CMPINT_LT);

        input1 = _mm512_mask_mov_epi32(input1, (compMaskOdd1 & 0x5555) | ((compMaskOdd1 & 0x5555)<<1), permNeighOdd1);
        input2 = _mm512_mask_mov_epi32(input2, (compMaskOdd2 & 0x5555) | ((compMaskOdd2 & 0x5555)<<1), permNeighOdd2);

        __m512i permNeighEven1 = _mm512_permutexvar_epi32(idxNoNeigh, input1);
        __m512i permNeighEven2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);

        __mmask16 compMaskEven1 = _mm512_cmp_epi32_mask(permNeighEven1, input1, _MM_CMPINT_LT);
        __mmask16 compMaskEven2 = _mm512_cmp_epi32_mask(permNeighEven2, input2, _MM_CMPINT_LT);

        input1 = _mm512_mask_mov_epi32(input1, (compMaskEven1 & 0x2AAA) | ((compMaskEven1 & 0x2AAA)<<1), permNeighEven1);
        input2 = _mm512_mask_mov_epi32(input2, (compMaskEven2 & 0x2AAA) | ((compMaskEven2 & 0x2AAA)<<1), permNeighEven2);

        // Exchange border if needed
        __m512i input1last = _mm512_permutexvar_epi32(idxAll7, input1);
        __m512i input2first = _mm512_permutexvar_epi32(idxAll0, input2);

        __mmask16 compDoExchange = _mm512_cmp_epi32_mask(input1last, input2first, _MM_CMPINT_GT);

        input1 = _mm512_mask_mov_epi32(input1, compDoExchange & 0x8000, input2first);
        input2 = _mm512_mask_mov_epi32(input2, compDoExchange & 1, input1last);

        if(!(compMaskOdd1 || compMaskOdd2 || compMaskEven1 || compMaskEven2 || compDoExchange)){
            break;
        }
    }
}

inline void Sort2VecWithTest(int* __restrict__ ptr1, int* __restrict__ ptr2 ){
    __m512i input1 = _mm512_loadu_si512(ptr1);
    __m512i input2 = _mm512_loadu_si512(ptr2);
    Sort2VecWithTest(input1, input2);
    _mm512_storeu_si512(ptr1, input1);
    _mm512_storeu_si512(ptr2, input2);
}


inline void Merge2Vec(__m512i& input1, __m512i& input2 ){
    __m512i reverseIdx = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i input2rev = _mm512_permutexvar_epi32(reverseIdx, input2);

    __mmask16 compDoExchange = _mm512_cmp_epi32_mask(input1, input2rev, _MM_CMPINT_GT);

    if(compDoExchange == 0){
        return;
    }

    __m512i newInput1 = _mm512_mask_permutexvar_epi32(input1, compDoExchange, reverseIdx, input2);
    __m512i newInput2 = _mm512_mask_permutexvar_epi32(input1, ~compDoExchange, reverseIdx, input2);

    input1 = SortVecWithTest(newInput1);
    input2 = SortVecWithTest(_mm512_permutexvar_epi32(reverseIdx, newInput2));
}

inline void Merge2Vec(int* __restrict__ ptr1, int* __restrict__ ptr2 ){
    __m512i input1 = _mm512_loadu_si512(ptr1);
    __m512i input2 = _mm512_loadu_si512(ptr2);
    Merge2Vec(input1, input2);
    _mm512_storeu_si512(ptr1, input1);
    _mm512_storeu_si512(ptr2, input2);
}



////////////////////////////////////////////////////////////
/// Sort Functions
////////////////////////////////////////////////////////////

template <class SortType, class IndexType = size_t>
class BitonicSort {
    inline static void mergeUp(SortType *arr, IndexType n) {
        for(IndexType step = n/2 ; step > 0 ; step /= 2) {
            for (IndexType i=0; i < n; i += step*2 ) {
                SortType* __restrict__ leftPtr = &arr[i];
                SortType* __restrict__ rightPtr = &arr[i+step];
                for (IndexType k=0 ; k < step ; k++) {
                    if ((*leftPtr) > (*rightPtr)) {
                        std::swap((*leftPtr), (*rightPtr));
                    }
                    ++leftPtr;
                    ++rightPtr;
                }
            }
        }
    }

    inline static void mergeDown(SortType *arr, IndexType n) {
        for(IndexType step = n/2 ; step > 0 ; step /= 2) {
            for (IndexType i=0; i < n; i += step*2 ) {
                SortType* __restrict__ leftPtr = &arr[i];
                SortType* __restrict__ rightPtr = &arr[i+step];
                for (IndexType k=0 ; k < step ; k++) {
                    if ((*leftPtr) < (*rightPtr)) {
                        std::swap((*leftPtr), (*rightPtr));
                    }
                    ++leftPtr;
                    ++rightPtr;
                }
            }
        }
    }


    inline static void mergeUpOmp(SortType *arr, IndexType n) {
        for(IndexType step = n/2 ; step > 0 ; step /= 2) {
#pragma omp parallel
            {
                const IndexType steploop = ((n/step*2)+omp_get_num_threads() - 1)/omp_get_num_threads();
                const IndexType first = steploop*step*2*omp_get_thread_num();
                const IndexType limit = std::min(steploop*step*2*(omp_get_thread_num()+1), n);

                for (IndexType i= first; i < limit; i += step*2 ) {
                    SortType* __restrict__ leftPtr = &arr[i];
                    SortType* __restrict__ rightPtr = &arr[i+step];
                    for (IndexType k=0 ; k < step ; k++) {
                        if ((*leftPtr) > (*rightPtr)) {
                            std::swap((*leftPtr), (*rightPtr));
                        }
                        ++leftPtr;
                        ++rightPtr;
                    }
                }
            }
        }
    }

    inline static void mergeDownOmp(SortType *arr, IndexType n) {
        for(IndexType step = n/2 ; step > 0 ; step /= 2) {
#pragma omp parallel
            {
                const IndexType steploop = ((n/step*2)+omp_get_num_threads() - 1)/omp_get_num_threads();
                const IndexType first = steploop*step*2*omp_get_thread_num();
                const IndexType limit = std::min(steploop*step*2*(omp_get_thread_num()+1), n);

                for (IndexType i= first; i < limit; i += step*2 ) {
                    SortType* __restrict__ leftPtr = &arr[i];
                    SortType* __restrict__ rightPtr = &arr[i+step];
                    for (IndexType k=0 ; k < step ; k++) {
                        if ((*leftPtr) < (*rightPtr)) {
                            std::swap((*leftPtr), (*rightPtr));
                        }
                        ++leftPtr;
                        ++rightPtr;
                    }
                }
            }
        }
    }

public:
    static inline void  BsSequential(SortType val[], const IndexType n){
        if(((n-1)&n) != 0){
            throw std::invalid_argument("Size of array must be a power of 2");
        }

        for (IndexType s=2; s <= n; s*=2) {
            for (IndexType i=0; i < n; i+= s) {
                if((i & s) == 0){
                    mergeUp((val+i), s);
                }
                else {
                    mergeDown((val+i), s);
                }
            }
        }
    }


    static inline void BsOmpV1(SortType val[], const IndexType n){
        if(((n-1)&n) != 0){
            throw std::invalid_argument("Size of array must be a power of 2");
        }

        for (IndexType s=2; s <= n; s*=2) {
#pragma omp parallel
            {
                const IndexType step = ((n/s)+omp_get_num_threads() - 1)/omp_get_num_threads();
                const IndexType first = step*omp_get_thread_num()*s;
                const IndexType limit = std::min(step*(omp_get_thread_num()+1)*s, n);
                for (IndexType i = first ; i < limit; i+= s) {
                    if((i & s) == 0){
                        mergeUp((val+i), s);
                    }
                    else {
                        mergeDown((val+i), s);
                    }
                }
            }
        }
    }

    static inline void BsOmpV2(SortType val[], const IndexType n){
        if(((n-1)&n) != 0){
            throw std::invalid_argument("Size of array must be a power of 2");
        }

        const size_t num_threads = size_t(omp_get_max_threads());

        IndexType s = 2;

        for (; s <= n && n/s >= num_threads; s*=2) {
#pragma omp parallel
            {
                const IndexType step = ((n/s)+omp_get_num_threads() - 1)/omp_get_num_threads();
                const IndexType first = step*omp_get_thread_num()*s;
                const IndexType limit = std::min(step*(omp_get_thread_num()+1)*s, n);
                for (IndexType i = first ; i < limit; i+= s) {
                    if((i & s) == 0){
                        mergeUp((val+i), s);
                    }
                    else {
                        mergeDown((val+i), s);
                    }
                }
            }
        }

        for (; s <= n ; s*=2) {
            for (IndexType i=0; i < n; i+= s) {
                if((i & s) == 0){
                    mergeUpOmp((val+i), s);
                }
                else {
                    mergeDownOmp((val+i), s);
                }
            }
        }
    }
};


template <class SortType, class IndexType = size_t>
class BitonicSortV2{
    inline static void mergeUpV2(SortType *arr, IndexType n) {
        {
            const IndexType step = n/2;
            SortType* __restrict__ leftPtr = &arr[0];
            SortType* __restrict__ rightPtr = &arr[0+n-1];
            for (IndexType k=0 ; k < step ; k++) {
                if ((*leftPtr) > (*rightPtr)) {
                    std::swap((*leftPtr), (*rightPtr));
                }
                ++leftPtr;
                --rightPtr;
            }
        }
        for(IndexType step = n/2/2 ; step > 0 ; step /= 2) {
            for (IndexType i=0; i < n; i += step*2 ) {
                SortType* __restrict__ leftPtr = &arr[i];
                SortType* __restrict__ rightPtr = &arr[i+step];
                for (IndexType k=0 ; k < step ; k++) {
                    if ((*leftPtr) > (*rightPtr)) {
                        std::swap((*leftPtr), (*rightPtr));
                    }
                    ++leftPtr;
                    ++rightPtr;
                }
            }
        }
    }

    inline static void mergeUpV2Limite(SortType *arr, IndexType n,
                                       IndexType limite) {
        if(limite <= 1){
            return;
        }

        if(limite > n/2){
            SortType* __restrict__ leftPtr = &arr[n - limite];
            SortType* __restrict__ rightPtr = &arr[limite-1];
            while(leftPtr < rightPtr) {
                if ((*leftPtr) > (*rightPtr)) {
                    std::swap((*leftPtr), (*rightPtr));
                }
                ++leftPtr;
                --rightPtr;
            }
        }
        for(IndexType step = n/2/2 ; step > 0 ; step /= 2) {
            IndexType i=0;
            for (; i+step*2 < limite; i += step*2 ) {
                SortType* __restrict__ leftPtr = &arr[i];
                SortType* __restrict__ rightPtr = &arr[i+step];
                for (IndexType k=0 ; k < step ; k++) {
                    if ((*leftPtr) > (*rightPtr)) {
                        std::swap((*leftPtr), (*rightPtr));
                    }
                    ++leftPtr;
                    ++rightPtr;
                }
            }
            {
                SortType* __restrict__ leftPtr = &arr[i];
                SortType* __restrict__ rightPtr = &arr[i+step];
                for (IndexType k=0 ; k+i+step < limite ; k++) {
                    if ((*leftPtr) > (*rightPtr)) {
                        std::swap((*leftPtr), (*rightPtr));
                    }
                    ++leftPtr;
                    ++rightPtr;
                }
            }
        }
    }


public:
    static inline void  BsSequential(SortType val[], const IndexType n){
        if(((n-1)&n) != 0){
            throw std::invalid_argument("Size of array must be a power of 2");
        }

        for (IndexType s=2; s <= n; s*=2) {
            for (IndexType i=0; i < n; i+= s) {
                mergeUpV2((val+i), s);
            }
        }
    }

    static inline void  BsSequentialV2(SortType val[], const IndexType size){
        IndexType n = 1;
        while( n <= size ) n *= 2;

        for (IndexType s=2; s <= n; s*=2) {
            IndexType i=0;

            for (; i+s < size; i+= s) {
                mergeUpV2((val+i), s);
            }

            {
                mergeUpV2Limite((val+i), s, size-i);
            }
        }
    }
};

template <class SortType, class IndexType = size_t>
class BitonicSortAVX512{
    static const IndexType NbValPerVec = 64/sizeof(SortType);

    inline static void mergeUpV2(SortType *arr, IndexType n) {
        {
            const IndexType step = n/2;
            SortType* __restrict__ leftPtr = &arr[0];
            SortType* __restrict__ rightPtr = &arr[(n-1)*NbValPerVec];
            for (IndexType k=0 ; k < step ; k++) {
                Sort2Vec(leftPtr, rightPtr);
                leftPtr += NbValPerVec;
                rightPtr -= NbValPerVec;
            }
        }
        for(IndexType step = n/2/2 ; step > 0 ; step /= 2) {
            for (IndexType i=0; i < n; i += step*2 ) {
                SortType* __restrict__ leftPtr = &arr[i*NbValPerVec];
                SortType* __restrict__ rightPtr = &arr[(i+step)*NbValPerVec];
                for (IndexType k=0 ; k < step ; k++) {
                    Sort2Vec(leftPtr, rightPtr);
                    leftPtr += NbValPerVec;
                    rightPtr += NbValPerVec;
                }
            }
        }
    }

    inline static void mergeUpV2Limite(SortType *arr, IndexType n,
                                       IndexType limite) {
        if(limite <= 1){
            return;
        }

        if(limite > n/2){
            SortType* __restrict__ leftPtr = &arr[(n - limite)*NbValPerVec];
            SortType* __restrict__ rightPtr = &arr[(limite-1)*NbValPerVec];
            while(leftPtr < rightPtr) {
                Sort2Vec(leftPtr, rightPtr);
                leftPtr += NbValPerVec;
                rightPtr -= NbValPerVec;
            }
        }
        for(IndexType step = n/2/2 ; step > 0 ; step /= 2) {
            IndexType i=0;
            for (; i+step*2 < limite; i += step*2 ) {
                SortType* __restrict__ leftPtr = &arr[i*NbValPerVec];
                SortType* __restrict__ rightPtr = &arr[(i+step)*NbValPerVec];
                for (IndexType k=0 ; k < step ; k++) {
                    Sort2Vec(leftPtr, rightPtr);
                    leftPtr += NbValPerVec;
                    rightPtr += NbValPerVec;
                }
            }
            {
                SortType* __restrict__ leftPtr = &arr[i*NbValPerVec];
                SortType* __restrict__ rightPtr = &arr[(i+step)*NbValPerVec];
                for (IndexType k=0 ; k+i+step < limite ; k++) {
                    Sort2Vec(leftPtr, rightPtr);
                    leftPtr += NbValPerVec;
                    rightPtr += NbValPerVec;
                }
            }
        }
    }


public:
    static inline void  BsSequentialV2(SortType val[], const IndexType size){
        const IndexType nbVecInArray = (size/NbValPerVec);
        if(nbVecInArray*NbValPerVec != size){
            throw std::invalid_argument("Size of array must be a multiple of 512 bytes");
        }

        if(nbVecInArray == 1){
            SortVec(val);
            return;
        }

        IndexType n = 1;
        while( n <= nbVecInArray ) n *= 2;

        for (IndexType s=2; s <= n; s*=2) {
            IndexType i=0;

            for (; i+s < nbVecInArray; i+= s) {
                mergeUpV2((val+i*NbValPerVec), s);
            }

            {
                mergeUpV2Limite((val+i*NbValPerVec), s, nbVecInArray-i);
            }
        }
    }
};


template <class SortType, class IndexType = size_t>
class BitonicSortAVX512WithMerge{
    static const IndexType NbValPerVec = 64/sizeof(SortType);

    inline static void mergeUpV2(SortType *arr, IndexType n) {
        {
            const IndexType step = n/2;
            SortType* __restrict__ leftPtr = &arr[0];
            SortType* __restrict__ rightPtr = &arr[(n-1)*NbValPerVec];
            for (IndexType k=0 ; k < step ; k++) {
                Merge2Vec(leftPtr, rightPtr);
                leftPtr += NbValPerVec;
                rightPtr -= NbValPerVec;
            }
        }
        for(IndexType step = n/2/2 ; step > 0 ; step /= 2) {
            for (IndexType i=0; i < n; i += step*2 ) {
                SortType* __restrict__ leftPtr = &arr[i*NbValPerVec];
                SortType* __restrict__ rightPtr = &arr[(i+step)*NbValPerVec];
                for (IndexType k=0 ; k < step ; k++) {
                    Merge2Vec(leftPtr, rightPtr);
                    leftPtr += NbValPerVec;
                    rightPtr += NbValPerVec;
                }
            }
        }
    }

    inline static void mergeUpV2Limite(SortType *arr, IndexType n,
                                       IndexType limite) {
        if(limite <= 1){
            return;
        }

        if(limite > n/2){
            SortType* __restrict__ leftPtr = &arr[(n - limite)*NbValPerVec];
            SortType* __restrict__ rightPtr = &arr[(limite-1)*NbValPerVec];
            while(leftPtr < rightPtr) {
                Merge2Vec(leftPtr, rightPtr);
                leftPtr += NbValPerVec;
                rightPtr -= NbValPerVec;
            }
        }
        for(IndexType step = n/2/2 ; step > 0 ; step /= 2) {
            IndexType i=0;
            for (; i+step*2 < limite; i += step*2 ) {
                SortType* __restrict__ leftPtr = &arr[i*NbValPerVec];
                SortType* __restrict__ rightPtr = &arr[(i+step)*NbValPerVec];
                for (IndexType k=0 ; k < step ; k++) {
                    Merge2Vec(leftPtr, rightPtr);
                    leftPtr += NbValPerVec;
                    rightPtr += NbValPerVec;
                }
            }
            {
                SortType* __restrict__ leftPtr = &arr[i*NbValPerVec];
                SortType* __restrict__ rightPtr = &arr[(i+step)*NbValPerVec];
                for (IndexType k=0 ; k+i+step < limite ; k++) {
                    Merge2Vec(leftPtr, rightPtr);
                    leftPtr += NbValPerVec;
                    rightPtr += NbValPerVec;
                }
            }
        }
    }


public:
    static inline void  BsSequentialV2(SortType val[], const IndexType size){
        const IndexType nbVecInArray = (size/NbValPerVec);
        if(nbVecInArray*NbValPerVec != size){
            throw std::invalid_argument("Size of array must be a multiple of 512 bytes");
        }

        if(nbVecInArray == 1){
            SortVec(val);
            return;
        }

        for(IndexType idx = 0 ; idx < size ; idx += NbValPerVec){
            SortVec(&val[idx]);
        }

        IndexType n = 1;
        while( n <= nbVecInArray ) n *= 2;

        for (IndexType s=2; s <= n; s*=2) {
            IndexType i=0;

            for (; i+s < nbVecInArray; i+= s) {
                mergeUpV2((val+i*NbValPerVec), s);
            }

            {
                mergeUpV2Limite((val+i*NbValPerVec), s, nbVecInArray-i);
            }
        }
    }
};


template <class SortType, class IndexType = size_t>
class BitonicSortAVX512WithTest{
    static const IndexType NbValPerVec = 64/sizeof(SortType);

    inline static void mergeUpV2(SortType *arr, IndexType n) {
        {
            const IndexType step = n/2;
            SortType* __restrict__ leftPtr = &arr[0];
            SortType* __restrict__ rightPtr = &arr[(n-1)*NbValPerVec];
            for (IndexType k=0 ; k < step ; k++) {
                Sort2VecWithTest(leftPtr, rightPtr);
                leftPtr += NbValPerVec;
                rightPtr -= NbValPerVec;
            }
        }
        for(IndexType step = n/2/2 ; step > 0 ; step /= 2) {
            for (IndexType i=0; i < n; i += step*2 ) {
                SortType* __restrict__ leftPtr = &arr[i*NbValPerVec];
                SortType* __restrict__ rightPtr = &arr[(i+step)*NbValPerVec];
                for (IndexType k=0 ; k < step ; k++) {
                    Sort2VecWithTest(leftPtr, rightPtr);
                    leftPtr += NbValPerVec;
                    rightPtr += NbValPerVec;
                }
            }
        }
    }

    inline static void mergeUpV2Limite(SortType *arr, IndexType n,
                                       IndexType limite) {
        if(limite <= 1){
            return;
        }

        if(limite > n/2){
            SortType* __restrict__ leftPtr = &arr[(n - limite)*NbValPerVec];
            SortType* __restrict__ rightPtr = &arr[(limite-1)*NbValPerVec];
            while(leftPtr < rightPtr) {
                Sort2VecWithTest(leftPtr, rightPtr);
                leftPtr += NbValPerVec;
                rightPtr -= NbValPerVec;
            }
        }
        for(IndexType step = n/2/2 ; step > 0 ; step /= 2) {
            IndexType i=0;
            for (; i+step*2 < limite; i += step*2 ) {
                SortType* __restrict__ leftPtr = &arr[i*NbValPerVec];
                SortType* __restrict__ rightPtr = &arr[(i+step)*NbValPerVec];
                for (IndexType k=0 ; k < step ; k++) {
                    Sort2VecWithTest(leftPtr, rightPtr);
                    leftPtr += NbValPerVec;
                    rightPtr += NbValPerVec;
                }
            }
            {
                SortType* __restrict__ leftPtr = &arr[i*NbValPerVec];
                SortType* __restrict__ rightPtr = &arr[(i+step)*NbValPerVec];
                for (IndexType k=0 ; k+i+step < limite ; k++) {
                    Sort2VecWithTest(leftPtr, rightPtr);
                    leftPtr += NbValPerVec;
                    rightPtr += NbValPerVec;
                }
            }
        }
    }


public:
    static inline void  BsSequentialV2(SortType val[], const IndexType size){
        const IndexType nbVecInArray = (size/NbValPerVec);
        if(nbVecInArray*NbValPerVec != size){
            throw std::invalid_argument("Size of array must be a multiple of 512 bytes");
        }

        if(nbVecInArray == 1){
            SortVecWithTest(val);
            return;
        }

        IndexType n = 1;
        while( n <= nbVecInArray ) n *= 2;

        for (IndexType s=2; s <= n; s*=2) {
            IndexType i=0;

            for (; i+s < nbVecInArray; i+= s) {
                mergeUpV2((val+i*NbValPerVec), s);
            }

            {
                mergeUpV2Limite((val+i*NbValPerVec), s, nbVecInArray-i);
            }
        }
    }
};




template <class SortType, class IndexType = size_t>
class BitonicSortAVX512V2{
    static const IndexType NbValPerVec = 64/sizeof(SortType);


    static inline void ExchangeInverse(int* __restrict__ ptr1, int* __restrict__ ptr2){
        __m512i input1 = _mm512_loadu_si512(ptr1);
        __m512i input2 = _mm512_loadu_si512(ptr2);
        __m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                              8, 9, 10, 11, 12, 13, 14, 15);
        __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
        input2 = _mm512_max_epi32(input1, permNeigh2);
        input1 = _mm512_min_epi32(input1, permNeigh2);
        _mm512_storeu_si512(ptr1, input1);
        _mm512_storeu_si512(ptr2, input2);
    }
    static inline void ExchangeInverse(double* __restrict__ ptr1, double* __restrict__ ptr2){
        __m512d input1 = _mm512_loadu_pd(ptr1);
        __m512d input2 = _mm512_loadu_pd(ptr2);
        __m512i idxNoNeigh = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
        input2 = _mm512_max_pd(input1, permNeigh2);
        input1 = _mm512_min_pd(input1, permNeigh2);
        _mm512_storeu_pd(ptr1, input1);
        _mm512_storeu_pd(ptr2, input2);
    }
    static inline void ExchangeNormal(int* __restrict__ ptr1, int* __restrict__ ptr2){
        __m512i input1 = _mm512_loadu_si512(ptr1);
        __m512i input2 = _mm512_loadu_si512(ptr2);
        __m512i input1Copy = input1;
        input1 = _mm512_min_epi32(input1Copy, input2);
        input2 = _mm512_max_epi32(input1Copy, input2);
        _mm512_storeu_si512(ptr1, input1);
        _mm512_storeu_si512(ptr2, input2);
    }
    static inline void ExchangeNormal(double* __restrict__ ptr1, double* __restrict__ ptr2){
        __m512d input1 = _mm512_loadu_pd(ptr1);
        __m512d input2 = _mm512_loadu_pd(ptr2);
        __m512d input1Copy = input1;
        input1 = _mm512_min_pd(input1Copy, input2);
        input2 = _mm512_max_pd(input1Copy, input2);
        _mm512_storeu_pd(ptr1, input1);
        _mm512_storeu_pd(ptr2, input2);
    }

    inline static void mergeUpV2(SortType *arr, IndexType n) {
        {
            const IndexType step = n/2;
            SortType* __restrict__ leftPtr = &arr[0];
            SortType* __restrict__ rightPtr = &arr[(n-1)*NbValPerVec];
            for (IndexType k=0 ; k < step ; k++) {
                ExchangeInverse(leftPtr, rightPtr);
                leftPtr += NbValPerVec;
                rightPtr -= NbValPerVec;
            }
        }
        for(IndexType step = n/2/2 ; step > 0 ; step /= 2) {
            for (IndexType i=0; i < n; i += step*2 ) {
                SortType* __restrict__ leftPtr = &arr[i*NbValPerVec];
                SortType* __restrict__ rightPtr = &arr[(i+step)*NbValPerVec];
                for (IndexType k=0 ; k < step ; k++) {
                    ExchangeNormal(leftPtr, rightPtr);
                    leftPtr += NbValPerVec;
                    rightPtr += NbValPerVec;
                }
            }
        }
    }

    inline static void mergeUpV2Limite(SortType *arr, IndexType n,
                                       IndexType limite) {
        if(limite <= 1){
            return;
        }

        if(limite > n/2){
            SortType* __restrict__ leftPtr = &arr[(n - limite)*NbValPerVec];
            SortType* __restrict__ rightPtr = &arr[(limite-1)*NbValPerVec];
            while(leftPtr < rightPtr) {
                ExchangeInverse(leftPtr, rightPtr);
                leftPtr += NbValPerVec;
                rightPtr -= NbValPerVec;
            }
        }
        for(IndexType step = n/2/2 ; step > 0 ; step /= 2) {
            IndexType i=0;
            for (; i+step*2 < limite; i += step*2 ) {
                SortType* __restrict__ leftPtr = &arr[i*NbValPerVec];
                SortType* __restrict__ rightPtr = &arr[(i+step)*NbValPerVec];
                for (IndexType k=0 ; k < step ; k++) {
                    ExchangeNormal(leftPtr, rightPtr);
                    leftPtr += NbValPerVec;
                    rightPtr += NbValPerVec;
                }
            }
            {
                SortType* __restrict__ leftPtr = &arr[i*NbValPerVec];
                SortType* __restrict__ rightPtr = &arr[(i+step)*NbValPerVec];
                for (IndexType k=0 ; k+i+step < limite ; k++) {
                    ExchangeNormal(leftPtr, rightPtr);
                    leftPtr += NbValPerVec;
                    rightPtr += NbValPerVec;
                }
            }
        }
    }


public:
    static inline void  BsSequentialV2(SortType val[], const IndexType size){
        const IndexType nbVecInArray = (size/NbValPerVec);
        if(nbVecInArray*NbValPerVec != size){
            throw std::invalid_argument("Size of array must be a multiple of 512 bytes");
        }

        for(IndexType idxvec = 0 ; idxvec < size ; idxvec += NbValPerVec){
            SortVecBitFull(&val[idxvec]);
        }

        if(nbVecInArray == 1){
            return;
        }

        IndexType n = 1;
        while( n <= nbVecInArray ) n *= 2;

        for (IndexType s=2; s <= n; s*=2) {
            IndexType i=0;

            for (; i+s < nbVecInArray; i+= s) {
                mergeUpV2((val+i*NbValPerVec), s);
            }

            {
                mergeUpV2Limite((val+i*NbValPerVec), s, nbVecInArray-i);
            }

            for(IndexType idxvec = 0 ; idxvec < size ; idxvec += NbValPerVec){
                SortVecBitFull(&val[idxvec]);
            }
        }
    }
};


////////////////////////////////////////////////////////////
/// Heap sort
////////////////////////////////////////////////////////////


template <class SortType>
class HeapSort512{
    using IndexType = long;
    static const IndexType NbValPerVec = 64/sizeof(SortType);

    static inline void swap512(int* __restrict__ ptr1, int* __restrict__ ptr2){
        __m512i input = _mm512_loadu_si512(ptr1);
        __m512i input2 = _mm512_loadu_si512(ptr2);
        _mm512_storeu_si512(ptr1, input2);
        _mm512_storeu_si512(ptr2, input);
    }

    static inline void swap512(double* __restrict__ ptr1, double* __restrict__ ptr2){
        __m512d input = _mm512_loadu_pd(ptr1);
        __m512d input2 = _mm512_loadu_pd(ptr2);
        _mm512_storeu_pd(ptr1, input2);
        _mm512_storeu_pd(ptr2, input);
    }

    static inline void swapMaxLimited(int* __restrict__ ptr1, int* __restrict__ ptr2, const int lastVecSize){
        const int rest = NbValPerVec-lastVecSize;
        __m512i v1 = _mm512_loadu_si512(ptr1);
        __m512i v2 = _mm512_or_si512(_mm512_maskz_loadu_epi32(0xFFFF>>rest, ptr2),
                                     _mm512_maskz_set1_epi32(0xFFFF<<lastVecSize, INT_MAX));
        Sort2VecBitFull(v1,v2);
        _mm512_storeu_si512(ptr1, v1);
        _mm512_mask_compressstoreu_epi32(ptr2, 0xFFFF>>rest, v2);
    }

    static inline void swapMaxLimited(double* __restrict__ ptr1, double* __restrict__ ptr2, const int lastVecSize){
        const int rest = NbValPerVec-lastVecSize;
        const double temp_DBL_MAX = DBL_MAX;
        const long int double_max = reinterpret_cast<const long int&>(temp_DBL_MAX);
        __m512d v1 = _mm512_loadu_pd(ptr1);
        __m512d v2 = _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(_mm512_maskz_loadu_pd(0xFF>>rest, ptr2)),
                                                         _mm512_maskz_set1_epi64(0xFF<<lastVecSize, double_max)));
        Sort2VecBitFull(v1, v2);
        _mm512_storeu_pd(ptr1, v1);
        _mm512_mask_compressstoreu_pd(ptr2, 0xFF>>rest, v2);
    }

    static inline bool ExchangeInverseSort(int* __restrict__ ptr1, int* __restrict__ ptr2){
        __m512i input = _mm512_loadu_si512(ptr1);
        __m512i input2 = _mm512_loadu_si512(ptr2);
        {
            __m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                                  8, 9, 10, 11, 12, 13, 14, 15);
            __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
            __mmask16 compMaskOdd = _mm512_cmp_epi32_mask(permNeigh, input, _MM_CMPINT_GT);
            if(compMaskOdd == 0){
                return false;
            }
            input = _mm512_min_epi32(input2, permNeigh);
            input2 = _mm512_permutexvar_epi32(idxNoNeigh, _mm512_max_epi32(input2, permNeigh));
        }
        {
            __m512i idxNoNeigh = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,
                                                  15, 14, 13, 12, 11, 10, 9, 8);
            __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
            __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
            __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
            __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
            __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
            __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
            input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);
            input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xFF00, permNeighMax2);
        }
        {
            __m512i idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,
                                                  3, 2, 1, 0, 7, 6, 5, 4);
            __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
            __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
            __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
            __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
            __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
            __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
            input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);
            input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xF0F0, permNeighMax2);
        }
        {
            __m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,
                                                  5, 4, 7, 6, 1, 0, 3, 2);
            __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
            __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
            __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
            __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
            __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
            __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
            input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);
            input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);
        }
        {
            __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,
                                                  6, 7, 4, 5, 2, 3, 0, 1);
            __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);
            __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);
            __m512i permNeighMin = _mm512_min_epi32(permNeigh, input);
            __m512i permNeighMin2 = _mm512_min_epi32(permNeigh2, input2);
            __m512i permNeighMax = _mm512_max_epi32(permNeigh, input);
            __m512i permNeighMax2 = _mm512_max_epi32(permNeigh2, input2);
            input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);
            input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);
        }
        _mm512_storeu_si512(ptr1, input);
        _mm512_storeu_si512(ptr2, input2);
        return true;
    }
    static inline bool ExchangeInverseSort(double* __restrict__ ptr1, double* __restrict__ ptr2){
        __m512d input = _mm512_loadu_pd(ptr1);
        __m512d input2 = _mm512_loadu_pd(ptr2);
        {
            __m512i idxNoNeigh = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
            __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
            __mmask8 compMaskOdd = _mm512_cmp_pd_mask(permNeigh, input, _CMP_LT_OQ);
            if(compMaskOdd == 0){
                return false;
            }
            input = _mm512_min_pd(input2, permNeigh);
            input2 = _mm512_max_pd(input2, permNeigh);
        }
        {
            __m512i idxNoNeigh = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
            __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
            __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
            __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
            __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
            __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
            __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
            input = _mm512_mask_mov_pd(permNeighMin, 0xF0, permNeighMax);
            input2 = _mm512_mask_mov_pd(permNeighMin2, 0xF0, permNeighMax2);
        }
        {
            __m512i idxNoNeigh = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
            __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
            __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
            __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
            __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
            __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
            __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
            input = _mm512_mask_mov_pd(permNeighMin, 0xCC, permNeighMax);
            input2 = _mm512_mask_mov_pd(permNeighMin2, 0xCC, permNeighMax2);
        }
        {
            __m512i idxNoNeigh = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
            __m512d permNeigh = _mm512_permutexvar_pd(idxNoNeigh, input);
            __m512d permNeigh2 = _mm512_permutexvar_pd(idxNoNeigh, input2);
            __m512d permNeighMin = _mm512_min_pd(permNeigh, input);
            __m512d permNeighMin2 = _mm512_min_pd(permNeigh2, input2);
            __m512d permNeighMax = _mm512_max_pd(permNeigh, input);
            __m512d permNeighMax2 = _mm512_max_pd(permNeigh2, input2);
            input = _mm512_mask_mov_pd(permNeighMin, 0xAA, permNeighMax);
            input2 = _mm512_mask_mov_pd(permNeighMin2, 0xAA, permNeighMax2);
        }
        _mm512_storeu_pd(ptr1, input);
        _mm512_storeu_pd(ptr2, input2);
        return true;
    }


    static inline void heapify(SortType array[], const IndexType idx, const IndexType max){
        const IndexType idxLeft = idx*2+1;
        if(idxLeft < max ){
            auto hasChanged = ExchangeInverseSort(&array[NbValPerVec*idxLeft], &array[NbValPerVec*idx]);
            if(hasChanged){
                heapify(array, idxLeft, max);
            }
        }

        const IndexType idxRight = idx*2+2;
        if(idxRight < max ){
            auto hasChanged = ExchangeInverseSort(&array[NbValPerVec*idxRight], &array[NbValPerVec*idx]);
            if(hasChanged){
                heapify(array, idxRight, max);
            }
        }
    }

public:

    static inline void HeapSort(SortType array[], IndexType size){
        const IndexType nbVecInArray = (size/NbValPerVec);
        if(nbVecInArray*NbValPerVec != size){
            throw std::invalid_argument("Size of array must be a multiple of 512 bytes");
        }

        for(IndexType idxvec = 0 ; idxvec < nbVecInArray ; ++idxvec){
            SortVecBitFull(&array[idxvec*NbValPerVec]);
        }

        // buildHeap
        for(IndexType idx1 = nbVecInArray/2-1 ; idx1 >= 0 ; --idx1){
            heapify(array, idx1, nbVecInArray);
        }

        for(IndexType idx1 = nbVecInArray-1 ; idx1 > 0 ; --idx1){
            swap512(&array[0], &array[idx1*NbValPerVec]);
            heapify(array, 0, idx1);
        }
    }


    static inline void HeapSortNotMultiple(SortType array[], IndexType size){
        const IndexType nbVecInArray = (size/NbValPerVec);
        if(nbVecInArray*NbValPerVec == size){
            HeapSort(array, size);
            return;
        }
        if(size < NbValPerVec){
            SortByVecBitFull(array, size);
            return;
        }

        for(IndexType idxvec = 0 ; idxvec < nbVecInArray ; ++idxvec){
            SortVecBitFull(&array[idxvec*NbValPerVec]);
        }

        // buildHeap
        for(IndexType idx1 = nbVecInArray/2-1 ; idx1 >= 0 ; --idx1){
            heapify(array, idx1, nbVecInArray);
        }

        {
            // Swap max
            swapMaxLimited(&array[0], &array[nbVecInArray*NbValPerVec], size-nbVecInArray*NbValPerVec);
            heapify(array, 0, nbVecInArray);
        }

        for(IndexType idx1 = nbVecInArray-1 ; idx1 > 0 ; --idx1){
            swap512(&array[0], &array[idx1*NbValPerVec]);
            heapify(array, 0, idx1);
        }
    }
};

template <class SortType>
class HeapSort{
    static inline void heapify(SortType array[], const long idx, const long max){
        const long idxLeft = idx*2+1;
        const long idxRight = idx*2+2;
        long largestIdx = idx;

        if(idxLeft < max && array[idxLeft] > array[idx] ){
            largestIdx = idxLeft;
        }
        if(idxRight < max && array[idxRight] > array[largestIdx] ){
            largestIdx = idxRight;
        }
        if(largestIdx != idx){
            std::swap(array[largestIdx], array[idx]);
            heapify(array, largestIdx, max);
        }
    }

public:
    static inline void sort(SortType array[], const long size){
        // buildHeap
        for(long idx1 = size/2-1 ; idx1 >= 0 ; --idx1){
            heapify(array, idx1, size);
        }

        for(long idx1 = size-1 ; idx1 > 0 ; --idx1){
            std::swap(array[0], array[idx1]);
            heapify(array, 0, idx1);
        }
    }
};

////////////////////////////////////////////////////////////
/// Insertion sort Functions
////////////////////////////////////////////////////////////

template <class SortType, class IndexType = size_t>
static inline void InsertionSort(SortType array[], IndexType size){
    for(IndexType idx1 = size-1 ; idx1 > 0 ; --idx1){
        IndexType maxValIdx = idx1;
        for(IndexType idx2 = 0 ; idx2 < idx1 ; ++idx2){
            if(array[idx2] > array[maxValIdx]){
                maxValIdx = idx2;
            }
        }
        std::swap(array[maxValIdx], array[idx1]);
    }
}


////////////////////////////////////////////////////////////
/// QuickSort Functions
////////////////////////////////////////////////////////////


template <class SortType, class IndexType = size_t>
class FQuickSort {
public:
    ////////////////////////////////////////////////////////////
    // Quick sort
    ////////////////////////////////////////////////////////////

    /* Use in the sequential qs */
    static inline IndexType QsPartition(SortType array[], IndexType left, IndexType right,
                                        const SortType pivot){

        for(; left <= right
             && array[left] <= pivot ; ++left){
        }

        for(IndexType idx = left ; idx <= right ; ++idx){
            if( array[idx] <= pivot ){
                std::swap(array[idx],array[left]);
                left += 1;
            }
        }

        return left;
    }

    /* Use in the sequential qs */
    static inline IndexType QsPivotPartition(SortType array[], IndexType left, IndexType right){
        std::swap(array[right],array[((right - left ) / 2) + left]);

        for( IndexType idx = left; idx < right ; ++idx){
            if( array[idx] <= array[right] ){
                std::swap(array[idx],array[left]);
                left += 1;
            }
        }

        std::swap(array[left],array[right]);

        return left;
    }


    /* The sequential qs */
    static void QsSequentialStep(SortType array[], const IndexType left, const IndexType right){
        if(right-left < 64){
            InsertionSort(array+left, right-left+1);
        }
        else {
            const IndexType part = QsPivotPartition(array, left, right);
            if(part+1 < right) QsSequentialStep(array,part + 1,right);
            if(part && left < part-1) QsSequentialStep(array,left,part - 1);
        }
    }

    /** A task dispatcher */
    static inline void QsOmpTask(SortType array[], const IndexType left, const IndexType right, const int deep){
        if(right-left < 64){
            InsertionSort(array+left, right-left+1);
        }
        else {
            const IndexType part = QsPivotPartition(array, left, right);
            if( deep ){
                // default(none) has been removed for clang compatibility
                if(part+1 < right){
                    #pragma omp task firstprivate(array, part, right, deep)
                    QsOmpTask(array,part + 1,right, deep - 1);
                }
                // #pragma omp task default(none) firstprivate(array, part, right, deep, infOrEqual) // not needed
                if(part && left < part-1) QsOmpTask(array,left,part - 1, deep - 1);
            }
            else {
                if(part+1 < right) QsSequentialStep(array,part + 1,right);
                if(part && left < part-1) QsSequentialStep(array,left,part - 1);
            }
        }
    }

    /* a sequential qs */
    static inline void QsSequential(SortType array[], const IndexType size){
        QsSequentialStep(array, 0, size-1);
    }

    /** The openmp quick sort */
    static inline void QsOmp(SortType array[], const IndexType size){
        const int nbTasksRequiere = (omp_get_max_threads() * 5);
        int deep = 0;
        while( (1 << deep) < nbTasksRequiere ) deep += 1;

#pragma omp parallel
        {
#pragma omp master
            {
                QsOmpTask(array, 0, size - 1 , deep);
            }
        }
    }
};


////////////////////////////////////////////////////////////
/// New QuickSort Functions
////////////////////////////////////////////////////////////



template <class SortType, class IndexType = size_t>
class NewQuickSort {
public:
    static inline IndexType PivotPartition(SortType array[], const IndexType left, const IndexType right){
        std::swap(array[((right - left ) / 2) + left], array[right]);
        const int part = Partition(array, left, right-1,
                         array[right]);
        std::swap(array[part], array[right]);
        return part;
    }

    /* a sequential qs */
    static inline IndexType Partition(SortType array[], IndexType left, IndexType right,
                                 const SortType pivot){
        const IndexType S = 1;

        if(right-left == 0) return (array[left] <= pivot?left+1:left);

        SortType left_val = array[left];
        IndexType left_w = left;
        left += S;

        right -= S-1;
        SortType right_val = array[right];
        IndexType right_w = right+S;

        while(right-left >= S){

            const IndexType free_left = left - left_w;
            const IndexType free_right = right_w - right;

            SortType val;
            if( free_left <= free_right ){
                val = array[left];
                left += S;
            }
            else{
                right -= S;
                val = array[right];
            }

            const bool mask = (val <= pivot);

            if(mask){
                const IndexType nb_low = 1;
                array[left_w] = val;
                left_w += nb_low;
            }
            if(!mask){
                const IndexType nb_high = 1;
                right_w -= nb_high;
                array[right_w] = val;
            }
        }


        {
            const bool mask = (left_val <= pivot);

            if(mask){
                const IndexType nb_low = 1; // count mask
                array[left_w] = left_val;
                left_w += nb_low;
            }
            if(!mask){
                const IndexType nb_high = 1; // S-nb_low
                right_w -= nb_high;
                array[right_w] = left_val;
            }
        }
        {
            const bool mask = (right_val <= pivot);

            if(mask){
                const IndexType nb_low = 1; // count mask
                array[left_w] = right_val;
                left_w += nb_low;
            }
            if(!mask){
                const IndexType nb_high = 1; // S-nb_low
                right_w -= nb_high;
                array[right_w] = right_val;
            }
        }
        return left_w;
    }

    /* The sequential qs */
    static void QsSequentialStep(SortType array[], const IndexType left, const IndexType right){
        if(right-left < 64){
            InsertionSort(array+left, right-left+1);
        }
        else {
            const IndexType part = PivotPartition(array, left, right);
            if(part+1 < right) QsSequentialStep(array,part + 1,right);
            if(part && left < part-1) QsSequentialStep(array,left,part - 1);
        }
    }

    /** A task dispatcher */
    static inline void QsTask(SortType array[], const IndexType left, const IndexType right, const int deep){
        if(right-left < 64){
            InsertionSort(array+left, right-left+1);
        }
        else{
            const IndexType part = PivotPartition(array, left, right);
            if( deep ){
                // default(none) has been removed for clang compatibility
                if(part+1 < right) {
                    #pragma omp task firstprivate(array, part, right, deep)
                    QsTask(array,part + 1,right, deep - 1);
                }
                // #pragma omp task default(none) firstprivate(array, part, right, deep, infOrEqual) // not needed
                if(part && left < part-1) QsTask(array,left,part - 1, deep - 1);
            }
            else {
                if(part+1 < right) QsSequentialStep(array,part + 1,right);
                if(part && left < part-1) QsSequentialStep(array,left,part - 1);
            }
        }
    }

    /* a sequential qs */
    static inline void QsSequential(SortType array[], const IndexType size){
        QsSequentialStep(array, 0, size-1);
    }

    /** The openmp quick sort */
    static inline void QsOmp(SortType array[], const IndexType size){
        const int nbTasksRequiere = (omp_get_max_threads() * 5);
        int deep = 0;
        while( (1 << deep) < nbTasksRequiere ) deep += 1;

#pragma omp parallel
        {
#pragma omp master
            {
                QsTask(array, 0, size - 1 , deep);
            }
        }
    }

};


inline int popcount(__mmask8 mask){
//    int res = int(mask);
//    res = (0x55 & res) + (0x55 & (res >> 1));
//    res = (res & 0x33) + ((res>>2) & 0x33);
//    return (res & 0xF) + ((res>>4) & 0xF);
#ifdef __INTEL_COMPILER
        return _mm_countbits_32(mask);
#else
        return __builtin_popcount(mask);
#endif
}

inline int popcount(__mmask16 mask){
//    int res = int(mask);
//    res = (0x5555 & res) + (0x5555 & (res >> 1));
//    res = (res & 0x3333) + ((res>>2) & 0x3333);
//    res = (res & 0x0F0F) + ((res>>4) & 0x0F0F);
//    return (res & 0xFF) + ((res>>8) & 0xFF);
#ifdef __INTEL_COMPILER
        return _mm_countbits_32(mask);
#else
        return __builtin_popcount(mask);
#endif
}


/* a sequential qs */
template <class IndexType>
static inline IndexType Partition512(int array[], IndexType left, IndexType right,
                             const int pivot){
    const IndexType S = 16;//(512/8)/sizeof(int);

    if(right-left+1 < 2*S){
        return FQuickSort<int,IndexType>::QsPartition(array, left, right, pivot);
    }

    __m512i pivotvec = _mm512_set1_epi32(pivot);

    __m512i left_val = _mm512_loadu_si512(&array[left]);
    IndexType left_w = left;
    left += S;

    IndexType right_w = right+1;
    right -= S-1;
    __m512i right_val = _mm512_loadu_si512(&array[right]);

    while(left + S <= right){
        const IndexType free_left = left - left_w;
        const IndexType free_right = right_w - right;

        __m512i val;
        if( free_left <= free_right ){
            val = _mm512_loadu_si512(&array[left]);
            left += S;
        }
        else{
            right -= S;
            val = _mm512_loadu_si512(&array[right]);
        }

        __mmask16 mask = _mm512_cmp_epi32_mask(val, pivotvec, _MM_CMPINT_LE);

        const IndexType nb_low = popcount(mask); // count mask
        // intel _popcnt32 or _mm_countbits_32 or __builtin_popcount(mask)
        const IndexType nb_high = S-nb_low; // S-nb_low

        if(mask){// if nb_low
            _mm512_mask_compressstoreu_epi32(&array[left_w],mask,val);
            left_w += nb_low;
        }
        if(mask != 0xFFFF){// if nb_high
            right_w -= nb_high;
            _mm512_mask_compressstoreu_epi32(&array[right_w],~mask,val);
        }
    }

    {
        const IndexType remaining = right - left;
        __m512i val = _mm512_loadu_si512(&array[left]);
        left = right;

        __mmask16 mask = _mm512_cmp_epi32_mask(val, pivotvec, _MM_CMPINT_LE);

        __mmask16 mask_low = mask & ~(0xFFFF << remaining);
        __mmask16 mask_high = (~mask) & ~(0xFFFF << remaining);

        const IndexType nb_low = popcount(mask_low); // count mask
        // intel _popcnt32 or _mm_countbits_32 or __builtin_popcount(mask)
        const IndexType nb_high = popcount(mask_high); // S-nb_low

        if(mask_low){// if nb_low
            _mm512_mask_compressstoreu_epi32(&array[left_w],mask_low,val);
            left_w += nb_low;
        }
        if(mask_high){// if nb_high
            right_w -= nb_high;
            _mm512_mask_compressstoreu_epi32(&array[right_w],mask_high,val);
        }
    }
    {
        __mmask16 mask = _mm512_cmp_epi32_mask(left_val, pivotvec, _MM_CMPINT_LE);

        const IndexType nb_low = popcount(mask); // count mask
        // intel _popcnt32 or _mm_countbits_32 or __builtin_popcount(mask)
        const IndexType nb_high = S-nb_low; // S-nb_low

        if(mask){// if nb_low
            _mm512_mask_compressstoreu_epi32(&array[left_w],mask,left_val);
            left_w += nb_low;
        }
        if(mask != 0xFFFF){// if nb_high
            right_w -= nb_high;
            _mm512_mask_compressstoreu_epi32(&array[right_w],~mask,left_val);
        }
    }
    {
        __mmask16 mask = _mm512_cmp_epi32_mask(right_val, pivotvec, _MM_CMPINT_LE);

        const IndexType nb_low = popcount(mask); // count mask
        // intel _popcnt32 or _mm_countbits_32 or __builtin_popcount(mask)
        const IndexType nb_high = S-nb_low; // S-nb_low

        if(mask){// if nb_low
            _mm512_mask_compressstoreu_epi32(&array[left_w],mask,right_val);
            left_w += nb_low;
        }
        if(mask != 0xFFFF){// if nb_high
            right_w -= nb_high;
            _mm512_mask_compressstoreu_epi32(&array[right_w],~mask,right_val);
         }
    }
    return left_w;
}


template <class IndexType>
static inline IndexType Partition512(double array[], IndexType left, IndexType right,
                             const double pivot){
    const IndexType S = 8;//(512/8)/sizeof(double);

    if(right-left+1 < 2*S){
        return FQuickSort<double,IndexType>::QsPartition(array, left, right, pivot);
    }

    __m512d pivotvec = _mm512_set1_pd(pivot);

    __m512d left_val = _mm512_loadu_pd(&array[left]);
    IndexType left_w = left;
    left += S;

    IndexType right_w = right+1;
    right -= S-1;
    __m512d right_val = _mm512_loadu_pd(&array[right]);

    while(left + S <= right){
        const IndexType free_left = left - left_w;
        const IndexType free_right = right_w - right;

        __m512d val;
        if( free_left <= free_right ){
            val = _mm512_loadu_pd(&array[left]);
            left += S;
        }
        else{
            right -= S;
            val = _mm512_loadu_pd(&array[right]);
        }

        __mmask8 mask = _mm512_cmp_pd_mask(val, pivotvec, _CMP_LE_OQ);

        const IndexType nb_low = popcount(mask); // count mask
        // intel _popcnt32 or _mm_countbits_32 or __builtin_popcount(mask)
        const IndexType nb_high = S-nb_low; // S-nb_low

        if(mask){// if nb_low
            _mm512_mask_compressstoreu_pd(&array[left_w],mask,val);
            left_w += nb_low;
        }
        if(mask != 0xFF){// if nb_high
            right_w -= nb_high;
            _mm512_mask_compressstoreu_pd(&array[right_w],~mask,val);
        }
    }

    {
        const IndexType remaining = right - left;
        __m512d val = _mm512_loadu_pd(&array[left]);
        left = right;

        __mmask8 mask = _mm512_cmp_pd_mask(val, pivotvec, _CMP_LE_OQ);

        __mmask8 mask_low = mask & ~(0xFF << remaining);
        __mmask8 mask_high = (~mask) & ~(0xFF << remaining);

        const IndexType nb_low = popcount(mask_low); // count mask
        // intel _popcnt32 or _mm_countbits_32 or __builtin_popcount(mask)
        const IndexType nb_high = popcount(mask_high); // S-nb_low

        if(mask_low){// if nb_low
            _mm512_mask_compressstoreu_pd(&array[left_w],mask_low,val);
            left_w += nb_low;
        }
        if(mask_high){// if nb_high
            right_w -= nb_high;
            _mm512_mask_compressstoreu_pd(&array[right_w],mask_high,val);
        }
    }
    {
        __mmask8 mask = _mm512_cmp_pd_mask(left_val, pivotvec, _CMP_LE_OQ);

        const IndexType nb_low = popcount(mask); // count mask
        // intel _popcnt32 or _mm_countbits_32 or __builtin_popcount(mask)
        const IndexType nb_high = S-nb_low; // S-nb_low

        if(mask){// if nb_low
            _mm512_mask_compressstoreu_pd(&array[left_w],mask,left_val);
            left_w += nb_low;
        }
        if(mask != 0xFF){// if nb_high
            right_w -= nb_high;
            _mm512_mask_compressstoreu_pd(&array[right_w],~mask,left_val);
        }
    }
    {
        __mmask8 mask = _mm512_cmp_pd_mask(right_val, pivotvec, _CMP_LE_OQ);

        const IndexType nb_low = popcount(mask); // count mask
        // intel _popcnt32 or _mm_countbits_32 or __builtin_popcount(mask)
        const IndexType nb_high = S-nb_low; // S-nb_low

        if(mask){// if nb_low
            _mm512_mask_compressstoreu_pd(&array[left_w],mask,right_val);
            left_w += nb_low;
        }
        if(mask != 0xFF){// if nb_high
            right_w -= nb_high;
            _mm512_mask_compressstoreu_pd(&array[right_w],~mask,right_val);
         }
    }
    return left_w;
}

template <class SortType, class IndexType = size_t>
class NewQuickSort512 {
public:
    static inline IndexType PivotPartition(SortType array[], const IndexType left, const IndexType right){
        std::swap(array[((right - left ) / 2) + left], array[right]);
        const int part = Partition512(array, left, right-1,
                         array[right]);
        std::swap(array[part], array[right]);
        return part;
    }

    static inline IndexType Partition(SortType array[], const IndexType left, const IndexType right,
                                      const SortType pivot){
        return  Partition512(array, left, right, pivot);
    }

    /* The sequential qs */
    static void QsSequentialStep(SortType array[], const IndexType left, const IndexType right){
        if(left < right){
            if(right-left < 64){
                InsertionSort(array+left, right-left+1);
            }
            else{
                const IndexType part = PivotPartition(array, left, right);
                QsSequentialStep(array,part + 1,right);
                if(part) QsSequentialStep(array,left,part - 1);
            }
        }
    }

    /** A task dispatcher */
    static inline void QsTask(SortType array[], const IndexType left, const IndexType right, const int deep){
        if(left < right){
            if(right-left < 64){
                InsertionSort(array+left, right-left+1);
            }
            else{
                const IndexType part = PivotPartition(array, left, right);
                if( deep ){
                    // default(none) has been removed for clang compatibility
    #pragma omp task firstprivate(array, part, right, deep)
                    QsTask(array,part + 1,right, deep - 1);
                    // #pragma omp task default(none) firstprivate(array, part, right, deep, infOrEqual) // not needed
                    if(part) QsTask(array,left,part - 1, deep - 1);
                }
                else {
                    QsSequentialStep(array,part + 1,right);
                    if(part) QsSequentialStep(array,left,part - 1);
                }
            }
        }
    }

    /* a sequential qs */
    static inline void QsSequential(SortType array[], const IndexType size){
        QsSequentialStep(array, 0, size-1);
    }

    /** The openmp quick sort */
    static inline void QsOmp(SortType array[], const IndexType size){
        const int nbTasksRequiere = (omp_get_max_threads() * 5);
        int deep = 0;
        while( (1 << deep) < nbTasksRequiere ) deep += 1;

#pragma omp parallel
        {
#pragma omp master
            {
                QsTask(array, 0, size - 1 , deep);
            }
        }
    }

};





/* a sequential qs */
template <class IndexType>
static inline IndexType Partition512V2(int array[], IndexType left, IndexType right,
                             const int pivot){
    const IndexType S = 16;//(512/8)/sizeof(int);

    if(right-left+1 < 2*S){
        return FQuickSort<int,IndexType>::QsPartition(array, left, right, pivot);
    }

    __m512i pivotvec = _mm512_set1_epi32(pivot);

    __m512i left_val = _mm512_loadu_si512(&array[left]);
    IndexType left_w = left;
    left += S;

    IndexType right_w = right+1;
    right -= S-1;
    __m512i right_val = _mm512_loadu_si512(&array[right]);

    while(left + S <= right){
        const IndexType free_left = left - left_w;
        const IndexType free_right = right_w - right;

        __m512i val;
        if( free_left <= free_right ){
            val = _mm512_loadu_si512(&array[left]);
            left += S;
        }
        else{
            right -= S;
            val = _mm512_loadu_si512(&array[right]);
        }

        __mmask16 mask = _mm512_cmp_epi32_mask(val, pivotvec, _MM_CMPINT_LE);

        const IndexType nb_low = popcount(mask); // count mask
        // intel _popcnt32 or _mm_countbits_32 or __builtin_popcount(mask)
        const IndexType nb_high = S-nb_low; // S-nb_low

        //if(mask){// if nb_low
            _mm512_mask_compressstoreu_epi32(&array[left_w],mask,val);
            left_w += nb_low;
        //}
        //if(mask != 0xFFFF){// if nb_high
            right_w -= nb_high;
            _mm512_mask_compressstoreu_epi32(&array[right_w],~mask,val);
        //}
    }

    {
        const IndexType remaining = right - left;
        __m512i val = _mm512_loadu_si512(&array[left]);
        left = right;

        __mmask16 mask = _mm512_cmp_epi32_mask(val, pivotvec, _MM_CMPINT_LE);

        __mmask16 mask_low = mask & ~(0xFFFF << remaining);
        __mmask16 mask_high = (~mask) & ~(0xFFFF << remaining);

        const IndexType nb_low = popcount(mask_low); // count mask
        // intel _popcnt32 or _mm_countbits_32 or __builtin_popcount(mask)
        const IndexType nb_high = popcount(mask_high); // S-nb_low

        //if(mask_low){// if nb_low
            _mm512_mask_compressstoreu_epi32(&array[left_w],mask_low,val);
            left_w += nb_low;
        //}
        //if(mask_high){// if nb_high
            right_w -= nb_high;
            _mm512_mask_compressstoreu_epi32(&array[right_w],mask_high,val);
        //}
    }
    {
        __mmask16 mask = _mm512_cmp_epi32_mask(left_val, pivotvec, _MM_CMPINT_LE);

        const IndexType nb_low = popcount(mask); // count mask
        // intel _popcnt32 or _mm_countbits_32 or __builtin_popcount(mask)
        const IndexType nb_high = S-nb_low; // S-nb_low

        //if(mask){// if nb_low
            _mm512_mask_compressstoreu_epi32(&array[left_w],mask,left_val);
            left_w += nb_low;
        //}
        //if(mask != 0xFFFF){// if nb_high
            right_w -= nb_high;
            _mm512_mask_compressstoreu_epi32(&array[right_w],~mask,left_val);
        //}
    }
    {
        __mmask16 mask = _mm512_cmp_epi32_mask(right_val, pivotvec, _MM_CMPINT_LE);

        const IndexType nb_low = popcount(mask); // count mask
        // intel _popcnt32 or _mm_countbits_32 or __builtin_popcount(mask)
        const IndexType nb_high = S-nb_low; // S-nb_low

        //if(mask){// if nb_low
            _mm512_mask_compressstoreu_epi32(&array[left_w],mask,right_val);
            left_w += nb_low;
        //}
        //if(mask != 0xFFFF){// if nb_high
            right_w -= nb_high;
            _mm512_mask_compressstoreu_epi32(&array[right_w],~mask,right_val);
         //}
    }
    return left_w;
}


template <class IndexType>
static inline IndexType Partition512V2(double array[], IndexType left, IndexType right,
                             const double pivot){
    const IndexType S = 8;//(512/8)/sizeof(double);

    if(right-left+1 < 2*S){
        return FQuickSort<double,IndexType>::QsPartition(array, left, right, pivot);
    }

    __m512d pivotvec = _mm512_set1_pd(pivot);

    __m512d left_val = _mm512_loadu_pd(&array[left]);
    IndexType left_w = left;
    left += S;

    IndexType right_w = right+1;
    right -= S-1;
    __m512d right_val = _mm512_loadu_pd(&array[right]);

    while(left + S <= right){
        const IndexType free_left = left - left_w;
        const IndexType free_right = right_w - right;

        __m512d val;
        if( free_left <= free_right ){
            val = _mm512_loadu_pd(&array[left]);
            left += S;
        }
        else{
            right -= S;
            val = _mm512_loadu_pd(&array[right]);
        }

        __mmask8 mask = _mm512_cmp_pd_mask(val, pivotvec, _CMP_LE_OQ);

        const IndexType nb_low = popcount(mask); // count mask
        // intel _popcnt32 or _mm_countbits_32 or __builtin_popcount(mask)
        const IndexType nb_high = S-nb_low; // S-nb_low

        //if(mask){// if nb_low
            _mm512_mask_compressstoreu_pd(&array[left_w],mask,val);
            left_w += nb_low;
        //}
        //if(mask != 0xFF){// if nb_high
            right_w -= nb_high;
            _mm512_mask_compressstoreu_pd(&array[right_w],~mask,val);
        //}
    }

    {
        const IndexType remaining = right - left;
        __m512d val = _mm512_loadu_pd(&array[left]);
        left = right;

        __mmask8 mask = _mm512_cmp_pd_mask(val, pivotvec, _CMP_LE_OQ);

        __mmask8 mask_low = mask & ~(0xFF << remaining);
        __mmask8 mask_high = (~mask) & ~(0xFF << remaining);

        const IndexType nb_low = popcount(mask_low); // count mask
        // intel _popcnt32 or _mm_countbits_32 or __builtin_popcount(mask)
        const IndexType nb_high = popcount(mask_high); // S-nb_low

        //if(mask_low){// if nb_low
            _mm512_mask_compressstoreu_pd(&array[left_w],mask_low,val);
            left_w += nb_low;
        //}
        //if(mask_high){// if nb_high
            right_w -= nb_high;
            _mm512_mask_compressstoreu_pd(&array[right_w],mask_high,val);
        //}
    }
    {
        __mmask8 mask = _mm512_cmp_pd_mask(left_val, pivotvec, _CMP_LE_OQ);

        const IndexType nb_low = popcount(mask); // count mask
        // intel _popcnt32 or _mm_countbits_32 or __builtin_popcount(mask)
        const IndexType nb_high = S-nb_low; // S-nb_low

        //if(mask){// if nb_low
            _mm512_mask_compressstoreu_pd(&array[left_w],mask,left_val);
            left_w += nb_low;
        //}
        //if(mask != 0xFF){// if nb_high
            right_w -= nb_high;
            _mm512_mask_compressstoreu_pd(&array[right_w],~mask,left_val);
        //}
    }
    {
        __mmask8 mask = _mm512_cmp_pd_mask(right_val, pivotvec, _CMP_LE_OQ);

        const IndexType nb_low = popcount(mask); // count mask
        // intel _popcnt32 or _mm_countbits_32 or __builtin_popcount(mask)
        const IndexType nb_high = S-nb_low; // S-nb_low

        //if(mask){// if nb_low
            _mm512_mask_compressstoreu_pd(&array[left_w],mask,right_val);
            left_w += nb_low;
        //}
        //if(mask != 0xFF){// if nb_high
            right_w -= nb_high;
            _mm512_mask_compressstoreu_pd(&array[right_w],~mask,right_val);
        //}
    }
    return left_w;
}

template <class SortType, class IndexType = size_t>
class NewQuickSort512V2 {
public:
    static const int SortLimite = 4*64/sizeof(SortType);

    static inline IndexType PivotPartition(SortType array[], const IndexType left, const IndexType right){
        std::swap(array[((right - left ) / 2) + left], array[right]);
        const int part = Partition512V2(array, left, right-1,
                         array[right]);
        std::swap(array[part], array[right]);
        return part;
    }

    static inline IndexType Partition(SortType array[], const IndexType left, const IndexType right,
                                      const SortType pivot){
        return  Partition512V2(array, left, right, pivot);
    }

    /* The sequential qs */
    static void QsSequentialStep(SortType array[], const IndexType left, const IndexType right){
        if(right-left < SortLimite){
            SortByVec(array+left, right-left+1);
        }
        else{
            const IndexType part = PivotPartition(array, left, right);
            if(part+1 < right) QsSequentialStep(array,part + 1,right);
            if(part && left < part-1)  QsSequentialStep(array,left,part - 1);
        }
    }

    /** A task dispatcher */
    static inline void QsTask(SortType array[], const IndexType left, const IndexType right, const int deep){
        if(right-left < SortLimite){
            SortByVec(array+left, right-left+1);
        }
        else{
            const IndexType part = PivotPartition(array, left, right);
            if( deep ){
                // default(none) has been removed for clang compatibility
                if(part+1 < right){
                    #pragma omp task firstprivate(array, part, right, deep)
                                    QsTask(array,part + 1,right, deep - 1);
                }
                // #pragma omp task default(none) firstprivate(array, part, right, deep, infOrEqual) // not needed
                if(part && left < part-1)  QsTask(array,left,part - 1, deep - 1);
            }
            else {
                if(part+1 < right) QsSequentialStep(array,part + 1,right);
                if(part && left < part-1)  QsSequentialStep(array,left,part - 1);
            }
        }
    }

    /* a sequential qs */
    static inline void QsSequential(SortType array[], const IndexType size){
        QsSequentialStep(array, 0, size-1);
    }

    /** The openmp quick sort */
    static inline void QsOmp(SortType array[], const IndexType size){
        const int nbTasksRequiere = (omp_get_max_threads() * 5);
        int deep = 0;
        while( (1 << deep) < nbTasksRequiere ) deep += 1;

#pragma omp parallel
        {
#pragma omp master
            {
                QsTask(array, 0, size - 1 , deep);
            }
        }
    }

};


template <class SortType, class IndexType = size_t>
class NewQuickSort512V3 {
public:
    static const int SortLimite = 8*64/sizeof(SortType);

    static inline IndexType PivotPartition(SortType array[], const IndexType left, const IndexType right){
        std::swap(array[((right - left ) / 2) + left], array[right]);
        const int part = Partition512V2(array, left, right-1,
                         array[right]);
        std::swap(array[part], array[right]);
        return part;
    }

    static inline IndexType Partition(SortType array[], const IndexType left, const IndexType right,
                                      const SortType pivot){
        return  Partition512V2(array, left, right, pivot);
    }

    /* The sequential qs */
    static void QsSequentialStep(SortType array[], const IndexType left, const IndexType right){
        if(right-left < SortLimite){
            SortByVecBitFull(array+left, right-left+1);
        }
        else{
            const IndexType part = PivotPartition(array, left, right);
            if(part+1 < right) QsSequentialStep(array,part + 1,right);
            if(part && left < part-1) QsSequentialStep(array,left,part - 1);
        }
    }

    /** A task dispatcher */
    static inline void QsTask(SortType array[], const IndexType left, const IndexType right, const int deep){
        if(right-left < SortLimite){
            SortByVecBitFull(array+left, right-left+1);
        }
        else{
            const IndexType part = PivotPartition(array, left, right);
            if( deep ){
                // default(none) has been removed for clang compatibility
                if(part+1 < right){
                    #pragma omp task firstprivate(array, part, right, deep)
                    QsTask(array,part + 1,right, deep - 1);
                }
                // #pragma omp task default(none) firstprivate(array, part, right, deep, infOrEqual) // not needed
                if(part && left < part-1)  QsTask(array,left,part - 1, deep - 1);
            }
            else {
                if(part+1 < right) QsSequentialStep(array,part + 1,right);
                if(part && left < part-1)  QsSequentialStep(array,left,part - 1);
            }
        }
    }

    /* a sequential qs */
    static inline void QsSequential(SortType array[], const IndexType size){
        QsSequentialStep(array, 0, size-1);
    }

    /** The openmp quick sort */
    static inline void QsOmp(SortType array[], const IndexType size){
        const int nbTasksRequiere = (omp_get_max_threads() * 5);
        int deep = 0;
        while( (1 << deep) < nbTasksRequiere ) deep += 1;

#pragma omp parallel
        {
#pragma omp master
            {
                QsTask(array, 0, size - 1 , deep);
            }
        }
    }

};

template <class SortType, class IndexType = size_t>
class NewQuickSort512V4 {
public:
    static const int SortLimite = 16*64/sizeof(SortType);

    static inline IndexType PivotPartition(SortType array[], const IndexType left, const IndexType right){
        std::swap(array[((right - left ) / 2) + left], array[right]);
        const int part = Partition512V2(array, left, right-1,
                         array[right]);
        std::swap(array[part], array[right]);
        return part;
    }

    static inline IndexType Partition(SortType array[], const IndexType left, const IndexType right,
                                      const SortType pivot){
        return  Partition512V2(array, left, right, pivot);
    }

    /* The sequential qs */
    static void QsSequentialStep(SortType array[], const IndexType left, const IndexType right){
        if(right-left < SortLimite){
            SortByVecBitFull(array+left, right-left+1);
        }
        else{
            const IndexType part = PivotPartition(array, left, right);
            if(part+1 < right) QsSequentialStep(array,part + 1,right);
            if(part && left < part-1)  QsSequentialStep(array,left,part - 1);
        }
    }

    /** A task dispatcher */
    static inline void QsTask(SortType array[], const IndexType left, const IndexType right, const int deep){
        if(right-left < SortLimite){
            SortByVecBitFull(array+left, right-left+1);
        }
        else{
            const IndexType part = PivotPartition(array, left, right);
            if( deep ){
                // default(none) has been removed for clang compatibility
                if(part+1 < right){
                    #pragma omp task firstprivate(array, part, right, deep)
                    QsTask(array,part + 1,right, deep - 1);
                }
                // #pragma omp task default(none) firstprivate(array, part, right, deep, infOrEqual) // not needed
                if(part && left < part-1) QsTask(array,left,part - 1, deep - 1);
            }
            else {
                if(part+1 < right) QsSequentialStep(array,part + 1,right);
                if(part && left < part-1) QsSequentialStep(array,left,part - 1);
            }
        }
    }

    /* a sequential qs */
    static inline void QsSequential(SortType array[], const IndexType size){
        QsSequentialStep(array, 0, size-1);
    }

    /** The openmp quick sort */
    static inline void QsOmp(SortType array[], const IndexType size){
        const int nbTasksRequiere = (omp_get_max_threads() * 5);
        int deep = 0;
        while( (1 << deep) < nbTasksRequiere ) deep += 1;

#pragma omp parallel
        {
#pragma omp master
            {
                QsTask(array, 0, size - 1 , deep);
            }
        }
    }

};


template <class SortType, class IndexType = size_t>
class NewQuickSort512V5 {
public:
    static inline IndexType GetPivot(const SortType array[], const IndexType left, const IndexType right){
        const IndexType middle = ((right-left)/2) + left;
        if(array[left] <= array[middle] && array[middle] <= array[right]){
            return middle;
        }
        else if(array[middle] <= array[left] && array[left] <= array[right]){
            return left;
        }
        else return right;
    }

    static const int SortLimite = 16*64/sizeof(SortType);

    static inline IndexType PivotPartition(SortType array[], const IndexType left, const IndexType right){
        if(right-left > 1){
            const IndexType pivotIdx = GetPivot(array, left, right);
            std::swap(array[pivotIdx], array[right]);
            const IndexType part = Partition512V2(array, left, right-1, array[right]);
            std::swap(array[part], array[right]);
            return part;
        }
        return left;
    }

    static inline IndexType Partition(SortType array[], const IndexType left, const IndexType right,
                                      const SortType pivot){
        return  Partition512V2(array, left, right, pivot);
    }

    /* The sequential qs */
    static void QsSequentialStep(SortType array[], const IndexType left, const IndexType right){
        if(right-left < SortLimite){
            SortByVecBitFull(array+left, right-left+1);
        }
        else{
            const IndexType part = PivotPartition(array, left, right);
            if(part+1 < right) QsSequentialStep(array,part+1,right);
            if(part && left < part-1)  QsSequentialStep(array,left,part - 1);
        }
    }

    /** A task dispatcher */
    static inline void QsTask(SortType array[], const IndexType left, const IndexType right, const int deep){
        if(right-left < SortLimite){
            SortByVecBitFull(array+left, right-left+1);
        }
        else{
            const IndexType part = PivotPartition(array, left, right);
            if( deep ){
                // default(none) has been removed for clang compatibility
                if(part+1 < right){
                    #pragma omp task firstprivate(array, part, right, deep)
                    QsTask(array,part+1,right, deep - 1);
                }
                // #pragma omp task default(none) firstprivate(array, part, right, deep, infOrEqual) // not needed
                if(part && left < part-1)  QsTask(array,left,part - 1, deep - 1);
            }
            else {
                if(part+1 < right) QsSequentialStep(array,part+1,right);
                if(part && left < part-1)  QsSequentialStep(array,left,part - 1);
            }
        }
    }

    /* a sequential qs */
    static inline void QsSequential(SortType array[], const IndexType size){
        QsSequentialStep(array, 0, size-1);
    }

    /** The openmp quick sort */
    static inline void QsOmp(SortType array[], const IndexType size){
        const int nbTasksRequiere = (omp_get_max_threads() * 5);
        int deep = 0;
        while( (1 << deep) < nbTasksRequiere ) deep += 1;

#pragma omp parallel
        {
#pragma omp master
            {
                QsTask(array, 0, size - 1 , deep);
            }
        }
    }

    /** The openmp quick sort */
    static inline void QsOmp2(SortType array[], const IndexType size){
        IndexType deep = 0;
        while( (1 << deep) < size ) deep += 1;

#pragma omp parallel
        {
#pragma omp master
            {
                QsTask(array, 0, size - 1 , deep);
            }
        }
    }


    /** A task dispatcher */
    static inline void QsTask3(SortType array[], const IndexType left, const IndexType right){
        if(right-left < SortLimite){
            SortByVecBitFull(array+left, right-left+1);
        }
        else{
            const IndexType part = PivotPartition(array, left, right);
                // default(none) has been removed for clang compatibility
                if(part+1 < right){
                    if(right-part+1 > 1000){
                        #pragma omp task firstprivate(array, part, right)
                        QsTask3(array,part+1,right);
                    }
                    else{
                        QsSequentialStep(array,part+1,right);
                    }
                }
                // #pragma omp task default(none) firstprivate(array, part, right, deep, infOrEqual) // not needed
                if(part && left < part-1){
                    if(part-1 - left > 1000){
                        QsTask3(array,left,part - 1);
                    }
                    else{
                        QsSequentialStep(array,left,part - 1);
                    }
                }
        }
    }

    /** The openmp quick sort */
    static inline void QsOmp3(SortType array[], const IndexType size){

#pragma omp parallel
        {
#pragma omp master
            {
                QsTask3(array, 0, size - 1);
            }
        }
    }

};




////////////////////////////////////////////////////////////
/// Init functions
////////////////////////////////////////////////////////////

#include <iostream>
#include <memory>
#include <cstdlib>

template <class NumType>
void assertNotSorted(const NumType array[], const size_t size, const std::string log){
    for(size_t idx = 1 ; idx < size ; ++idx){
        if(array[idx-1] > array[idx]){
            std::cout << "assertNotSorted -- Array is not sorted\n"
                         "assertNotSorted --    - at pos " << idx << "\n"
                          "assertNotSorted --    - log " << log << std::endl;
        }
    }
}

template <class NumType>
void assertNotPartitioned(const NumType array[], const size_t size, const NumType pivot,
                          const size_t limite, const std::string log){
    for(size_t idx = 0 ; idx < limite ; ++idx){
        if(array[idx] > pivot){
            std::cout << "assertNotPartitioned -- Array is not partitioned\n"
                         "assertNotPartitioned --    - at pos " << idx << "\n"
                         "assertNotPartitioned --    - log " << log << std::endl;
        }
    }
    for(size_t idx = limite ; idx < size ; ++idx){
        if(array[idx] <= pivot){
            std::cout << "assertNotPartitioned -- Array is not partitioned\n"
                         "assertNotPartitioned --    - at pos " << idx << "\n"
                         "assertNotPartitioned --    - log " << log << std::endl;
        }
    }
}

template <class NumType>
void assertNotEqual(const NumType array1[], const NumType array2[],
                    const int size, const std::string log){
    for(int idx = 0 ; idx < size ; ++idx){
        if(array1[idx] != array2[idx]){
            std::cout << "assertNotEqual -- Array is not equal\n"
                         "assertNotEqual --    - at pos " << idx << "\n"
                                                                    "assertNotEqual --    - array1 " << array1[idx] << "\n"
                                                                                                                       "assertNotEqual --    - array2 " << array2[idx] << "\n"
                                                                                                                                                                          "assertNotEqual --    - log " << log << std::endl;
        }
    }
}

template <class NumType>
void createRandVec(NumType array[], const size_t size){
    for(size_t idx = 0 ; idx < size ; ++idx){
        array[idx] = NumType(drand48()*double(size));
    }
}

// To ensure vec is used and to kill extra optimization
template <class NumType>
void useVec(NumType array[], const size_t size){
    double all = 0;
    for(size_t idx = 0 ; idx < size ; ++idx){
        all += double(array[idx]) * 0.000000000001;
    }
    // This will never happen!
    if(all == std::numeric_limits<double>::max()){
        std::cout << "The impossible happens!!" << std::endl;
        exit(99);
    }
}

#include <cstring>

template <class NumType, class SizeType = size_t>
class Checker{
    std::unique_ptr<NumType[]> cpArray;
    NumType* ptrArray;
    SizeType size;
public:
    Checker(const NumType sourceArray[],
            NumType toCheck[],
            const SizeType inSinze)
        : ptrArray(toCheck), size(inSinze){
        cpArray.reset(new NumType[size]);
        memcpy(cpArray.get(), sourceArray, size*sizeof(NumType));
    }

    ~Checker(){
        std::sort(ptrArray, ptrArray+size);
        std::sort(cpArray.get(), cpArray.get()+size);
        assertNotEqual(cpArray.get(), ptrArray, size, "Checker");
    }
};

////////////////////////////////////////////////////////////
/// Testing functions
////////////////////////////////////////////////////////////

void testSortVec_Core_Equal(const double toSort[8], const double sorted[8]){
    double res[8];

    _mm512_storeu_pd(res, SortVec(_mm512_loadu_pd(toSort)));
    assertNotSorted(res, 8, "testSortVec_Core_Equal");
    assertNotEqual(res, sorted, 8, "testSortVec_Core_Equal");

    _mm512_storeu_pd(res, SortVecWithTest(_mm512_loadu_pd(toSort)));
    assertNotSorted(res, 8, "testSortVec_Core_Equal");
    assertNotEqual(res, sorted, 8, "testSortVec_Core_Equal");

    _mm512_storeu_pd(res, SortVecBitFull(_mm512_loadu_pd(toSort)));
    assertNotSorted(res, 8, "testSortVec_Core_Equal");
    assertNotEqual(res, sorted, 8, "testSortVec_Core_Equal");
}


void testSortVec_Core_Equal(const int toSort[16], const int sorted[16]){
    int res[16];

    _mm512_storeu_si512(res, SortVec(_mm512_loadu_si512(toSort)));
    assertNotSorted(res, 16, "testSortVec_Core_Equal");
    assertNotEqual(res, sorted, 16, "testSortVec_Core_Equal");

    _mm512_storeu_si512(res, SortVecWithTest(_mm512_loadu_si512(toSort)));
    assertNotSorted(res, 16, "testSortVec_Core_Equal");
    assertNotEqual(res, sorted, 16, "testSortVec_Core_Equal");

    _mm512_storeu_si512(res, SortVecBitFull(_mm512_loadu_si512(toSort)));
    assertNotSorted(res, 16, "testSortVec_Core_Equal");
    assertNotEqual(res, sorted, 16, "testSortVec_Core_Equal");
}


void testSortVec(){
    std::cout << "Start testSortVec double...\n";
    {
        {
            double vecTest[8] = { 1., 2., 3., 4., 5., 6., 7., 8.};
            double vecRes[8] = { 1., 2., 3., 4., 5., 6., 7., 8.};
            testSortVec_Core_Equal(vecTest, vecRes);
        }
        {
            double vecTest[8] = { 8., 7., 6., 5., 4., 3., 2., 1};
            double vecRes[8] = { 1., 2., 3., 4., 5., 6., 7., 8.};
            testSortVec_Core_Equal(vecTest, vecRes);
        }
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[8];
            createRandVec(vecTest, 8);

            double res[8];
            {
                Checker<double> checker(vecTest, res, 8);
                _mm512_storeu_pd(res, SortVec(_mm512_loadu_pd(vecTest)));
                assertNotSorted(res, 8, "testSortVec_Core_Equal");
            }
            {
                Checker<double> checker(vecTest, res, 8);
                _mm512_storeu_pd(res, SortVecWithTest(_mm512_loadu_pd(vecTest)));
                assertNotSorted(res, 8, "testSortVec_Core_Equal");
            }
            {
                Checker<double> checker(vecTest, res, 8);
                _mm512_storeu_pd(res, SortVecBitFull(_mm512_loadu_pd(vecTest)));
                assertNotSorted(res, 8, "testSortVec_Core_Equal");
            }
        }
    }
    std::cout << "Start testSortVec int...\n";
    {
        {
            int vecTest[16] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
            int vecRes[16] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
            testSortVec_Core_Equal(vecTest, vecRes);
        }
        {
            int vecTest[16] = { 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
            int vecRes[16] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
            testSortVec_Core_Equal(vecTest, vecRes);
        }
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[16];
            createRandVec(vecTest, 16);

            int res[16];
            {
                Checker<int> checker(vecTest, res, 16);
                _mm512_storeu_si512(res, SortVec(_mm512_loadu_si512(vecTest)));
                assertNotSorted(res, 16, "testSortVec_Core_Equal");
            }
            {
                Checker<int> checker(vecTest, res, 16);
                _mm512_storeu_si512(res, SortVecWithTest(_mm512_loadu_si512(vecTest)));
                assertNotSorted(res, 16, "testSortVec_Core_Equal");
            }
            {
                Checker<int> checker(vecTest, res, 16);
                _mm512_storeu_si512(res, SortVecBitFull(_mm512_loadu_si512(vecTest)));
                assertNotSorted(res, 16, "testSortVec_Core_Equal");
            }
        }
    }
}

void testSort2Vec_Core_Equal(const double toSort[16], const double sorted[16]){
    double res[16];

    __m512d vec1 = _mm512_loadu_pd(toSort);
    __m512d vec2 = _mm512_loadu_pd(toSort+8);
    Sort2Vec(vec1, vec2);
    _mm512_storeu_pd(res, vec1);
    _mm512_storeu_pd(res+8, vec2);
    assertNotSorted(res, 16, "testSort2Vec_Core_Equal");
    assertNotEqual(res, sorted, 16, "testSort2Vec_Core_Equal");

    vec1 = _mm512_loadu_pd(toSort);
    vec2 = _mm512_loadu_pd(toSort+8);
    Sort2VecWithTest(vec1, vec2);
    _mm512_storeu_pd(res, vec1);
    _mm512_storeu_pd(res+8, vec2);
    assertNotSorted(res, 16, "testSort2Vec_Core_Equal");
    assertNotEqual(res, sorted, 16, "testSort2Vec_Core_Equal");


    vec1 = _mm512_loadu_pd(toSort);
    vec2 = _mm512_loadu_pd(toSort+8);
    Sort2VecBitFull(vec1, vec2);
    _mm512_storeu_pd(res, vec1);
    _mm512_storeu_pd(res+8, vec2);
    assertNotSorted(res, 16, "testSort2Vec_Core_Equal");
    assertNotEqual(res, sorted, 16, "testSort2Vec_Core_Equal");
}

void testSort2Vec_Core_Equal(const int toSort[32], const int sorted[32]){
    int res[32];

    __m512i vec1 = _mm512_loadu_si512(toSort);
    __m512i vec2 = _mm512_loadu_si512(toSort+16);
    Sort2Vec(vec1, vec2);
    _mm512_storeu_si512(res, vec1);
    _mm512_storeu_si512(res+16, vec2);
    assertNotSorted(res, 32, "testSort2Vec_Core_Equal");
    assertNotEqual(res, sorted, 32, "testSort2Vec_Core_Equal");

    vec1 = _mm512_loadu_si512(toSort);
    vec2 = _mm512_loadu_si512(toSort+16);
    Sort2VecWithTest(vec1, vec2);
    _mm512_storeu_si512(res, vec1);
    _mm512_storeu_si512(res+16, vec2);
    assertNotSorted(res, 32, "testSort2Vec_Core_Equal");
    assertNotEqual(res, sorted, 32, "testSort2Vec_Core_Equal");

    vec1 = _mm512_loadu_si512(toSort);
    vec2 = _mm512_loadu_si512(toSort+16);
    Sort2VecBitFull(vec1, vec2);
    _mm512_storeu_si512(res, vec1);
    _mm512_storeu_si512(res+16, vec2);
    assertNotSorted(res, 32, "testSort2Vec_Core_Equal");
    assertNotEqual(res, sorted, 32, "testSort2Vec_Core_Equal");
}


void testSort2Vec(){
    std::cout << "Start testSort2Vec double...\n";
    {
        {
            double vecTest[16] = { 1., 1., 2., 2., 3., 3.,4., 4.,
                                   5., 5., 6., 6., 7., 7., 8., 8.};
            double vecRes[16] = { 1., 1., 2., 2., 3., 3.,4., 4.,
                                  5., 5., 6., 6., 7., 7., 8., 8.};
            testSort2Vec_Core_Equal(vecTest, vecRes);
        }
        {
            double vecTest[16] = { 8., 8., 7., 7., 6., 6., 5., 5.,
                                   4., 4., 3., 3., 2., 2., 1., 1.};
            double vecRes[16] = { 1., 1., 2., 2., 3., 3.,4., 4.,
                                  5., 5., 6., 6., 7., 7., 8., 8.};
            testSort2Vec_Core_Equal(vecTest, vecRes);
        }
        {
            double vecTest[16] = { 5., 5., 6., 6., 7., 7., 8., 8.,
                                   1., 1., 2., 2., 3., 3.,4., 4.};
            double vecRes[16] = { 1., 1., 2., 2., 3., 3.,4., 4.,
                                  5., 5., 6., 6., 7., 7., 8., 8.};
            testSort2Vec_Core_Equal(vecTest, vecRes);
        }
        {
            double vecTest[16] = { 4., 4., 3., 3., 2., 2., 1., 1.,
                                   8., 8., 7., 7., 6., 6., 5., 5.};
            double vecRes[16] = { 1., 1., 2., 2., 3., 3.,4., 4.,
                                  5., 5., 6., 6., 7., 7., 8., 8.};
            testSort2Vec_Core_Equal(vecTest, vecRes);
        }
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[16];
            createRandVec(vecTest, 16);

            {
                Checker<double> checker(vecTest, vecTest, 16);
                Sort2Vec(vecTest, vecTest+8);
                assertNotSorted(vecTest, 16, "testSortVec_Core_Equal");
            }
            {
                createRandVec(vecTest, 16);
                Checker<double> checker(vecTest, vecTest, 16);
                Sort2VecWithTest(vecTest, vecTest+8);
                assertNotSorted(vecTest, 16, "testSortVec_Core_Equal");
            }
            {
                createRandVec(vecTest, 16);
                Checker<double> checker(vecTest, vecTest, 16);
                Sort2VecBitFull(vecTest, vecTest+8);
                assertNotSorted(vecTest, 16, "testSortVec_Core_Equal");
            }
        }
    }
    std::cout << "Start testSort2Vec int...\n";
    {
        {
            int vecTest[32] = { 1, 1, 2, 2, 3, 3,4, 4,
                                5, 5, 6, 6, 7, 7, 8, 8,
                                9, 9, 10, 10, 11, 11, 12, 12,
                                13, 13, 14, 14, 15, 15, 16, 16};
            int vecRes[32] = { 1, 1, 2, 2, 3, 3,4, 4,
                               5, 5, 6, 6, 7, 7, 8, 8,
                               9, 9, 10, 10, 11, 11, 12, 12,
                               13, 13, 14, 14, 15, 15, 16, 16};
            testSort2Vec_Core_Equal(vecTest, vecRes);
        }
        {
            int vecTest[32] = { 16, 16, 15, 15, 14, 14, 13, 13,
                                12, 12, 11, 11, 10, 10, 9, 9,
                                8, 8, 7, 7, 6, 6, 5, 5,
                                4, 4, 3, 3, 2, 2, 1, 1};
            int vecRes[32] = { 1, 1, 2, 2, 3, 3,4, 4,
                               5, 5, 6, 6, 7, 7, 8, 8,
                               9, 9, 10, 10, 11, 11, 12, 12,
                               13, 13, 14, 14, 15, 15, 16, 16};
            testSort2Vec_Core_Equal(vecTest, vecRes);
        }
        {
            int vecTest[32] = { 13, 13, 14, 14, 15, 15, 16, 16,
                                9, 9, 10, 10, 11, 11, 12, 12,
                                5, 5, 6, 6, 7, 7, 8, 8,
                                1, 1, 2, 2, 3, 3,4, 4};
            int vecRes[32] = { 1, 1, 2, 2, 3, 3,4, 4,
                               5, 5, 6, 6, 7, 7, 8, 8,
                               9, 9, 10, 10, 11, 11, 12, 12,
                               13, 13, 14, 14, 15, 15, 16, 16};
            testSort2Vec_Core_Equal(vecTest, vecRes);
        }
        {
            int vecTest[32] = { 4, 4, 3, 3, 2, 2, 1, 1,
                                8, 8, 7, 7, 6, 6, 5, 5,
                                16, 16, 15, 15, 14, 14, 13, 13,
                                12, 12, 11, 11, 10, 10, 9, 9};
            int vecRes[32] = { 1, 1, 2, 2, 3, 3,4, 4,
                               5, 5, 6, 6, 7, 7, 8, 8,
                               9, 9, 10, 10, 11, 11, 12, 12,
                               13, 13, 14, 14, 15, 15, 16, 16};
            testSort2Vec_Core_Equal(vecTest, vecRes);
        }
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[32];
            createRandVec(vecTest, 32);

            {
                Checker<int> checker(vecTest, vecTest, 32);
                Sort2Vec(vecTest, vecTest+16);
                assertNotSorted(vecTest, 32, "testSortVec_Core_Equal");
            }
            {
                createRandVec(vecTest, 32);
                Checker<int> checker(vecTest, vecTest, 32);
                Sort2VecWithTest(vecTest, vecTest+16);
                assertNotSorted(vecTest, 32, "testSortVec_Core_Equal");
            }
            {
                createRandVec(vecTest, 32);
                Checker<int> checker(vecTest, vecTest, 32);
                Sort2VecBitFull(vecTest, vecTest+16);
                assertNotSorted(vecTest, 32, "testSortVec_Core_Equal");
            }
        }
    }
}


void testSort3Vec(){
    std::cout << "Start testSort3Vec double...\n";
    {
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[24];
            createRandVec(vecTest, 24);

            {
                Checker<double> checker(vecTest, vecTest, 24);
                Sort3Vec(vecTest, vecTest+8, vecTest+16);
                assertNotSorted(vecTest, 24, "testSortVec_Core_Equal");
            }

            createRandVec(vecTest, 24);
            {
                Checker<double> checker(vecTest, vecTest, 24);
                Sort3VecBitFull(vecTest, vecTest+8, vecTest+16);
                assertNotSorted(vecTest, 24, "testSortVec_Core_Equal");
            }
        }
    }
    std::cout << "Start testSort3Vec int...\n";
    {
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[48];
            createRandVec(vecTest, 48);

            {
                Checker<int> checker(vecTest, vecTest, 48);
                Sort3Vec(vecTest, vecTest+16, vecTest+32);
                assertNotSorted(vecTest, 48, "testSortVec_Core_Equal");
            }
            createRandVec(vecTest, 48);
            {
                Checker<int> checker(vecTest, vecTest, 48);
                Sort3VecBitFull(vecTest, vecTest+16, vecTest+32);
                assertNotSorted(vecTest, 48, "testSortVec_Core_Equal");
            }
        }
    }
}

void testSort4Vec(){
    std::cout << "Start testSort4Vec double...\n";
    {
        {
            double vecTest[32];
            for(int idx = 31 ; idx >= 0 ; --idx){
                vecTest[idx] = double(idx);
            }
            {
                Checker<double> checker(vecTest, vecTest, 32);
                Sort4Vec(vecTest, vecTest+8, vecTest+16, vecTest+24);
                assertNotSorted(vecTest, 32, "testSortVec_Core_Equal");
            }
            for(int idx = 31 ; idx >= 0 ; --idx){
                vecTest[idx] = double(idx);
            }
//          TODO
//            double vecTest[32] = {1., 1., 1., 1., 2., 2.,  2., 2.,
//                                  1., 1., 1., 1., 1., 1., 1., 1.,
//                                  5., 5., 4., 4., 3.,  3., 1.9, 1.9,
//                                  1.5, 1.5, 1.5, 1.5, 1., 1.,  1., 1.};
            {
                Checker<double> checker(vecTest, vecTest, 32);
                Sort4VecBitFull(vecTest, vecTest+8, vecTest+16, vecTest+24);
                assertNotSorted(vecTest, 32, "testSortVec_Core_Equal");
            }
        }
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[32];
            createRandVec(vecTest, 32);

            {
                Checker<double> checker(vecTest, vecTest, 32);
                Sort4Vec(vecTest, vecTest+8, vecTest+16, vecTest+24);
                assertNotSorted(vecTest, 32, "testSortVec_Core_Equal");
            }
            createRandVec(vecTest, 32);
            {
                Checker<double> checker(vecTest, vecTest, 32);
                Sort4VecBitFull(vecTest, vecTest+8, vecTest+16, vecTest+24);
                assertNotSorted(vecTest, 32, "testSortVec_Core_Equal");
            }
        }
    }
    std::cout << "Start testSort4Vec int...\n";
    {
        {
            int vecTest[64] = {0};
            for(int idx = 63 ; idx >= 0 ; --idx){
                vecTest[idx] = int(idx);
            }
            {
                Checker<int> checker(vecTest, vecTest, 64);
                Sort4Vec(vecTest, vecTest+16, vecTest+32, vecTest+48);
                assertNotSorted(vecTest, 64, "testSortVec_Core_Equal");
            }
            for(int idx = 63 ; idx >= 0 ; --idx){
                vecTest[idx] = int(idx);
            }
            {
                Checker<int> checker(vecTest, vecTest, 64);
                Sort4VecBitFull(vecTest, vecTest+16, vecTest+32, vecTest+48);
                assertNotSorted(vecTest, 64, "testSortVec_Core_Equal");
            }
        }
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[64];
            createRandVec(vecTest, 64);
            {
                Checker<int> checker(vecTest, vecTest, 64);
                Sort4Vec(vecTest, vecTest+16, vecTest+32, vecTest+48);
                assertNotSorted(vecTest, 64, "testSortVec_Core_Equal");
            }
            createRandVec(vecTest, 64);
            {
                Checker<int> checker(vecTest, vecTest, 64);
                Sort4VecBitFull(vecTest, vecTest+16, vecTest+32, vecTest+48);
                assertNotSorted(vecTest, 64, "testSortVec_Core_Equal");
            }
        }
    }
}

void testSort5Vec(){
    std::cout << "Start testSort5Vec double...\n";
    {
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[5*8];

            createRandVec(vecTest, 5*8);

            Checker<double> checker(vecTest, vecTest, 5*8);
            Sort5VecBitFull(vecTest, vecTest+8, vecTest+8*2, vecTest+8*3, vecTest+8*4);
            assertNotSorted(vecTest, 5*8, "testSortVec_Core_Equal");
        }
    }
    std::cout << "Start testSort5Vec int...\n";
    {
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[5*16];

            createRandVec(vecTest, 5*16);

            Checker<int> checker(vecTest, vecTest, 5*16);
            Sort5VecBitFull(vecTest, vecTest+16, vecTest+16*2, vecTest+16*3, vecTest+16*4);
            assertNotSorted(vecTest, 5*16, "testSortVec_Core_Equal");
        }
    }
}


void testSort6Vec(){
    std::cout << "Start testSort6Vec double...\n";
    {
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[6*8];

            createRandVec(vecTest, 6*8);

            Checker<double> checker(vecTest, vecTest, 6*8);
            Sort6VecBitFull(vecTest, vecTest+8, vecTest+8*2, vecTest+8*3, vecTest+8*4, vecTest+8*5);
            assertNotSorted(vecTest, 6*8, "testSortVec_Core_Equal");
        }
    }
    std::cout << "Start testSort6Vec int...\n";
    {
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[6*16];

            createRandVec(vecTest, 6*16);

            Checker<int> checker(vecTest, vecTest, 6*16);
            Sort6VecBitFull(vecTest, vecTest+16, vecTest+16*2, vecTest+16*3, vecTest+16*4, vecTest+16*5);
            assertNotSorted(vecTest, 6*16, "testSortVec_Core_Equal");
        }
    }
}


void testSort7Vec(){
    std::cout << "Start testSort7Vec double...\n";
    {
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[7*8];

            createRandVec(vecTest, 7*8);

            Checker<double> checker(vecTest, vecTest, 7*8);
            Sort7VecBitFull(vecTest, vecTest+8, vecTest+8*2, vecTest+8*3, vecTest+8*4, vecTest+8*5, vecTest+8*6);
            assertNotSorted(vecTest, 7*8, "testSortVec_Core_Equal");
        }
    }
    std::cout << "Start testSort7Vec int...\n";
    {
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[7*16];

            createRandVec(vecTest, 7*16);

            Checker<int> checker(vecTest, vecTest, 7*16);
            Sort7VecBitFull(vecTest, vecTest+16, vecTest+16*2, vecTest+16*3, vecTest+16*4, vecTest+16*5, vecTest+16*6);
            assertNotSorted(vecTest, 7*16, "testSortVec_Core_Equal");
        }
    }
}

void testSort8Vec(){
    std::cout << "Start testSort8Vec double...\n";
    {
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[8*8];

            createRandVec(vecTest, 8*8);

            Checker<double> checker(vecTest, vecTest, 8*8);
            Sort8VecBitFull(vecTest, vecTest+8, vecTest+8*2, vecTest+8*3, vecTest+8*4, vecTest+8*5, vecTest+8*6, vecTest+8*7);
            assertNotSorted(vecTest, 8*8, "testSortVec_Core_Equal");
        }
    }
    std::cout << "Start testSort8Vec int...\n";
    {
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[8*16];

            createRandVec(vecTest, 8*16);

            Checker<int> checker(vecTest, vecTest, 8*16);
            Sort8VecBitFull(vecTest, vecTest+16, vecTest+16*2, vecTest+16*3, vecTest+16*4, vecTest+16*5, vecTest+16*6, vecTest+16*7);
            assertNotSorted(vecTest, 8*16, "testSortVec_Core_Equal");
        }
    }
}

void testSort9Vec(){
    const int nbVecs = 9;
    std::cout << "Start testSort9Vec double...\n";
    {
        const int sizeVec = 8;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
            Sort9VecBitFull(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                             vecTest+sizeVec*8);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
    }
    std::cout << "Start testSort9Vec int...\n";
    {
        const int sizeVec = 16;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
            Sort9VecBitFull(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                            vecTest+sizeVec*8);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
    }
}

void testSort10Vec(){
    const int nbVecs = 10;
    std::cout << "Start testSort10Vec double...\n";
    {
        const int sizeVec = 8;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
            Sort10VecBitFull(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                             vecTest+sizeVec*8, vecTest+sizeVec*9);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
    }
    std::cout << "Start testSort10Vec int...\n";
    {
        const int sizeVec = 16;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
            Sort10VecBitFull(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                            vecTest+sizeVec*8, vecTest+sizeVec*9);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
    }
}

void testSort11Vec(){
    const int nbVecs = 11;
    std::cout << "Start testSort11Vec double...\n";
    {
        const int sizeVec = 8;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
            Sort11VecBitFull(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                             vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
    }
    std::cout << "Start testSort11Vec int...\n";
    {
        const int sizeVec = 16;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
            Sort11VecBitFull(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                            vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
    }
}

void testSort12Vec(){
    const int nbVecs = 12;
    std::cout << "Start testSort12Vec double...\n";
    {
        const int sizeVec = 8;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
            Sort12VecBitFull(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                             vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
    }
    std::cout << "Start testSort12Vec int...\n";
    {
        const int sizeVec = 16;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
            Sort12VecBitFull(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                            vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
    }
}

void testSort13Vec(){
    const int nbVecs = 13;
    std::cout << "Start testSort13Vec double...\n";
    {
        const int sizeVec = 8;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
            Sort13VecBitFull(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                             vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
    }
    std::cout << "Start testSort13Vec int...\n";
    {
        const int sizeVec = 16;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
            Sort13VecBitFull(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                            vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
    }
}

void testSort14Vec(){
    const int nbVecs = 14;
    std::cout << "Start testSort14Vec double...\n";
    {
        const int sizeVec = 8;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
            Sort14VecBitFull(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                             vecTest+sizeVec*sizeVec, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12, vecTest+sizeVec*13);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
    }
    std::cout << "Start testSort16Vec int...\n";
    {
        const int sizeVec = 16;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
            Sort14VecBitFull(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                            vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12, vecTest+sizeVec*13);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
    }
}

void testSort15Vec(){
    const int nbVecs = 15;
    std::cout << "Start testSort15Vec double...\n";
    {
        const int sizeVec = 8;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
            Sort15VecBitFull(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                             vecTest+sizeVec*sizeVec, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12, vecTest+sizeVec*13, vecTest+sizeVec*14);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
    }
    std::cout << "Start testSort16Vec int...\n";
    {
        const int sizeVec = 16;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
            Sort15VecBitFull(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                            vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12, vecTest+sizeVec*13, vecTest+sizeVec*14);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
    }
}


void testSort16Vec(){
    const int nbVecs = 16;
    std::cout << "Start testSort16Vec double...\n";
    {
        const int sizeVec = 8;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
            Sort16VecBitFull(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                             vecTest+sizeVec*sizeVec, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12, vecTest+sizeVec*13, vecTest+sizeVec*14, vecTest+sizeVec*15);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
    }
    std::cout << "Start testSort16Vec int...\n";
    {
        const int sizeVec = 16;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
            Sort16VecBitFull(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                            vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12, vecTest+sizeVec*13, vecTest+sizeVec*14, vecTest+sizeVec*15);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
    }
}

void testMerge2Vec_Core_Equal(const double toSort[16], const double sorted[16]){
    double res[16];

    __m512d vec1 = _mm512_loadu_pd(toSort);
    __m512d vec2 = _mm512_loadu_pd(toSort+8);
    Merge2Vec(vec1, vec2);
    _mm512_storeu_pd(res, vec1);
    _mm512_storeu_pd(res+8, vec2);
    assertNotSorted(res, 16, "testMerge2Vec_Core_Equal");
    assertNotEqual(res, sorted, 16, "testMerge2Vec_Core_Equal");
}

void testMerge2Vec_Core_Equal(const int toSort[32], const int sorted[32]){
    int res[32];

    __m512i vec1 = _mm512_loadu_si512(toSort);
    __m512i vec2 = _mm512_loadu_si512(toSort+16);
    Merge2Vec(vec1, vec2);
    _mm512_storeu_si512(res, vec1);
    _mm512_storeu_si512(res+16, vec2);
    assertNotSorted(res, 32, "testMerge2Vec_Core_Equal");
    assertNotEqual(res, sorted, 32, "testMerge2Vec_Core_Equal");
}


void testMerge2Vec(){
    std::cout << "Start testSort2Vec double...\n";
    {
        {
            double vecTest[16] = { 1., 1., 2., 2., 3., 3.,4., 4.,
                                   5., 5., 6., 6., 7., 7., 8., 8.};
            double vecRes[16] = { 1., 1., 2., 2., 3., 3.,4., 4.,
                                  5., 5., 6., 6., 7., 7., 8., 8.};
            testMerge2Vec_Core_Equal(vecTest, vecRes);
        }
        {
            double vecTest[16] = { 5., 5., 6., 6., 7., 7., 8., 8.,
                                   1., 1., 2., 2., 3., 3.,4., 4.};
            double vecRes[16] = { 1., 1., 2., 2., 3., 3.,4., 4.,
                                  5., 5., 6., 6., 7., 7., 8., 8.};
            testMerge2Vec_Core_Equal(vecTest, vecRes);
        }
        {
            double vecTest[16] = { 3., 3.,4., 4., 7., 7., 8., 8.,
                                   1., 1., 2., 2., 5., 5., 6., 6.};
            double vecRes[16] = { 1., 1., 2., 2., 3., 3.,4., 4.,
                                  5., 5., 6., 6., 7., 7., 8., 8.};
            testMerge2Vec_Core_Equal(vecTest, vecRes);
        }
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[16];
            createRandVec(vecTest, 8);
            createRandVec(vecTest + 8, 8);

            {
                SortVec(vecTest);
                assertNotSorted(vecTest, 8, "testSortVec_Core_Equal");
            }
            {
                SortVec(vecTest+8);
                assertNotSorted(vecTest + 8, 8, "testSortVec_Core_Equal");
            }
            {
                Checker<double> checker(vecTest, vecTest, 16);
                Merge2Vec(vecTest, vecTest + 8);
                assertNotSorted(vecTest, 16, "testSortVec_Core_Equal");
            }
        }
    }
    std::cout << "Start testSort2Vec int...\n";
    {
        {
            int vecTest[32] = { 1, 1, 2, 2, 3, 3,4, 4,
                                5, 5, 6, 6, 7, 7, 8, 8,
                                9, 9, 10, 10, 11, 11, 12, 12,
                                13, 13, 14, 14, 15, 15, 16, 16};
            int vecRes[32] = { 1, 1, 2, 2, 3, 3,4, 4,
                               5, 5, 6, 6, 7, 7, 8, 8,
                               9, 9, 10, 10, 11, 11, 12, 12,
                               13, 13, 14, 14, 15, 15, 16, 16};
            testMerge2Vec_Core_Equal(vecTest, vecRes);
        }
        {
            int vecTest[32] = {
                9, 9, 10, 10, 11, 11, 12, 12,
                13, 13, 14, 14, 15, 15, 16, 16,
                1, 1, 2, 2, 3, 3,4, 4,
                                5, 5, 6, 6, 7, 7, 8, 8};
            int vecRes[32] = { 1, 1, 2, 2, 3, 3,4, 4,
                               5, 5, 6, 6, 7, 7, 8, 8,
                               9, 9, 10, 10, 11, 11, 12, 12,
                               13, 13, 14, 14, 15, 15, 16, 16};
            testMerge2Vec_Core_Equal(vecTest, vecRes);
        }
        {
            int vecTest[32] = { 1, 1, 2, 2, 3, 3,4, 4,
                                13, 13, 14, 14, 15, 15, 16, 16,
                                5, 5, 6, 6, 7, 7, 8, 8,
                                9, 9, 10, 10, 11, 11, 12, 12};
            int vecRes[32] = { 1, 1, 2, 2, 3, 3,4, 4,
                               5, 5, 6, 6, 7, 7, 8, 8,
                               9, 9, 10, 10, 11, 11, 12, 12,
                               13, 13, 14, 14, 15, 15, 16, 16};
            testMerge2Vec_Core_Equal(vecTest, vecRes);
        }
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[32];
            createRandVec(vecTest, 16);
            createRandVec(vecTest + 16, 16);

            SortVec(vecTest);
            assertNotSorted(vecTest, 16, "testSortVec_Core_Equal");
            SortVec(vecTest+16);
            assertNotSorted(vecTest + 16, 16, "testSortVec_Core_Equal");

            Checker<int> checker(vecTest, vecTest, 32);
            Merge2Vec(vecTest, vecTest + 16);
            assertNotSorted(vecTest, 32, "testSortVec_Core_Equal");
        }
    }
}


template <class NumType>
void testBitonic(){
    std::cout << "Start testBitonic...\n";
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        BitonicSort<NumType, size_t>::BsSequential(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        BitonicSort<NumType, size_t>::BsOmpV1(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        BitonicSort<NumType, size_t>::BsOmpV2(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
}

template <class NumType>
void testBitonicV2(){
    std::cout << "Start testBitonicV2...\n";
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        BitonicSortV2<NumType, size_t>::BsSequential(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        BitonicSortV2<NumType, size_t>::BsSequentialV2(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
    for(size_t idx = 1 ; idx < 100; idx += 1){
        if( ((idx-1) % 10) == 0) std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        BitonicSortV2<NumType, size_t>::BsSequentialV2(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
    for(size_t idx = 1 ; idx < 20000; idx += 100){
        if( ((idx-1) % 1000) == 0) std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        BitonicSortV2<NumType, size_t>::BsSequentialV2(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
}


template <class NumType>
void testBitonicSortAVX512(){
    std::cout << "Start testBitonicSortAVX512...\n";
    const size_t SizeVec = 64/sizeof(NumType);
    for(size_t idx = SizeVec ; idx <= SizeVec*50; idx += SizeVec){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        BitonicSortAVX512<NumType, size_t>::BsSequentialV2(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
    for(size_t idx = SizeVec ; idx < 20000; idx += SizeVec*50){
        if( ((idx-SizeVec)/SizeVec % 100) == 0) std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        BitonicSortAVX512<NumType, size_t>::BsSequentialV2(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }

    std::cout << "Start testBitonicSortAVX512Test...\n";
    for(size_t idx = SizeVec ; idx <= SizeVec*50; idx += SizeVec){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        BitonicSortAVX512WithTest<NumType, size_t>::BsSequentialV2(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
    for(size_t idx = SizeVec ; idx < 20000; idx += SizeVec*50){
        if( ((idx-SizeVec)/SizeVec % 100) == 0) std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        BitonicSortAVX512WithTest<NumType, size_t>::BsSequentialV2(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }

    std::cout << "Start BitonicSortAVX512WithMerge...\n";
    for(size_t idx = SizeVec ; idx <= SizeVec*50; idx += SizeVec){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        BitonicSortAVX512WithMerge<NumType, size_t>::BsSequentialV2(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
    for(size_t idx = SizeVec ; idx < 20000; idx += SizeVec*50){
        if( ((idx-SizeVec)/SizeVec % 100) == 0) std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        BitonicSortAVX512WithMerge<NumType, size_t>::BsSequentialV2(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
}

template <class NumType>
void testBitonicSortAVX512V2(){
    std::cout << "Start testBitonicSortAVX512V2...\n";
    const size_t SizeVec = 64/sizeof(NumType);
    for(size_t idx = SizeVec ; idx <= SizeVec*50; idx += SizeVec){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        BitonicSortAVX512V2<NumType, size_t>::BsSequentialV2(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
    for(size_t idx = SizeVec ; idx < 20000; idx += SizeVec*50){
        if( ((idx-SizeVec)/SizeVec % 100) == 0) std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        BitonicSortAVX512V2<NumType, size_t>::BsSequentialV2(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
}


template <class NumType>
void testHeapsortSort512(){
    std::cout << "Start testHeapsortSort512...\n";
    const size_t SizeVec = 64/sizeof(NumType);
    for(size_t idx = SizeVec ; idx <= SizeVec*50; idx += SizeVec){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        HeapSort512<NumType>::HeapSort(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
    for(size_t idx = SizeVec ; idx < 20000; idx += SizeVec*50){
        if( ((idx-SizeVec)/SizeVec % 100) == 0) std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        HeapSort512<NumType>::HeapSort(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
    for(size_t idx = 1 ; idx < 200; idx++){
        if( ((idx-SizeVec)/SizeVec % 10) == 0) std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        HeapSort512<NumType>::HeapSortNotMultiple(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
}

template <class NumType>
void testHeapsortSort(){
    std::cout << "Start testHeapsortSort...\n";
    const size_t SizeVec = 64/sizeof(NumType);
    for(size_t idx = SizeVec ; idx <= SizeVec*50; idx += SizeVec){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        HeapSort<NumType>::sort(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
    for(size_t idx = SizeVec ; idx < 20000; idx += SizeVec*50){
        if( ((idx-SizeVec)/SizeVec % 100) == 0) std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        HeapSort<NumType>::sort(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
}

template <class NumType>
void testQs(){
    std::cout << "Start testQs...\n";
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        FQuickSort<NumType,size_t>::QsSequential(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        FQuickSort<NumType,size_t>::QsOmp(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
}

template <class NumType>
void testNewQs(){
    std::cout << "Start testNewQs...\n";
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        NewQuickSort<NumType,size_t>::QsSequential(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
    /*for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        NewQuickSort<NumType,size_t>::QsOmp(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }*/
}

template <class NumType>
void testNewQs512(){
    std::cout << "Start testNewQs512...\n";
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        NewQuickSort512<NumType,size_t>::QsSequential(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        NewQuickSort512<NumType,size_t>::QsOmp(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }

    std::cout << "Start NewQuickSort512V2...\n";
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        NewQuickSort512V2<NumType,size_t>::QsSequential(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        NewQuickSort512V2<NumType,size_t>::QsOmp(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }

    std::cout << "Start NewQuickSort512V3...\n";
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        NewQuickSort512V3<NumType,size_t>::QsSequential(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        NewQuickSort512V3<NumType,size_t>::QsOmp(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }

    std::cout << "Start NewQuickSort512V4...\n";
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        NewQuickSort512V4<NumType,size_t>::QsSequential(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        NewQuickSort512V4<NumType,size_t>::QsOmp(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }

    std::cout << "Start NewQuickSort512V5...\n";
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        NewQuickSort512V5<NumType,size_t>::QsSequential(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        NewQuickSort512V5<NumType,size_t>::QsOmp(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
}

template <class NumType>
void testCppSort(){
    std::cout << "Start testCppSort...\n";
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        std::sort(&array[0], &array[idx], [&](const NumType& v1, const NumType& v2){
            return v1 < v2;
        });
        assertNotSorted(array.get(), idx, "");
    }
}

template <class NumType>
void testSmallVecSort(){
    std::cout << "Start testSmallVecSort...\n";
    {
        const int SizeVec = 64/sizeof(NumType);
        const int MaxSizeAllVec = SizeVec * 4;
        for(size_t idx = 1 ; idx <= MaxSizeAllVec; idx++){
            std::cout << "   " << idx << std::endl;
            std::unique_ptr<NumType[]> array(new NumType[idx]);
            for(int idxTest = 0 ; idxTest < 100 ; ++idxTest){
                createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
                SortByVec(array.get(), idx);
                assertNotSorted(array.get(), idx, "");
            }
        }
    }
    std::cout << "Start testSmallVecSort bitfull...\n";
    {
        const int SizeVec = 64/sizeof(NumType);
        const int MaxSizeAllVec = SizeVec * 16;
        for(size_t idx = 1 ; idx <= MaxSizeAllVec; idx++){
            std::cout << "   " << idx << std::endl;
            std::unique_ptr<NumType[]> array(new NumType[idx]);
            for(int idxTest = 0 ; idxTest < 100 ; ++idxTest){
                createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
                SortByVecBitFull(array.get(), idx);
                assertNotSorted(array.get(), idx, "");
            }
        }
    }
}

template <class NumType>
void testCppPartition(){
    std::cout << "Start testCppPartition...\n";
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        const NumType pivot = NumType(idx/2);
        NumType* limitePtr = std::partition(&array[0], &array[idx], [&](const NumType& v){
            return v <= pivot;
        });
        assertNotPartitioned(array.get(), idx, pivot, size_t(limitePtr-array.get()), "");
    }
    for(size_t idx = 1 ; idx <= 1000; ++idx){
        if(idx%100 == 0) std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        const NumType pivot = NumType(idx/2);
        NumType* limitePtr = std::partition(&array[0], &array[idx], [&](const NumType& v){
            return v <= pivot;
        });
        assertNotPartitioned(array.get(), idx, pivot, size_t(limitePtr-array.get()), "");
    }
}

template <class NumType>
void testQsPartition(){
    std::cout << "Start testQsPartition...\n";
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        const NumType pivot = NumType(idx/2);
        size_t limite = FQuickSort<NumType,size_t>::QsPartition(&array[0], 0, idx-1, pivot);
        assertNotPartitioned(array.get(), idx, pivot, limite, "");
    }
    for(size_t idx = 1 ; idx <= 1000; ++idx){
        if(idx%100 == 0) std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        const NumType pivot = NumType(idx/2);
        size_t limite = FQuickSort<NumType,size_t>::QsPartition(&array[0], 0, idx-1, pivot);
        assertNotPartitioned(array.get(), idx, pivot, limite, "");
    }
}

template <class NumType>
void testNewPartition(){
    std::cout << "Start testNewPartition...\n";
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        const NumType pivot = NumType(idx/2);
        size_t limite = NewQuickSort<NumType,size_t>::Partition(&array[0], 0, idx-1, pivot);
        assertNotPartitioned(array.get(), idx, pivot, limite, "");
    }
    for(size_t idx = 1 ; idx <= 1000; ++idx){
        if(idx%100 == 0) std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        const NumType pivot = NumType(idx/2);
        size_t limite = NewQuickSort<NumType,size_t>::Partition(&array[0], 0, idx-1, pivot);
        assertNotPartitioned(array.get(), idx, pivot, limite, "");
    }
}

template <class NumType>
void testNewPartition512(){
    std::cout << "Start testNewPartition512...\n";
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        const NumType pivot = NumType(idx/2);
        size_t limite = NewQuickSort512<NumType,size_t>::Partition(&array[0], 0, idx-1, pivot);
        assertNotPartitioned(array.get(), idx, pivot, limite, "");
    }
    for(size_t idx = 1 ; idx <= 1000; ++idx){
        if(idx%100 == 0) std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        const NumType pivot = NumType(idx/2);
        size_t limite = NewQuickSort512<NumType,size_t>::Partition(&array[0], 0, idx-1, pivot);
        assertNotPartitioned(array.get(), idx, pivot, limite, "");
    }
}

template <class NumType>
void testNewPartition512V2(){
    std::cout << "Start NewQuickSort512V2...\n";
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        const NumType pivot = NumType(idx/2);
        size_t limite = NewQuickSort512V2<NumType,size_t>::Partition(&array[0], 0, idx-1, pivot);
        assertNotPartitioned(array.get(), idx, pivot, limite, "");
    }
    for(size_t idx = 1 ; idx <= 1000; ++idx){
        if(idx%100 == 0) std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        const NumType pivot = NumType(idx/2);
        size_t limite = NewQuickSort512V2<NumType,size_t>::Partition(&array[0], 0, idx-1, pivot);
        assertNotPartitioned(array.get(), idx, pivot, limite, "");
    }
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);

        for(size_t idxVal = 0 ; idxVal < idx ; ++idxVal){
            array[idxVal] = NumType(idx);
        }

        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        const NumType pivot = NumType(idx/2);
        size_t limite = NewQuickSort512V2<NumType,size_t>::Partition(&array[0], 0, idx-1, pivot);
        assertNotPartitioned(array.get(), idx, pivot, limite, "");
    }
}

void testPopcount(){
    std::cout << "Start testPopcount...\n";
    auto assertFunc = [](const int trueres, const int test, const int val, const std::string& logbuf){
        if(test != trueres){
            std::cout << "testPopcount errror - " << logbuf << "\n";
            std::cout << "testPopcount errror - for val " << val << "\n";
            std::cout << "testPopcount errror - should be " << trueres << " is " << test << "\n";
        }
    };

    assertFunc(0, popcount(__mmask16(0)), 0, "__mmask16");
    assertFunc(0, popcount(__mmask8(0)), 0, "__mmask8");

    for(int idx = 0 ; idx < 16 ; ++idx){
        assertFunc(1, popcount(__mmask16(1)), 1<<idx, "__mmask16");
        if(idx < 8) assertFunc(1, popcount(__mmask8(1)), 1<<idx, "__mmask8");
    }

    assertFunc(2, popcount(__mmask16(3)), 3, "__mmask16");
    assertFunc(2, popcount(__mmask8(3)), 3, "__mmask8");

    assertFunc(16, popcount(__mmask16(0xFFFF)), 0xFFFF, "__mmask16");
    assertFunc(8, popcount(__mmask8(0xFF)), 0xFF, "__mmask8");
}


template <class NumType>
void testInsertion(){
    std::cout << "Start testInsertion...\n";
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        InsertionSort<NumType,size_t>(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        InsertionSort<NumType,size_t>(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
}

void testAll(){
    testPopcount();

    testSortVec();
    testSort2Vec();
    testSort3Vec();
    testSort4Vec();
    testSort5Vec();
    testSort6Vec();
    testSort7Vec();
    testSort8Vec();
    testMerge2Vec();


    testSort9Vec();
    testSort10Vec();
    testSort11Vec();
    testSort12Vec();
    testSort13Vec();
    testSort14Vec();
    testSort15Vec();
    testSort16Vec();

    testSmallVecSort<int>();
    testSmallVecSort<double>();
    testBitonic<double>();
    testBitonicV2<double>();
    testBitonicSortAVX512<double>();
    testBitonicSortAVX512V2<double>();
    testHeapsortSort<double>();
    testHeapsortSort512<double>();
    testQs<double>();
    testCppSort<double>();
    testNewQs<double>();
    testNewQs512<double>();
    testInsertion<double>();

    testBitonic<int>();
    testBitonicV2<int>();
    testBitonicSortAVX512<int>();
    testBitonicSortAVX512V2<int>();
    testHeapsortSort<int>();
    testHeapsortSort512<int>();
    testQs<int>();
    testCppSort<int>();
    testNewQs<int>();
      testNewQs512<int>();
    testInsertion<int>();

    testCppPartition<int>();
    testQsPartition<int>();
    testNewPartition<int>();
      testNewPartition512<int>();

    testCppPartition<double>();
    testQsPartition<double>();
    testNewPartition<double>();
      testNewPartition512<double>();
      testNewPartition512V2<double>();
}


////////////////////////////////////////////////////////////
/// Timing functions
////////////////////////////////////////////////////////////

#include <fstream>

template <class NumType>
void timeAll(std::ostream& fres, const std::string prefix){
    const size_t MaxSize = 1073741824;//10L*1024L*1024L*1024L;//10*1024*1024*1024;262144*8;//
    const int NbLoops = 5;

    std::unique_ptr<NumType[]> array(new NumType[MaxSize]);

    for(size_t currentSize = 64 ; currentSize <= MaxSize ; currentSize *= 8 ){
        std::cout << "currentSize " << currentSize << std::endl;


        double allTimes[15][3] = {{ std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                            { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                            { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                             { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                             { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                             { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                             { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                             { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                             { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                             { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                                  { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                                  { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                                  { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                                  { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. }};

        for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            std::cout << "  idxLoop " << idxLoop << std::endl;
            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                dtimer timer;
                std::sort(&array[0], &array[currentSize], [&](const NumType& v1, const NumType& v2){
                    return v1 < v2;
                });
                timer.stop();
                std::cout << "    std::sort " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 0;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                dtimer timer;
                FQuickSort<NumType,size_t>::QsSequential(array.get(), currentSize);
                timer.stop();
                std::cout << "    FQuickSort " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 1;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
            /*{
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                dtimer timer;
                BitonicSortV2<NumType,size_t>::BsSequentialV2(array.get(), currentSize);
                timer.stop();
                std::cout << "    bt " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 2;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                dtimer timer;
                BitonicSortAVX512<NumType, size_t>::BsSequentialV2(array.get(), currentSize);
                timer.stop();
                std::cout << "    BitonicSortAVX512 " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 3;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                dtimer timer;
                BitonicSortAVX512WithTest<NumType, size_t>::BsSequentialV2(array.get(), currentSize);
                timer.stop();
                std::cout << "    BitonicSortAVX512WithTest " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 4;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                dtimer timer;
                BitonicSortAVX512WithMerge<NumType, size_t>::BsSequentialV2(array.get(), currentSize);
                timer.stop();
                std::cout << "    BitonicSortAVX512WithMerge " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 5;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                dtimer timer;
                NewQuickSort<NumType, size_t>::QsSequential(array.get(), currentSize);
                timer.stop();
                std::cout << "    NewQuickSort " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 6;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }*/
            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                dtimer timer;
                NewQuickSort512<NumType, size_t>::QsSequential(array.get(), currentSize);
                timer.stop();
                std::cout << "    NewQuickSort512 " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 7;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                dtimer timer;
                NewQuickSort512V2<NumType, size_t>::QsSequential(array.get(), currentSize);
                timer.stop();
                std::cout << "    NewQuickSort512V2 " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 8;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                dtimer timer;
                NewQuickSort512V3<NumType, size_t>::QsSequential(array.get(), currentSize);
                timer.stop();
                std::cout << "    NewQuickSort512V3 " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 9;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
            /*{
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                dtimer timer;
                HeapSort<NumType>::sort(array.get(), currentSize);
                timer.stop();
                std::cout << "    HeapSort " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 10;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }*/
            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                dtimer timer;
                BitonicSortAVX512V2<NumType>::BsSequentialV2(array.get(), currentSize);
                timer.stop();
                std::cout << "    BitonicSortAVX512V2 " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 11;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                dtimer timer;
                NewQuickSort512V4<NumType, size_t>::QsSequential(array.get(), currentSize);
                timer.stop();
                std::cout << "    NewQuickSort512V4 " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 12;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                dtimer timer;
                NewQuickSort512V5<NumType, size_t>::QsSequential(array.get(), currentSize);
                timer.stop();
                std::cout << "    NewQuickSort512V5 " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 13;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
        }

        fres << prefix << currentSize << ",\"stdsort\"," << allTimes[0][0] << "," << allTimes[0][1] << "," << allTimes[0][2] << "\n";
        fres << prefix << currentSize << ",\"qs\"," << allTimes[1][0] << "," << allTimes[1][1] << "," << allTimes[1][2] << "\n";
        /*fres << prefix << currentSize << ",\"bt\"," << allTimes[2][0] << "," << allTimes[2][1] << "," << allTimes[2][2] << "\n";
        fres << prefix << currentSize << ",\"bt512\"," << allTimes[3][0] << "," << allTimes[3][1] << "," << allTimes[3][2] << "\n";
        fres << prefix << currentSize << ",\"bt512wt\"," << allTimes[4][0] << "," << allTimes[4][1] << "," << allTimes[4][2] << "\n";
        fres << prefix << currentSize << ",\"bt512wm\"," << allTimes[5][0] << "," << allTimes[5][1] << "," << allTimes[5][2] << "\n";
        fres << prefix << currentSize << ",\"newqs\"," << allTimes[6][0] << "," << allTimes[6][1] << "," << allTimes[6][2] << "\n";*/
        fres << prefix << currentSize << ",\"newqs512\"," << allTimes[7][0] << "," << allTimes[7][1] << "," << allTimes[7][2] << "\n";
        fres << prefix << currentSize << ",\"newqs512v2\"," << allTimes[8][0] << "," << allTimes[8][1] << "," << allTimes[8][2] << "\n";
        fres << prefix << currentSize << ",\"newqs512v3\"," << allTimes[9][0] << "," << allTimes[9][1] << "," << allTimes[9][2] << "\n";
        //fres << prefix << currentSize << ",\"heapsort\"," << allTimes[10][0] << "," << allTimes[10][1] << "," << allTimes[10][2] << "\n";
        fres << prefix << currentSize << ",\"bt512v2\"," << allTimes[11][0] << "," << allTimes[11][1] << "," << allTimes[11][2] << "\n";
        fres << prefix << currentSize << ",\"newqs512v4\"," << allTimes[12][0] << "," << allTimes[12][1] << "," << allTimes[12][2] << "\n";
        fres << prefix << currentSize << ",\"newqs512v5\"," << allTimes[13][0] << "," << allTimes[13][1] << "," << allTimes[13][2] << "\n";
        fres.flush();
    }

}



template <class NumType>
void timeAllOmp(std::ostream& fres, const std::string prefix){
    const size_t MaxSize = 1073741824;//10L*1024L*1024L*1024L;//10*1024*1024*1024;262144*8;//
    const int NbLoops = 5;

    std::unique_ptr<NumType[]> array(new NumType[MaxSize]);

    for(size_t currentSize = 64 ; currentSize <= MaxSize ; currentSize *= 8 ){
        std::cout << "currentSize " << currentSize << std::endl;


        double allTimes[3][3] = {{ std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                            { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                            { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. }};

        for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            std::cout << "  idxLoop " << idxLoop << std::endl;

            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                dtimer timer;
                NewQuickSort512V5<NumType, size_t>::QsOmp(array.get(), currentSize);
                timer.stop();
                std::cout << "    NewQuickSort512V5 " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 0;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                dtimer timer;
                NewQuickSort512V5<NumType, size_t>::QsOmp2(array.get(), currentSize);
                timer.stop();
                std::cout << "    NewQuickSort512V5 " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 1;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                dtimer timer;
                NewQuickSort512V5<NumType, size_t>::QsOmp3(array.get(), currentSize);
                timer.stop();
                std::cout << "    NewQuickSort512V5 " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 2;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
        }

        fres << prefix << currentSize << ",\"omp1\"," << allTimes[0][0] << "," << allTimes[0][1] << "," << allTimes[0][2] << "\n";
        fres << prefix << currentSize << ",\"omp2\"," << allTimes[1][0] << "," << allTimes[1][1] << "," << allTimes[1][2] << "\n";
        fres << prefix << currentSize << ",\"omp3\"," << allTimes[2][0] << "," << allTimes[2][1] << "," << allTimes[2][2] << "\n";
        fres.flush();
    }

}


template <class NumType>
void timeSmall(std::ostream& fres, const std::string prefix){
    const size_t MaxSizeV1 = 4*64/sizeof(NumType);
    const size_t MaxSizeV2 = 16*64/sizeof(NumType);
    const int NbLoops = 10000;

    std::unique_ptr<NumType[]> array(new NumType[MaxSizeV2*NbLoops]);

    double allTimes[6] = {0};

    for(size_t currentSize = 1 ; currentSize <= MaxSizeV2 ; currentSize++ ){
        std::cout << "currentSize " << currentSize << std::endl;
        std::cout << "    std::sort " << std::endl;
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                std::sort(&array[idxLoop*currentSize], &array[(idxLoop+1)*currentSize], [&](const NumType& v1, const NumType& v2){
                    return v1 < v2;
                });
            }
            timer.stop();
            std::cout << "    std::sort " << timer.getElapsed() << std::endl;
            const int idxType = 0;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        std::cout << "    insertion " << std::endl;
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                InsertionSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "    insertion " << timer.getElapsed() << std::endl;
            const int idxType = 1;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        if(currentSize <= MaxSizeV1){
            std::cout << "    newqs512 " << std::endl;
            {
                srand48((long int)(currentSize));
                for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                    useVec(&array[idxLoop*currentSize], currentSize);
                    createRandVec(&array[idxLoop*currentSize], currentSize);
                }
            }
            {
                dtimer timer;
                for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                    SortByVec(&array[idxLoop*currentSize], currentSize);
                }
                timer.stop();
                std::cout << "    newqs512 " << timer.getElapsed() << std::endl;
                const int idxType = 2;
                allTimes[idxType] = timer.getElapsed()/double(NbLoops);
            }
            {
                for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                    useVec(&array[idxLoop*currentSize], currentSize);
                }
            }
        }
        std::cout << "    newqs512bitfull " << std::endl;
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                SortByVecBitFull(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "    newqs512bitfull " << timer.getElapsed() << std::endl;
            const int idxType = 3;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        std::cout << "    heapsort " << std::endl;
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort<NumType>::sort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "    heapsort " << timer.getElapsed() << std::endl;
            const int idxType = 4;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        std::cout << "    heapsort512 " << std::endl;
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort512<NumType>::HeapSortNotMultiple(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "    heapsort512 " << timer.getElapsed() << std::endl;
            const int idxType = 5;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        fres << prefix << currentSize << ",\"stdsort\"," << allTimes[0] << "\n";
        fres << prefix << currentSize << ",\"insertion\"," << allTimes[1] << "\n";
        if(currentSize <= MaxSizeV1) fres << prefix << currentSize << ",\"newqs512\"," << allTimes[2] << "\n";
        else  fres << prefix << currentSize << ",\"newqs512\"," << "nan" << "\n";
        fres << prefix << currentSize << ",\"newqs512bitfull\"," << allTimes[3] << "\n";
        fres << prefix << currentSize << ",\"heapsort\"," << allTimes[4] << "\n";
        fres << prefix << currentSize << ",\"heapsort512\"," << allTimes[5] << "\n";
    }

}


template <class NumType>
void timePartitionAll(std::ostream& fres, const std::string prefix){
    const size_t MaxSize = 1073741824;//10L*1024L*1024L*1024L;//10*1024*1024*1024;
    const int NbLoops = 20;

    std::unique_ptr<NumType[]> array(new NumType[MaxSize]);

    for(size_t currentSize = 64 ; currentSize <= MaxSize ; currentSize *= 8 ){
        std::cout << "currentSize " << currentSize << std::endl;


        double allTimes[5][3] = {{ std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                            { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                            { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                            { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                            { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. }};

        for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            std::cout << "  idxLoop " << idxLoop << std::endl;
            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                const NumType pivot = array[(idxLoop*currentSize/NbLoops)];
                dtimer timer;
                std::partition(&array[0], &array[currentSize], [&](const NumType& v){
                    return v < pivot;
                });
                timer.stop();
                std::cout << "    std::partition " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 0;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                const NumType pivot = array[(idxLoop*currentSize/NbLoops)];
                dtimer timer;
                FQuickSort<NumType,size_t>::QsPartition(array.get(), 0, currentSize-1, pivot);
                timer.stop();
                std::cout << "    FQuickSort " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 1;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
            /*{
                srand48((long int)(currentSize));
                createRandVec(array.get(), currentSize);
                const NumType pivot = array[(idxLoop*currentSize/NbLoops)];
                dtimer timer;
                NewQuickSort<NumType,size_t>::Partition(array.get(), 0, currentSize-1, pivot);
                timer.stop();
                std::cout << "    bt " << timer.getElapsed() << std::endl;
                const int idxType = 2;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }*/
            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                const NumType pivot = array[(idxLoop*currentSize/NbLoops)];
                dtimer timer;
                NewQuickSort512<NumType,size_t>::Partition(array.get(), 0, currentSize-1, pivot);
                timer.stop();
                std::cout << "    bt512 " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 3;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                const NumType pivot = array[(idxLoop*currentSize/NbLoops)];
                dtimer timer;
                NewQuickSort512V2<NumType,size_t>::Partition(array.get(), 0, currentSize-1, pivot);
                timer.stop();
                std::cout << "    bt512v2 " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 4;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
        }

        fres << prefix << currentSize << ",\"stdpartion\"," << allTimes[0][0] << "," << allTimes[0][1] << "," << allTimes[0][2] << "\n";
        fres << prefix << currentSize << ",\"qspartition\"," << allTimes[1][0] << "," << allTimes[1][1] << "," << allTimes[1][2] << "\n";
        //fres << prefix << currentSize << ",\"newpartition\"," << allTimes[2][0] << "," << allTimes[2][1] << "," << allTimes[2][2] << "\n";
        fres << prefix << currentSize << ",\"newpartition512\"," << allTimes[3][0] << "," << allTimes[3][1] << "," << allTimes[3][2] << "\n";
        fres << prefix << currentSize << ",\"newpartition512V2\"," << allTimes[4][0] << "," << allTimes[4][1] << "," << allTimes[4][2] << "\n";
        fres.flush();
    }

}



template <class NumType>
void time1Vec(std::ostream& fres, const std::string prefix){
    const int NbLoops = 10*1024*1024;

    double allTimes[9] = {0.};

    {
        const size_t currentSize = 512/(sizeof(NumType)*8);
        std::cout << "currentSize " << currentSize << std::endl;

        std::unique_ptr<NumType[]> array(new NumType[currentSize*NbLoops]);
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                std::sort(&array[idxLoop*currentSize], &array[idxLoop*currentSize+currentSize], [&](const NumType& v1, const NumType& v2){
                    return v1 < v2;
                });
            }
            timer.stop();
            std::cout << "    std::sort " << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 0;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                InsertionSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     InsertionSort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 1;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        /////////////////////////////////////////////////////////////////
//        {
//            srand48((long int)(currentSize));
//            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
//                useVec(&array[idxLoop*currentSize], currentSize);
//                createRandVec(&array[idxLoop*currentSize], currentSize);
//            }
//        }
//        {
//            dtimer timer;
//            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
//                BitonicSortV2<NumType,size_t>::BsSequentialV2(&array[idxLoop*currentSize], currentSize);
//            }
//            timer.stop();
//            std::cout << "     BitonicSortV2" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
//            const int idxType = 2;
//            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
//        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                SortVec(&array[idxLoop*currentSize]);
            }
            timer.stop();
            std::cout << "     SortVec" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 3;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                SortVecWithTest(&array[idxLoop*currentSize]);
            }
            timer.stop();
            std::cout << "     SortVecWithTest" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 4;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                SortVec(&array[idxLoop*currentSize]);
            }
            timer.stop();
            std::cout << "     SortVec" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 5;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                SortVecBitFull(&array[idxLoop*currentSize]);
            }
            timer.stop();
            std::cout << "     SortVecBitFull" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 6;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort<NumType>::sort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 7;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort512<NumType>::HeapSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort512" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 8;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        fres << prefix << currentSize << ",\"stdsort\"," << allTimes[0] << "\n";
        fres << prefix << currentSize << ",\"insertionsort\"," << allTimes[1] << "\n";
        //fres << prefix << currentSize << ",\"bt\"," << allTimes[2] << "\n";
        fres << prefix << currentSize << ",\"bt512\"," << allTimes[3] << "\n";
        fres << prefix << currentSize << ",\"bt512wt\"," << allTimes[4] << "\n";
        fres << prefix << currentSize << ",\"bt512bit\"," << allTimes[5] << "\n";
        fres << prefix << currentSize << ",\"bt512bitfull\"," << allTimes[6] << "\n";
        fres << prefix << currentSize << ",\"heapsort\"," << allTimes[7] << "\n";
        fres << prefix << currentSize << ",\"heapsort512\"," << allTimes[8] << "\n";
    }
}


template <class NumType>
void time2Vec(std::ostream& fres, const std::string prefix){
    const int NbLoops = 10*1024*1024;

    double allTimes[8] = {0.};

    {
        const size_t currentSize = 2*512/(sizeof(NumType)*8);
        std::cout << "currentSize " << currentSize << std::endl;

        std::unique_ptr<NumType[]> array(new NumType[currentSize*NbLoops]);
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                std::sort(&array[idxLoop*currentSize], &array[idxLoop*currentSize+currentSize], [&](const NumType& v1, const NumType& v2){
                    return v1 < v2;
                });
            }
            timer.stop();
            std::cout << "    std::sort " << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 0;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                InsertionSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     InsertionSort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 1;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
//        {
//            srand48((long int)(currentSize));
//            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
//                createRandVec(&array[idxLoop*currentSize], currentSize);
//            }
//        }
//        {
//            dtimer timer;
//            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
//                BitonicSortV2<NumType,size_t>::BsSequentialV2(array.get(), currentSize);
//            }
//            timer.stop();
//            std::cout << "     BitonicSortV2" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
//            const int idxType = 2;
//            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
//        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                Sort2Vec(&array[idxLoop*currentSize], &array[idxLoop*currentSize+currentSize/2]);
            }
            timer.stop();
            std::cout << "     SortVec" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 3;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                Sort2VecWithTest(&array[idxLoop*currentSize], &array[idxLoop*currentSize+currentSize/2]);
            }
            timer.stop();
            std::cout << "     SortVecWithTest" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 4;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                Sort2VecBitFull(&array[idxLoop*currentSize], &array[idxLoop*currentSize+currentSize/2]);
            }
            timer.stop();
            std::cout << "     Sort2VecBitFull" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 5;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort<NumType>::sort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 6;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort512<NumType>::HeapSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort512" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 7;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        fres << prefix << currentSize << ",\"stdsort\"," << allTimes[0] << "\n";
        fres << prefix << currentSize << ",\"insertionsort\"," << allTimes[1] << "\n";
        //fres << prefix << currentSize << ",\"bt\"," << allTimes[2] << "\n";
        fres << prefix << currentSize << ",\"bt512\"," << allTimes[3] << "\n";
        fres << prefix << currentSize << ",\"bt512wt\"," << allTimes[4] << "\n";
        fres << prefix << currentSize << ",\"bt512bitfull\"," << allTimes[5] << "\n";
        fres << prefix << currentSize << ",\"heapsort\"," << allTimes[6] << "\n";
        fres << prefix << currentSize << ",\"heapsort512\"," << allTimes[7] << "\n";
    }
}

template <class NumType>
void time3Vec(std::ostream& fres, const std::string prefix){
    const int NbLoops = 10*1024*1024;

    double allTimes[9] = {0.};

    {
        const size_t currentSize = 3*512/(sizeof(NumType)*8);
        std::cout << "currentSize " << currentSize << std::endl;

        std::unique_ptr<NumType[]> array(new NumType[currentSize*NbLoops]);
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                std::sort(&array[idxLoop*currentSize], &array[idxLoop*currentSize+currentSize], [&](const NumType& v1, const NumType& v2){
                    return v1 < v2;
                });
            }
            timer.stop();
            std::cout << "    std::sort " << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 0;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                InsertionSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     InsertionSort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 1;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
//        {
//            srand48((long int)(currentSize));
//            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
//                createRandVec(&array[idxLoop*currentSize], currentSize);
//            }
//        }
//        {
//            dtimer timer;
//            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
//                BitonicSortV2<NumType,size_t>::BsSequentialV2(array.get(), currentSize);
//            }
//            timer.stop();
//            std::cout << "     BitonicSortV2" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
//            const int idxType = 2;
//            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
//        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            const size_t sizeInVec = 512/(sizeof(NumType)*8);
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                Sort3Vec(&array[idxLoop*currentSize], &array[idxLoop*currentSize+sizeInVec],
                        &array[idxLoop*currentSize+sizeInVec*2]);
            }
            timer.stop();
            std::cout << "     SortVec" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 3;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            const size_t sizeInVec = 512/(sizeof(NumType)*8);
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                Sort3VecBitFull(&array[idxLoop*currentSize], &array[idxLoop*currentSize+sizeInVec],
                        &array[idxLoop*currentSize+sizeInVec*2]);
            }
            timer.stop();
            std::cout << "     Sort3VecBitFull" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 4;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort<NumType>::sort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 5;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort512<NumType>::HeapSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort512" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 6;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        fres << prefix << currentSize << ",\"stdsort\"," << allTimes[0] << "\n";
        fres << prefix << currentSize << ",\"insertionsort\"," << allTimes[1] << "\n";
        //fres << prefix << currentSize << ",\"bt\"," << allTimes[2] << "\n";
        fres << prefix << currentSize << ",\"bt512\"," << allTimes[3] << "\n";
        fres << prefix << currentSize << ",\"bt512bitfull\"," << allTimes[4] << "\n";
        fres << prefix << currentSize << ",\"heapsort\"," << allTimes[5] << "\n";
        fres << prefix << currentSize << ",\"heapsort512\"," << allTimes[6] << "\n";
    }
}



template <class NumType>
void time4Vec(std::ostream& fres, const std::string prefix){
    const int NbLoops = 10*1024*1024;

    double allTimes[8] = {0.};

    {
        const size_t currentSize = 4*512/(sizeof(NumType)*8);
        std::cout << "currentSize " << currentSize << std::endl;

        std::unique_ptr<NumType[]> array(new NumType[currentSize*NbLoops]);
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                std::sort(&array[idxLoop*currentSize], &array[idxLoop*currentSize+currentSize], [&](const NumType& v1, const NumType& v2){
                    return v1 < v2;
                });
            }
            timer.stop();
            std::cout << "    std::sort " << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 0;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                InsertionSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     InsertionSort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 1;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
//        {
//            srand48((long int)(currentSize));
//            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
//                createRandVec(&array[idxLoop*currentSize], currentSize);
//            }
//        }
//        {
//            dtimer timer;
//            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
//                BitonicSortV2<NumType,size_t>::BsSequentialV2(array.get(), currentSize);
//            }
//            timer.stop();
//            std::cout << "     BitonicSortV2" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
//            const int idxType = 2;
//            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
//        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            const size_t sizeInVec = 512/(sizeof(NumType)*8);
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                Sort4Vec(&array[idxLoop*currentSize], &array[idxLoop*currentSize+sizeInVec],
                        &array[idxLoop*currentSize+sizeInVec*2], &array[idxLoop*currentSize+sizeInVec*3]);
            }
            timer.stop();
            std::cout << "     SortVec" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 3;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            const size_t sizeInVec = 512/(sizeof(NumType)*8);
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                Sort4VecBitFull(&array[idxLoop*currentSize], &array[idxLoop*currentSize+sizeInVec],
                        &array[idxLoop*currentSize+sizeInVec*2], &array[idxLoop*currentSize+sizeInVec*3]);
            }
            timer.stop();
            std::cout << "     Sort4VecBitFull" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 4;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort<NumType>::sort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 5;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort512<NumType>::HeapSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort512" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 6;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        fres << prefix << currentSize << ",\"stdsort\"," << allTimes[0] << "\n";
        fres << prefix << currentSize << ",\"insertionsort\"," << allTimes[1] << "\n";
        //fres << prefix << currentSize << ",\"bt\"," << allTimes[2] << "\n";
        fres << prefix << currentSize << ",\"bt512\"," << allTimes[3] << "\n";
        fres << prefix << currentSize << ",\"bt512bitfull\"," << allTimes[4] << "\n";
        fres << prefix << currentSize << ",\"heapsort\"," << allTimes[5] << "\n";
        fres << prefix << currentSize << ",\"heapsort512\"," << allTimes[6] << "\n";
    }
}


template <class NumType>
void time5Vec(std::ostream& fres, const std::string prefix){
    const int NbLoops = 10*1024*1024;

    double allTimes[8] = {0.};

    {
        const size_t currentSize = 5*512/(sizeof(NumType)*8);
        std::cout << "currentSize " << currentSize << std::endl;

        std::unique_ptr<NumType[]> array(new NumType[currentSize*NbLoops]);
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                std::sort(&array[idxLoop*currentSize], &array[idxLoop*currentSize+currentSize], [&](const NumType& v1, const NumType& v2){
                    return v1 < v2;
                });
            }
            timer.stop();
            std::cout << "    std::sort " << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 0;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                InsertionSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     InsertionSort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 1;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            const size_t sizeInVec = 512/(sizeof(NumType)*8);
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                Sort5VecBitFull(&array[idxLoop*currentSize], &array[idxLoop*currentSize+sizeInVec],
                        &array[idxLoop*currentSize+sizeInVec*2], &array[idxLoop*currentSize+sizeInVec*3],
                        &array[idxLoop*currentSize+sizeInVec*4]);
            }
            timer.stop();
            std::cout << "     Sort5VecBitFull" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 4;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort<NumType>::sort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 5;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort512<NumType>::HeapSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort512" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 6;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        fres << prefix << currentSize << ",\"stdsort\"," << allTimes[0] << "\n";
        fres << prefix << currentSize << ",\"insertionsort\"," << allTimes[1] << "\n";
        fres << prefix << currentSize << ",\"bt512\"," << "nan" << "\n";
        fres << prefix << currentSize << ",\"bt512bitfull\"," << allTimes[4] << "\n";
        fres << prefix << currentSize << ",\"heapsort\"," << allTimes[5] << "\n";
        fres << prefix << currentSize << ",\"heapsort512\"," << allTimes[6] << "\n";
    }
}


template <class NumType>
void time6Vec(std::ostream& fres, const std::string prefix){
    const int NbLoops = 10*1024*1024;

    double allTimes[8] = {0.};

    {
        const size_t currentSize = 6*512/(sizeof(NumType)*8);
        std::cout << "currentSize " << currentSize << std::endl;

        std::unique_ptr<NumType[]> array(new NumType[currentSize*NbLoops]);
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                std::sort(&array[idxLoop*currentSize], &array[idxLoop*currentSize+currentSize], [&](const NumType& v1, const NumType& v2){
                    return v1 < v2;
                });
            }
            timer.stop();
            std::cout << "    std::sort " << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 0;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                InsertionSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     InsertionSort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 1;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            const size_t sizeInVec = 512/(sizeof(NumType)*8);
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                Sort6VecBitFull(&array[idxLoop*currentSize], &array[idxLoop*currentSize+sizeInVec],
                        &array[idxLoop*currentSize+sizeInVec*2], &array[idxLoop*currentSize+sizeInVec*3],
                        &array[idxLoop*currentSize+sizeInVec*4], &array[idxLoop*currentSize+sizeInVec*5]);
            }
            timer.stop();
            std::cout << "     Sort6VecBitFull" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 4;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort<NumType>::sort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 5;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort512<NumType>::HeapSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort512" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 6;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        fres << prefix << currentSize << ",\"stdsort\"," << allTimes[0] << "\n";
        fres << prefix << currentSize << ",\"insertionsort\"," << allTimes[1] << "\n";
        fres << prefix << currentSize << ",\"bt512\"," << "nan" << "\n";
        fres << prefix << currentSize << ",\"bt512bitfull\"," << allTimes[4] << "\n";
        fres << prefix << currentSize << ",\"heapsort\"," << allTimes[5] << "\n";
        fres << prefix << currentSize << ",\"heapsort512\"," << allTimes[6] << "\n";
    }
}



template <class NumType>
void time7Vec(std::ostream& fres, const std::string prefix){
    const int NbLoops = 10*1024*1024;

    double allTimes[8] = {0.};

    {
        const size_t currentSize = 7*512/(sizeof(NumType)*8);
        std::cout << "currentSize " << currentSize << std::endl;

        std::unique_ptr<NumType[]> array(new NumType[currentSize*NbLoops]);
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                std::sort(&array[idxLoop*currentSize], &array[idxLoop*currentSize+currentSize], [&](const NumType& v1, const NumType& v2){
                    return v1 < v2;
                });
            }
            timer.stop();
            std::cout << "    std::sort " << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 0;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                InsertionSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     InsertionSort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 1;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            const size_t sizeInVec = 512/(sizeof(NumType)*8);
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                Sort7VecBitFull(&array[idxLoop*currentSize], &array[idxLoop*currentSize+sizeInVec],
                        &array[idxLoop*currentSize+sizeInVec*2], &array[idxLoop*currentSize+sizeInVec*3],
                        &array[idxLoop*currentSize+sizeInVec*4], &array[idxLoop*currentSize+sizeInVec*5],
                        &array[idxLoop*currentSize+sizeInVec*6]);
            }
            timer.stop();
            std::cout << "     Sort7VecBitFull" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 4;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort<NumType>::sort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 5;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort512<NumType>::HeapSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort512" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 6;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        fres << prefix << currentSize << ",\"stdsort\"," << allTimes[0] << "\n";
        fres << prefix << currentSize << ",\"insertionsort\"," << allTimes[1] << "\n";
        fres << prefix << currentSize << ",\"bt512\"," << "nan" << "\n";
        fres << prefix << currentSize << ",\"bt512bitfull\"," << allTimes[4] << "\n";
        fres << prefix << currentSize << ",\"heapsort\"," << allTimes[5] << "\n";
        fres << prefix << currentSize << ",\"heapsort512\"," << allTimes[6] << "\n";
    }
}


template <class NumType>
void time8Vec(std::ostream& fres, const std::string prefix){
    const int NbLoops = 10*1024*1024;

    double allTimes[8] = {0.};

    {
        const size_t currentSize = 8*512/(sizeof(NumType)*8);
        std::cout << "currentSize " << currentSize << std::endl;

        std::unique_ptr<NumType[]> array(new NumType[currentSize*NbLoops]);
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                std::sort(&array[idxLoop*currentSize], &array[idxLoop*currentSize+currentSize], [&](const NumType& v1, const NumType& v2){
                    return v1 < v2;
                });
            }
            timer.stop();
            std::cout << "    std::sort " << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 0;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                InsertionSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     InsertionSort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 1;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            const size_t sizeInVec = 512/(sizeof(NumType)*8);
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                Sort8VecBitFull(&array[idxLoop*currentSize], &array[idxLoop*currentSize+sizeInVec],
                        &array[idxLoop*currentSize+sizeInVec*2], &array[idxLoop*currentSize+sizeInVec*3],
                        &array[idxLoop*currentSize+sizeInVec*4], &array[idxLoop*currentSize+sizeInVec*5],
                        &array[idxLoop*currentSize+sizeInVec*6], &array[idxLoop*currentSize+sizeInVec*7]);
            }
            timer.stop();
            std::cout << "     Sort8VecBitFull" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 4;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort<NumType>::sort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 5;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort512<NumType>::HeapSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort512" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 6;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        fres << prefix << currentSize << ",\"stdsort\"," << allTimes[0] << "\n";
        fres << prefix << currentSize << ",\"insertionsort\"," << allTimes[1] << "\n";
        fres << prefix << currentSize << ",\"bt512\"," << "nan" << "\n";
        fres << prefix << currentSize << ",\"bt512bitfull\"," << allTimes[4] << "\n";
        fres << prefix << currentSize << ",\"heapsort\"," << allTimes[5] << "\n";
        fres << prefix << currentSize << ",\"heapsort512\"," << allTimes[6] << "\n";
    }
}



template <class NumType>
void time9Vec(std::ostream& fres, const std::string prefix){
    const int NbLoops = 10*1024*1024;

    double allTimes[8] = {0.};

    {
        const size_t currentSize = 9*512/(sizeof(NumType)*8);
        std::cout << "currentSize " << currentSize << std::endl;

        std::unique_ptr<NumType[]> array(new NumType[currentSize*NbLoops]);
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                std::sort(&array[idxLoop*currentSize], &array[idxLoop*currentSize+currentSize], [&](const NumType& v1, const NumType& v2){
                    return v1 < v2;
                });
            }
            timer.stop();
            std::cout << "    std::sort " << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 0;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                InsertionSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     InsertionSort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 1;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            const size_t sizeInVec = 512/(sizeof(NumType)*8);
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                Sort9VecBitFull(&array[idxLoop*currentSize], &array[idxLoop*currentSize+sizeInVec],
                        &array[idxLoop*currentSize+sizeInVec*2], &array[idxLoop*currentSize+sizeInVec*3],
                        &array[idxLoop*currentSize+sizeInVec*4], &array[idxLoop*currentSize+sizeInVec*5],
                        &array[idxLoop*currentSize+sizeInVec*6], &array[idxLoop*currentSize+sizeInVec*7],
                        &array[idxLoop*currentSize+sizeInVec*8]);
            }
            timer.stop();
            std::cout << "     Sort9VecBitFull" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 4;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort<NumType>::sort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 5;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort512<NumType>::HeapSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort512" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 6;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        fres << prefix << currentSize << ",\"stdsort\"," << allTimes[0] << "\n";
        fres << prefix << currentSize << ",\"insertionsort\"," << allTimes[1] << "\n";
        fres << prefix << currentSize << ",\"bt512\"," << "nan" << "\n";
        fres << prefix << currentSize << ",\"bt512bitfull\"," << allTimes[4] << "\n";
        fres << prefix << currentSize << ",\"heapsort\"," << allTimes[5] << "\n";
        fres << prefix << currentSize << ",\"heapsort512\"," << allTimes[6] << "\n";
    }
}


template <class NumType>
void time10Vec(std::ostream& fres, const std::string prefix){
    const int NbLoops = 10*1024*1024;

    double allTimes[8] = {0.};

    {
        const size_t currentSize = 10*512/(sizeof(NumType)*8);
        std::cout << "currentSize " << currentSize << std::endl;

        std::unique_ptr<NumType[]> array(new NumType[currentSize*NbLoops]);
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                std::sort(&array[idxLoop*currentSize], &array[idxLoop*currentSize+currentSize], [&](const NumType& v1, const NumType& v2){
                    return v1 < v2;
                });
            }
            timer.stop();
            std::cout << "    std::sort " << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 0;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                InsertionSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     InsertionSort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 1;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            const size_t sizeInVec = 512/(sizeof(NumType)*8);
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                Sort10VecBitFull(&array[idxLoop*currentSize], &array[idxLoop*currentSize+sizeInVec],
                        &array[idxLoop*currentSize+sizeInVec*2], &array[idxLoop*currentSize+sizeInVec*3],
                        &array[idxLoop*currentSize+sizeInVec*4], &array[idxLoop*currentSize+sizeInVec*5],
                        &array[idxLoop*currentSize+sizeInVec*6], &array[idxLoop*currentSize+sizeInVec*7],
                        &array[idxLoop*currentSize+sizeInVec*8], &array[idxLoop*currentSize+sizeInVec*9]);
            }
            timer.stop();
            std::cout << "     Sort10VecBitFull" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 4;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort<NumType>::sort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 5;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort512<NumType>::HeapSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort512" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 6;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        fres << prefix << currentSize << ",\"stdsort\"," << allTimes[0] << "\n";
        fres << prefix << currentSize << ",\"insertionsort\"," << allTimes[1] << "\n";
        fres << prefix << currentSize << ",\"bt512\"," << "nan" << "\n";
        fres << prefix << currentSize << ",\"bt512bitfull\"," << allTimes[4] << "\n";
        fres << prefix << currentSize << ",\"heapsort\"," << allTimes[5] << "\n";
        fres << prefix << currentSize << ",\"heapsort512\"," << allTimes[6] << "\n";
    }
}

template <class NumType>
void time11Vec(std::ostream& fres, const std::string prefix){
    const int NbLoops = 10*1024*1024;

    double allTimes[8] = {0.};

    {
        const size_t currentSize = 11*512/(sizeof(NumType)*8);
        std::cout << "currentSize " << currentSize << std::endl;

        std::unique_ptr<NumType[]> array(new NumType[currentSize*NbLoops]);
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                std::sort(&array[idxLoop*currentSize], &array[idxLoop*currentSize+currentSize], [&](const NumType& v1, const NumType& v2){
                    return v1 < v2;
                });
            }
            timer.stop();
            std::cout << "    std::sort " << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 0;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                InsertionSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     InsertionSort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 1;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            const size_t sizeInVec = 512/(sizeof(NumType)*8);
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                Sort11VecBitFull(&array[idxLoop*currentSize], &array[idxLoop*currentSize+sizeInVec],
                        &array[idxLoop*currentSize+sizeInVec*2], &array[idxLoop*currentSize+sizeInVec*3],
                        &array[idxLoop*currentSize+sizeInVec*4], &array[idxLoop*currentSize+sizeInVec*5],
                        &array[idxLoop*currentSize+sizeInVec*6], &array[idxLoop*currentSize+sizeInVec*7],
                        &array[idxLoop*currentSize+sizeInVec*8], &array[idxLoop*currentSize+sizeInVec*9],
                        &array[idxLoop*currentSize+sizeInVec*10]);
            }
            timer.stop();
            std::cout << "     Sort11VecBitFull" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 4;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort<NumType>::sort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 5;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort512<NumType>::HeapSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort512" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 6;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        fres << prefix << currentSize << ",\"stdsort\"," << allTimes[0] << "\n";
        fres << prefix << currentSize << ",\"insertionsort\"," << allTimes[1] << "\n";
        fres << prefix << currentSize << ",\"bt512\"," << "nan" << "\n";
        fres << prefix << currentSize << ",\"bt512bitfull\"," << allTimes[4] << "\n";
        fres << prefix << currentSize << ",\"heapsort\"," << allTimes[5] << "\n";
        fres << prefix << currentSize << ",\"heapsort512\"," << allTimes[6] << "\n";
    }
}


template <class NumType>
void time12Vec(std::ostream& fres, const std::string prefix){
    const int NbLoops = 10*1024*1024;

    double allTimes[8] = {0.};

    {
        const size_t currentSize = 12*512/(sizeof(NumType)*8);
        std::cout << "currentSize " << currentSize << std::endl;

        std::unique_ptr<NumType[]> array(new NumType[currentSize*NbLoops]);
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                std::sort(&array[idxLoop*currentSize], &array[idxLoop*currentSize+currentSize], [&](const NumType& v1, const NumType& v2){
                    return v1 < v2;
                });
            }
            timer.stop();
            std::cout << "    std::sort " << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 0;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                InsertionSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     InsertionSort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 1;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            const size_t sizeInVec = 512/(sizeof(NumType)*8);
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                Sort12VecBitFull(&array[idxLoop*currentSize], &array[idxLoop*currentSize+sizeInVec],
                        &array[idxLoop*currentSize+sizeInVec*2], &array[idxLoop*currentSize+sizeInVec*3],
                        &array[idxLoop*currentSize+sizeInVec*4], &array[idxLoop*currentSize+sizeInVec*5],
                        &array[idxLoop*currentSize+sizeInVec*6], &array[idxLoop*currentSize+sizeInVec*7],
                        &array[idxLoop*currentSize+sizeInVec*8], &array[idxLoop*currentSize+sizeInVec*9],
                        &array[idxLoop*currentSize+sizeInVec*10], &array[idxLoop*currentSize+sizeInVec*11]);
            }
            timer.stop();
            std::cout << "     Sort12VecBitFull" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 4;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort<NumType>::sort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 5;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort512<NumType>::HeapSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort512" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 6;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        fres << prefix << currentSize << ",\"stdsort\"," << allTimes[0] << "\n";
        fres << prefix << currentSize << ",\"insertionsort\"," << allTimes[1] << "\n";
        fres << prefix << currentSize << ",\"bt512\"," << "nan" << "\n";
        fres << prefix << currentSize << ",\"bt512bitfull\"," << allTimes[4] << "\n";
        fres << prefix << currentSize << ",\"heapsort\"," << allTimes[5] << "\n";
        fres << prefix << currentSize << ",\"heapsort512\"," << allTimes[6] << "\n";
    }
}


template <class NumType>
void time13Vec(std::ostream& fres, const std::string prefix){
    const int NbLoops = 10*1024*1024;

    double allTimes[8] = {0.};

    {
        const size_t currentSize = 13*512/(sizeof(NumType)*8);
        std::cout << "currentSize " << currentSize << std::endl;

        std::unique_ptr<NumType[]> array(new NumType[currentSize*NbLoops]);
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                std::sort(&array[idxLoop*currentSize], &array[idxLoop*currentSize+currentSize], [&](const NumType& v1, const NumType& v2){
                    return v1 < v2;
                });
            }
            timer.stop();
            std::cout << "    std::sort " << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 0;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                InsertionSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     InsertionSort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 1;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            const size_t sizeInVec = 512/(sizeof(NumType)*8);
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                Sort13VecBitFull(&array[idxLoop*currentSize], &array[idxLoop*currentSize+sizeInVec],
                        &array[idxLoop*currentSize+sizeInVec*2], &array[idxLoop*currentSize+sizeInVec*3],
                        &array[idxLoop*currentSize+sizeInVec*4], &array[idxLoop*currentSize+sizeInVec*5],
                        &array[idxLoop*currentSize+sizeInVec*6], &array[idxLoop*currentSize+sizeInVec*7],
                        &array[idxLoop*currentSize+sizeInVec*8], &array[idxLoop*currentSize+sizeInVec*9],
                        &array[idxLoop*currentSize+sizeInVec*10], &array[idxLoop*currentSize+sizeInVec*11],
                        &array[idxLoop*currentSize+sizeInVec*12]);
            }
            timer.stop();
            std::cout << "     Sort12VecBitFull" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 4;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort<NumType>::sort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 5;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort512<NumType>::HeapSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort512" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 6;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        fres << prefix << currentSize << ",\"stdsort\"," << allTimes[0] << "\n";
        fres << prefix << currentSize << ",\"insertionsort\"," << allTimes[1] << "\n";
        fres << prefix << currentSize << ",\"bt512\"," << "nan" << "\n";
        fres << prefix << currentSize << ",\"bt512bitfull\"," << allTimes[4] << "\n";
        fres << prefix << currentSize << ",\"heapsort\"," << allTimes[5] << "\n";
        fres << prefix << currentSize << ",\"heapsort512\"," << allTimes[6] << "\n";
    }
}



template <class NumType>
void time14Vec(std::ostream& fres, const std::string prefix){
    const int NbLoops = 10*1024*1024;

    double allTimes[8] = {0.};

    {
        const size_t currentSize = 14*512/(sizeof(NumType)*8);
        std::cout << "currentSize " << currentSize << std::endl;

        std::unique_ptr<NumType[]> array(new NumType[currentSize*NbLoops]);
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                std::sort(&array[idxLoop*currentSize], &array[idxLoop*currentSize+currentSize], [&](const NumType& v1, const NumType& v2){
                    return v1 < v2;
                });
            }
            timer.stop();
            std::cout << "    std::sort " << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 0;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                InsertionSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     InsertionSort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 1;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            const size_t sizeInVec = 512/(sizeof(NumType)*8);
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                Sort14VecBitFull(&array[idxLoop*currentSize], &array[idxLoop*currentSize+sizeInVec],
                        &array[idxLoop*currentSize+sizeInVec*2], &array[idxLoop*currentSize+sizeInVec*3],
                        &array[idxLoop*currentSize+sizeInVec*4], &array[idxLoop*currentSize+sizeInVec*5],
                        &array[idxLoop*currentSize+sizeInVec*6], &array[idxLoop*currentSize+sizeInVec*7],
                        &array[idxLoop*currentSize+sizeInVec*8], &array[idxLoop*currentSize+sizeInVec*9],
                        &array[idxLoop*currentSize+sizeInVec*10], &array[idxLoop*currentSize+sizeInVec*11],
                        &array[idxLoop*currentSize+sizeInVec*12], &array[idxLoop*currentSize+sizeInVec*13]);
            }
            timer.stop();
            std::cout << "     Sort12VecBitFull" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 4;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort<NumType>::sort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 5;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort512<NumType>::HeapSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort512" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 6;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        fres << prefix << currentSize << ",\"stdsort\"," << allTimes[0] << "\n";
        fres << prefix << currentSize << ",\"insertionsort\"," << allTimes[1] << "\n";
        fres << prefix << currentSize << ",\"bt512\"," << "nan" << "\n";
        fres << prefix << currentSize << ",\"bt512bitfull\"," << allTimes[4] << "\n";
        fres << prefix << currentSize << ",\"heapsort\"," << allTimes[5] << "\n";
        fres << prefix << currentSize << ",\"heapsort512\"," << allTimes[6] << "\n";
    }
}





template <class NumType>
void time15Vec(std::ostream& fres, const std::string prefix){
    const int NbLoops = 10*1024*1024;

    double allTimes[8] = {0.};

    {
        const size_t currentSize = 15*512/(sizeof(NumType)*8);
        std::cout << "currentSize " << currentSize << std::endl;

        std::unique_ptr<NumType[]> array(new NumType[currentSize*NbLoops]);
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                std::sort(&array[idxLoop*currentSize], &array[idxLoop*currentSize+currentSize], [&](const NumType& v1, const NumType& v2){
                    return v1 < v2;
                });
            }
            timer.stop();
            std::cout << "    std::sort " << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 0;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                InsertionSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     InsertionSort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 1;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            const size_t sizeInVec = 512/(sizeof(NumType)*8);
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                Sort15VecBitFull(&array[idxLoop*currentSize], &array[idxLoop*currentSize+sizeInVec],
                        &array[idxLoop*currentSize+sizeInVec*2], &array[idxLoop*currentSize+sizeInVec*3],
                        &array[idxLoop*currentSize+sizeInVec*4], &array[idxLoop*currentSize+sizeInVec*5],
                        &array[idxLoop*currentSize+sizeInVec*6], &array[idxLoop*currentSize+sizeInVec*7],
                        &array[idxLoop*currentSize+sizeInVec*8], &array[idxLoop*currentSize+sizeInVec*9],
                        &array[idxLoop*currentSize+sizeInVec*10], &array[idxLoop*currentSize+sizeInVec*11],
                        &array[idxLoop*currentSize+sizeInVec*12], &array[idxLoop*currentSize+sizeInVec*13],
                        &array[idxLoop*currentSize+sizeInVec*14]);
            }
            timer.stop();
            std::cout << "     Sort12VecBitFull" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 4;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort<NumType>::sort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 5;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort512<NumType>::HeapSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort512" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 6;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        fres << prefix << currentSize << ",\"stdsort\"," << allTimes[0] << "\n";
        fres << prefix << currentSize << ",\"insertionsort\"," << allTimes[1] << "\n";
        fres << prefix << currentSize << ",\"bt512\"," << "nan" << "\n";
        fres << prefix << currentSize << ",\"bt512bitfull\"," << allTimes[4] << "\n";
        fres << prefix << currentSize << ",\"heapsort\"," << allTimes[5] << "\n";
        fres << prefix << currentSize << ",\"heapsort512\"," << allTimes[6] << "\n";
    }
}




template <class NumType>
void time16Vec(std::ostream& fres, const std::string prefix){
    const int NbLoops = 10*1024*1024;

    double allTimes[8] = {0.};

    {
        const size_t currentSize = 16*512/(sizeof(NumType)*8);
        std::cout << "currentSize " << currentSize << std::endl;

        std::unique_ptr<NumType[]> array(new NumType[currentSize*NbLoops]);
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                std::sort(&array[idxLoop*currentSize], &array[idxLoop*currentSize+currentSize], [&](const NumType& v1, const NumType& v2){
                    return v1 < v2;
                });
            }
            timer.stop();
            std::cout << "    std::sort " << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 0;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                InsertionSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     InsertionSort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 1;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            const size_t sizeInVec = 512/(sizeof(NumType)*8);
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                Sort16VecBitFull(&array[idxLoop*currentSize], &array[idxLoop*currentSize+sizeInVec],
                        &array[idxLoop*currentSize+sizeInVec*2], &array[idxLoop*currentSize+sizeInVec*3],
                        &array[idxLoop*currentSize+sizeInVec*4], &array[idxLoop*currentSize+sizeInVec*5],
                        &array[idxLoop*currentSize+sizeInVec*6], &array[idxLoop*currentSize+sizeInVec*7],
                        &array[idxLoop*currentSize+sizeInVec*8], &array[idxLoop*currentSize+sizeInVec*9],
                        &array[idxLoop*currentSize+sizeInVec*10], &array[idxLoop*currentSize+sizeInVec*11],
                        &array[idxLoop*currentSize+sizeInVec*12], &array[idxLoop*currentSize+sizeInVec*13],
                        &array[idxLoop*currentSize+sizeInVec*14], &array[idxLoop*currentSize+sizeInVec*15]);
            }
            timer.stop();
            std::cout << "     Sort12VecBitFull" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 4;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort<NumType>::sort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 5;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        /////////////////////////////////////////////////////////////////
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                HeapSort512<NumType>::HeapSort(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "     heapsort512" << timer.getElapsed() << "(" << timer.getElapsed()/double(NbLoops) << ")" << std::endl;
            const int idxType = 6;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }

        fres << prefix << currentSize << ",\"stdsort\"," << allTimes[0] << "\n";
        fres << prefix << currentSize << ",\"insertionsort\"," << allTimes[1] << "\n";
        fres << prefix << currentSize << ",\"bt512\"," << "nan" << "\n";
        fres << prefix << currentSize << ",\"bt512bitfull\"," << allTimes[4] << "\n";
        fres << prefix << currentSize << ",\"heapsort\"," << allTimes[5] << "\n";
        fres << prefix << currentSize << ",\"heapsort512\"," << allTimes[6] << "\n";
    }
}


/**
 * TODO:
 * increase the lenght of by-hand bitonic 512 sort
 * Variant 1
 * sort locally
 * merge and resort locally
 *
 * Variant 2
 * intro sort (partition 512 + heap sort todo)
 */

int main(){
    const bool test = false;
    if(test){
        testAll();
    }
    else{
        {
            std::ofstream fres("smallres.csv");
            fres << "\"type\",\"size\",\"algo\",\"avgtime\"\n";

            timeSmall<int>(fres, "\"int\",");
            timeSmall<double>(fres, "\"double\",");
        }
        {
            std::ofstream fres("vec.csv");
            fres << "\"type\",\"size\",\"algo\",\"avgtime\"\n";

            time1Vec<int>(fres, "\"int\",");
            time1Vec<double>(fres, "\"double\",");

            time2Vec<int>(fres, "\"int\",");
            time2Vec<double>(fres, "\"double\",");

            time3Vec<int>(fres, "\"int\",");
            time3Vec<double>(fres, "\"double\",");

            time4Vec<int>(fres, "\"int\",");
            time4Vec<double>(fres, "\"double\",");

            time5Vec<int>(fres, "\"int\",");
            time5Vec<double>(fres, "\"double\",");

            time6Vec<int>(fres, "\"int\",");
            time6Vec<double>(fres, "\"double\",");

            time7Vec<int>(fres, "\"int\",");
            time7Vec<double>(fres, "\"double\",");

            time8Vec<int>(fres, "\"int\",");
            time8Vec<double>(fres, "\"double\",");

            time9Vec<int>(fres, "\"int\",");
            time9Vec<double>(fres, "\"double\",");

            time10Vec<int>(fres, "\"int\",");
            time10Vec<double>(fres, "\"double\",");

            time11Vec<int>(fres, "\"int\",");
            time11Vec<double>(fres, "\"double\",");

            time12Vec<int>(fres, "\"int\",");
            time12Vec<double>(fres, "\"double\",");

            time13Vec<int>(fres, "\"int\",");
            time13Vec<double>(fres, "\"double\",");

            time14Vec<int>(fres, "\"int\",");
            time14Vec<double>(fres, "\"double\",");

            time15Vec<int>(fres, "\"int\",");
            time15Vec<double>(fres, "\"double\",");

            time16Vec<int>(fres, "\"int\",");
            time16Vec<double>(fres, "\"double\",");
        }
        {
            std::ofstream fres("partitions.csv");
            fres << "\"type\",\"size\",\"algo\",\"mintime\",\"maxtime\",\"avgtime\"\n";

            timePartitionAll<int>(fres, "\"int\",");
            timePartitionAll<double>(fres, "\"double\",");
        }
        {
            std::ofstream fres("res.csv");
            fres << "\"type\",\"size\",\"algo\",\"mintime\",\"maxtime\",\"avgtime\"\n";

            timeAll<int>(fres, "\"int\",");
            timeAll<double>(fres, "\"double\",");
        }
    }

    return 0;
}
