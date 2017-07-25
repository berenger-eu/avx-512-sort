//////////////////////////////////////////////////////////
/// Code to sort an array of integer or double
/// using avx 512 (targeting intel KNL).
/// By berenger.bramas@mpcdf.mpg.de 2017.
/// Licence is MIT.
/// Comes without any warranty.
///
/// This file contains tests (which may be used as examples)
///
/// Can be compiled with:
/// - KNL
/// Gcc : g++ -DNDEBUG -O3 -funroll-loops -faggressive-loop-optimizations -std=c++11 -mavx512f -mavx512pf -mavx512er -mavx512cd -fopenmp sort512test.cpp -o sort512test.gcc.exe
/// Intel : icpc -DNDEBUG -O3 -std=c++11 -xCOMMON-AVX512 -xMIC-AVX512 -qopenmp sort512test.cpp -o sort512test.intel.exe
/// - SKL
/// Gcc : g++ -DNDEBUG -O3 -funroll-loops -faggressive-loop-optimizations -std=c++11 -mavx512f -mavx512cd -mavx512vl -mavx512bw -mavx512dq -fopenmp sort512test.cpp -o sort512test.gcc.exe
/// Intel : icpc -DNDEBUG -O3 -std=c++11 -xCOMMON-AVX512 -xCORE-AVX512 -qopenmp sort512test.cpp -o sort512test.intel.exe
//////////////////////////////////////////////////////////

#include "sort512.hpp"
#include "sort512kv.hpp"

#include <iostream>
#include <memory>
#include <cstdlib>

int test_res = 0;


template <class NumType>
void assertNotSorted(const NumType array[], const size_t size, const std::string log){
    for(size_t idx = 1 ; idx < size ; ++idx){
        if(array[idx-1] > array[idx]){
            std::cout << "assertNotSorted -- Array is not sorted\n"
                         "assertNotSorted --    - at pos " << idx << "\n"
                                                                     "assertNotSorted --    - log " << log << std::endl;
            test_res = 1;
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
            test_res = 1;
        }
    }
    for(size_t idx = limite ; idx < size ; ++idx){
        if(array[idx] <= pivot){
            std::cout << "assertNotPartitioned -- Array is not partitioned\n"
                         "assertNotPartitioned --    - at pos " << idx << "\n"
                                                                          "assertNotPartitioned --    - log " << log << std::endl;
            test_res = 1;
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
            test_res = 1;
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


void testPopcount(){
    std::cout << "Start testPopcount...\n";
    auto assertFunc = [](const int trueres, const int test, const int val, const std::string& logbuf){
        if(test != trueres){
            std::cout << "testPopcount errror - " << logbuf << "\n";
            std::cout << "testPopcount errror - for val " << val << "\n";
            std::cout << "testPopcount errror - should be " << trueres << " is " << test << "\n";
            test_res = 1;
        }
    };

    assertFunc(0, Sort512::popcount(__mmask16(0)), 0, "__mmask16");
    assertFunc(0, Sort512::popcount(__mmask8(0)), 0, "__mmask8");

    for(int idx = 0 ; idx < 16 ; ++idx){
        assertFunc(1, Sort512::popcount(__mmask16(1)), 1<<idx, "__mmask16");
        if(idx < 8) assertFunc(1, Sort512::popcount(__mmask8(1)), 1<<idx, "__mmask8");
    }

    assertFunc(2, Sort512::popcount(__mmask16(3)), 3, "__mmask16");
    assertFunc(2, Sort512::popcount(__mmask8(3)), 3, "__mmask8");

    assertFunc(16, Sort512::popcount(__mmask16(0xFFFF)), 0xFFFF, "__mmask16");
    assertFunc(8, Sort512::popcount(__mmask8(0xFF)), 0xFF, "__mmask8");
}

void testSortVec_Core_Equal(const double toSort[8], const double sorted[8]){
    double res[8];

    _mm512_storeu_pd(res, Sort512::CoreSmallSort(_mm512_loadu_pd(toSort)));
    assertNotSorted(res, 8, "testSortVec_Core_Equal");
    assertNotEqual(res, sorted, 8, "testSortVec_Core_Equal");
}


void testSortVec_Core_Equal(const int toSort[16], const int sorted[16]){
    int res[16];

    _mm512_storeu_si512(res, Sort512::CoreSmallSort(_mm512_loadu_si512(toSort)));
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
                _mm512_storeu_pd(res, Sort512::CoreSmallSort(_mm512_loadu_pd(vecTest)));
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
                _mm512_storeu_si512(res, Sort512::CoreSmallSort(_mm512_loadu_si512(vecTest)));
                assertNotSorted(res, 16, "testSortVec_Core_Equal");
            }
        }
    }
}

void testSortVec_pair(){
    std::cout << "Start testSortVec_pair int...\n";
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

            int values[16];
            for(int idxval = 0 ; idxval < 16 ; ++idxval){
                values[idxval] = vecTest[idxval]*100+1;
            }

            {
                Sort512kv::CoreSmallSort(vecTest, values);
                assertNotSorted(vecTest, 16, "testSortVec_Core_Equal");
            }
            for(int idxval = 0 ; idxval < 16 ; ++idxval){
                if(values[idxval] != vecTest[idxval]*100+1){
                    std::cout << "Error in testSortVec_pair "
                                 " is " << values[idxval] <<
                                 " should be " << vecTest[idxval]*100+1 << std::endl;
                    test_res = 1;
                }
            }

        }
    }
}

void testSort2Vec_Core_Equal(const double toSort[16], const double sorted[16]){
    double res[16];

    __m512d vec1 = _mm512_loadu_pd(toSort);
    __m512d vec2 = _mm512_loadu_pd(toSort+8);
    Sort512::CoreSmallSort2(vec1, vec2);
    _mm512_storeu_pd(res, vec1);
    _mm512_storeu_pd(res+8, vec2);
    assertNotSorted(res, 16, "testSort2Vec_Core_Equal");
    assertNotEqual(res, sorted, 16, "testSort2Vec_Core_Equal");
}

void testSort2Vec_Core_Equal(const int toSort[32], const int sorted[32]){
    int res[32];

    __m512i vec1 = _mm512_loadu_si512(toSort);
    __m512i vec2 = _mm512_loadu_si512(toSort+16);
    Sort512::CoreSmallSort2(vec1, vec2);
    _mm512_storeu_si512(res, vec1);
    _mm512_storeu_si512(res+16, vec2);
    assertNotSorted(res, 32, "testSort2Vec_Core_Equal");
    assertNotEqual(res, sorted, 32, "testSort2Vec_Core_Equal");
}


void testSort2Vec(){
    std::cout << "Start Sort512::CoreSmallSort2 double...\n";
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
                createRandVec(vecTest, 16);
                Checker<double> checker(vecTest, vecTest, 16);
                Sort512::CoreSmallSort2(vecTest, vecTest+8);
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
                createRandVec(vecTest, 32);
                Checker<int> checker(vecTest, vecTest, 32);
                Sort512::CoreSmallSort2(vecTest, vecTest+16);
                assertNotSorted(vecTest, 32, "testSortVec_Core_Equal");
            }
        }
    }
}

void testSort2Vec_pair(){
    std::cout << "Start testSort2Vec_pair int...\n";
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[32];
            createRandVec(vecTest, 32);

            int values[32];
            for(int idxval = 0 ; idxval < 32 ; ++idxval){
                values[idxval] = vecTest[idxval]*100+1;
            }
            {
                Checker<int> checker(vecTest, vecTest, 32);
                Sort512kv::CoreSmallSort2(vecTest, values);
                assertNotSorted(vecTest, 32, "testSortVec_Core_Equal");
            }
            for(int idxval = 0 ; idxval < 32 ; ++idxval){
                if(values[idxval] != vecTest[idxval]*100+1){
                    std::cout << "Error in testSortVec_pair "
                                 " is " << values[idxval] <<
                                 " should be " << vecTest[idxval]*100+1 << std::endl;
                    test_res = 1;
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
                Sort512::CoreSmallSort3(vecTest, vecTest+8, vecTest+16);
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
                Sort512::CoreSmallSort3(vecTest, vecTest+16, vecTest+32);
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
                Sort512::CoreSmallSort4(vecTest, vecTest+8, vecTest+16, vecTest+24);
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
                Sort512::CoreSmallSort4(vecTest, vecTest+8, vecTest+16, vecTest+24);
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
                Sort512::CoreSmallSort4(vecTest, vecTest+16, vecTest+32, vecTest+48);
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
                Sort512::CoreSmallSort4(vecTest, vecTest+16, vecTest+32, vecTest+48);
                assertNotSorted(vecTest, 64, "testSortVec_Core_Equal");
            }
        }
    }
}

void testSort3Vec_pair(){
    std::cout << "Start testSort3Vec_pair int...\n";
    {
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[48];
            createRandVec(vecTest, 48);

            int values[48];
            for(int idxval = 0 ; idxval < 48 ; ++idxval){
                values[idxval] = vecTest[idxval]*100+1;
            }
            {
                Checker<int> checker(vecTest, vecTest, 48);
                Sort512kv::CoreSmallSort3(vecTest, values);
                assertNotSorted(vecTest, 48, "testSortVec_Core_Equal");
            }

            for(int idxval = 0 ; idxval < 48 ; ++idxval){
                if(values[idxval] != vecTest[idxval]*100+1){
                    std::cout << "Error in testSortVec_pair "
                                 " is " << values[idxval] <<
                                 " should be " << vecTest[idxval]*100+1 << std::endl;
                    test_res = 1;
                }
            }
        }
    }
}

void testSort4Vec_pair(){
    std::cout << "Start testSort4Vec_pair int...\n";
    {
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[64];
            createRandVec(vecTest, 64);

            int values[64];
            for(int idxval = 0 ; idxval < 64 ; ++idxval){
                values[idxval] = vecTest[idxval]*100+1;
            }

            {
                Checker<int> checker(vecTest, vecTest, 64);
                Sort512kv::CoreSmallSort4(vecTest, values);
                assertNotSorted(vecTest, 64, "testSortVec_Core_Equal");
            }

            for(int idxval = 0 ; idxval < 64 ; ++idxval){
                if(values[idxval] != vecTest[idxval]*100+1){
                    std::cout << "Error in testSortVec_pair "
                                 " is " << values[idxval] <<
                                 " should be " << vecTest[idxval]*100+1 << std::endl;
                    test_res = 1;
                }
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
            Sort512::CoreSmallSort5(vecTest, vecTest+8, vecTest+8*2, vecTest+8*3, vecTest+8*4);
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
            Sort512::CoreSmallSort5(vecTest, vecTest+16, vecTest+16*2, vecTest+16*3, vecTest+16*4);
            assertNotSorted(vecTest, 5*16, "testSortVec_Core_Equal");
        }
    }
}

void testSort5Vec_pair(){
    std::cout << "Start testSort5Vec_pair int...\n";
    {
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[5*16];

            createRandVec(vecTest, 5*16);

            int values[5*16];
            for(int idxval = 0 ; idxval < 5*16 ; ++idxval){
                values[idxval] = vecTest[idxval]*100+1;
            }

            Checker<int> checker(vecTest, vecTest, 5*16);
            Sort512kv::CoreSmallSort5(vecTest, values);
            assertNotSorted(vecTest, 5*16, "testSortVec_Core_Equal");

            for(int idxval = 0 ; idxval < 5*16 ; ++idxval){
                if(values[idxval] != vecTest[idxval]*100+1){
                    std::cout << "Error in testSortVec_pair "
                                 " is " << values[idxval] <<
                                 " should be " << vecTest[idxval]*100+1 << std::endl;
                    test_res = 1;
                }
            }
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
            Sort512::CoreSmallSort6(vecTest, vecTest+8, vecTest+8*2, vecTest+8*3, vecTest+8*4, vecTest+8*5);
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
            Sort512::CoreSmallSort6(vecTest, vecTest+16, vecTest+16*2, vecTest+16*3, vecTest+16*4, vecTest+16*5);
            assertNotSorted(vecTest, 6*16, "testSortVec_Core_Equal");
        }
    }
}

void testSort6Vec_pair(){
    std::cout << "Start testSort6Vec_pair int...\n";
    {
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[6*16];

            createRandVec(vecTest, 6*16);

            int values[6*16];
            for(int idxval = 0 ; idxval < 6*16 ; ++idxval){
                values[idxval] = vecTest[idxval]*100+1;
            }

            Checker<int> checker(vecTest, vecTest, 6*16);
            Sort512kv::CoreSmallSort6(vecTest, values);
            assertNotSorted(vecTest, 6*16, "testSortVec_Core_Equal");

            for(int idxval = 0 ; idxval < 6*16 ; ++idxval){
                if(values[idxval] != vecTest[idxval]*100+1){
                    std::cout << "Error in testSortVec_pair "
                                 " is " << values[idxval] <<
                                 " should be " << vecTest[idxval]*100+1 << std::endl;
                    test_res = 1;
                }
            }
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
            Sort512::CoreSmallSort7(vecTest, vecTest+8, vecTest+8*2, vecTest+8*3, vecTest+8*4, vecTest+8*5, vecTest+8*6);
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
            Sort512::CoreSmallSort7(vecTest, vecTest+16, vecTest+16*2, vecTest+16*3, vecTest+16*4, vecTest+16*5, vecTest+16*6);
            assertNotSorted(vecTest, 7*16, "testSortVec_Core_Equal");
        }
    }
}

void testSort7Vec_pair(){
    std::cout << "Start testSort7Vec_pair int...\n";
    {
        static const int Size = 7;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[Size*16];

            createRandVec(vecTest, Size*16);

            int values[Size*16];
            for(int idxval = 0 ; idxval < Size*16 ; ++idxval){
                values[idxval] = vecTest[idxval]*100+1;
            }

            Checker<int> checker(vecTest, vecTest, Size*16);
            Sort512kv::CoreSmallSort7(vecTest, values);
            assertNotSorted(vecTest, Size*16, "testSortVec_Core_Equal");

            for(int idxval = 0 ; idxval < Size*16 ; ++idxval){
                if(values[idxval] != vecTest[idxval]*100+1){
                    std::cout << "Error in testSortVec_pair "
                                 " is " << values[idxval] <<
                                 " should be " << vecTest[idxval]*100+1 << std::endl;
                    test_res = 1;
                }
            }
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
            Sort512::CoreSmallSort8(vecTest, vecTest+8, vecTest+8*2, vecTest+8*3, vecTest+8*4, vecTest+8*5, vecTest+8*6, vecTest+8*7);
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
            Sort512::CoreSmallSort8(vecTest, vecTest+16, vecTest+16*2, vecTest+16*3, vecTest+16*4, vecTest+16*5, vecTest+16*6, vecTest+16*7);
            assertNotSorted(vecTest, 8*16, "testSortVec_Core_Equal");
        }
    }
}

void testSort8Vec_pair(){
    std::cout << "Start testSort8Vec_pair int...\n";
    {
        static const int Size = 8;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[Size*16];

            createRandVec(vecTest, Size*16);

            int values[Size*16];
            for(int idxval = 0 ; idxval < Size*16 ; ++idxval){
                values[idxval] = vecTest[idxval]*100+1;
            }

            Checker<int> checker(vecTest, vecTest, Size*16);
            Sort512kv::CoreSmallSort8(vecTest, values);
            assertNotSorted(vecTest, Size*16, "testSortVec_Core_Equal");

            for(int idxval = 0 ; idxval < Size*16 ; ++idxval){
                if(values[idxval] != vecTest[idxval]*100+1){
                    std::cout << "Error in testSortVec_pair "
                                 " is " << values[idxval] <<
                                 " should be " << vecTest[idxval]*100+1 << std::endl;
                    test_res = 1;
                }
            }
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
            Sort512::CoreSmallSort9(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
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
            Sort512::CoreSmallSort9(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                    vecTest+sizeVec*8);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
    }
}

void testSort9Vec_pair(){
    std::cout << "Start testSort9Vec_pair int...\n";
    {
        static const int Size = 9;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[Size*16];

            createRandVec(vecTest, Size*16);

            int values[Size*16];
            for(int idxval = 0 ; idxval < Size*16 ; ++idxval){
                values[idxval] = vecTest[idxval]*100+1;
            }

            Checker<int> checker(vecTest, vecTest, Size*16);
            Sort512kv::CoreSmallSort9(vecTest, values);
            assertNotSorted(vecTest, Size*16, "testSortVec_Core_Equal");

            for(int idxval = 0 ; idxval < Size*16 ; ++idxval){
                if(values[idxval] != vecTest[idxval]*100+1){
                    std::cout << "Error in testSortVec_pair "
                                 " is " << values[idxval] <<
                                 " should be " << vecTest[idxval]*100+1 << std::endl;
                    test_res = 1;
                }
            }
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
            Sort512::CoreSmallSort10(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
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
            Sort512::CoreSmallSort10(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                     vecTest+sizeVec*8, vecTest+sizeVec*9);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
    }
}

void testSort10Vec_pair(){
    std::cout << "Start testSort10Vec_pair int...\n";
    {
        static const int Size = 10;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[Size*16];

            createRandVec(vecTest, Size*16);

            int values[Size*16];
            for(int idxval = 0 ; idxval < Size*16 ; ++idxval){
                values[idxval] = vecTest[idxval]*100+1;
            }

            Checker<int> checker(vecTest, vecTest, Size*16);
            Sort512kv::CoreSmallSort10(vecTest, values);
            assertNotSorted(vecTest, Size*16, "testSortVec_Core_Equal");

            for(int idxval = 0 ; idxval < Size*16 ; ++idxval){
                if(values[idxval] != vecTest[idxval]*100+1){
                    std::cout << "Error in testSortVec_pair "
                                 " is " << values[idxval] <<
                                 " should be " << vecTest[idxval]*100+1 << std::endl;
                    test_res = 1;
                }
            }
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
            Sort512::CoreSmallSort11(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
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
            Sort512::CoreSmallSort11(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                     vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
    }
}

void testSort11Vec_pair(){
    std::cout << "Start testSort11Vec_pair int...\n";
    {
        static const int Size = 11;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[Size*16];

            createRandVec(vecTest, Size*16);

            int values[Size*16];
            for(int idxval = 0 ; idxval < Size*16 ; ++idxval){
                values[idxval] = vecTest[idxval]*100+1;
            }

            Checker<int> checker(vecTest, vecTest, Size*16);
            Sort512kv::CoreSmallSort11(vecTest, values);
            assertNotSorted(vecTest, Size*16, "testSortVec_Core_Equal");

            for(int idxval = 0 ; idxval < Size*16 ; ++idxval){
                if(values[idxval] != vecTest[idxval]*100+1){
                    std::cout << "Error in testSortVec_pair "
                                 " is " << values[idxval] <<
                                 " should be " << vecTest[idxval]*100+1 << std::endl;
                    test_res = 1;
                }
            }
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
            Sort512::CoreSmallSort12(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
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
            Sort512::CoreSmallSort12(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                     vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
    }
}

void testSort12Vec_pair(){
    std::cout << "Start testSort12Vec_pair int...\n";
    {
        static const int Size = 12;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[Size*16];

            createRandVec(vecTest, Size*16);

            int values[Size*16];
            for(int idxval = 0 ; idxval < Size*16 ; ++idxval){
                values[idxval] = vecTest[idxval]*100+1;
            }

            Checker<int> checker(vecTest, vecTest, Size*16);
            Sort512kv::CoreSmallSort12(vecTest, values);
            assertNotSorted(vecTest, Size*16, "testSortVec_Core_Equal");

            for(int idxval = 0 ; idxval < Size*16 ; ++idxval){
                if(values[idxval] != vecTest[idxval]*100+1){
                    std::cout << "Error in testSortVec_pair "
                                 " is " << values[idxval] <<
                                 " should be " << vecTest[idxval]*100+1 << std::endl;
                    test_res = 1;
                }
            }
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
            Sort512::CoreSmallSort13(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
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
            Sort512::CoreSmallSort13(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                     vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
    }
}

void testSort13Vec_pair(){
    std::cout << "Start testSort13Vec_pair int...\n";
    {
        static const int Size = 13;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[Size*16];

            createRandVec(vecTest, Size*16);

            int values[Size*16];
            for(int idxval = 0 ; idxval < Size*16 ; ++idxval){
                values[idxval] = vecTest[idxval]*100+1;
            }

            Checker<int> checker(vecTest, vecTest, Size*16);
            Sort512kv::CoreSmallSort13(vecTest, values);
            assertNotSorted(vecTest, Size*16, "testSortVec_Core_Equal");

            for(int idxval = 0 ; idxval < Size*16 ; ++idxval){
                if(values[idxval] != vecTest[idxval]*100+1){
                    std::cout << "Error in testSortVec_pair "
                                 " is " << values[idxval] <<
                                 " should be " << vecTest[idxval]*100+1 << std::endl;
                    test_res = 1;
                }
            }
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
            Sort512::CoreSmallSort14(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
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
            Sort512::CoreSmallSort14(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                     vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12, vecTest+sizeVec*13);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
    }
}

void testSort14Vec_pair(){
    std::cout << "Start testSort14Vec_pair int...\n";
    {
        static const int Size = 14;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[Size*16];

            createRandVec(vecTest, Size*16);

            int values[Size*16];
            for(int idxval = 0 ; idxval < Size*16 ; ++idxval){
                values[idxval] = vecTest[idxval]*100+1;
            }

            Checker<int> checker(vecTest, vecTest, Size*16);
            Sort512kv::CoreSmallSort14(vecTest, values);
            assertNotSorted(vecTest, Size*16, "testSortVec_Core_Equal");

            for(int idxval = 0 ; idxval < Size*16 ; ++idxval){
                if(values[idxval] != vecTest[idxval]*100+1){
                    std::cout << "Error in testSortVec_pair "
                                 " is " << values[idxval] <<
                                 " should be " << vecTest[idxval]*100+1 << std::endl;
                    test_res = 1;
                }
            }
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
            Sort512::CoreSmallSort15(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
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
            Sort512::CoreSmallSort15(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                     vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12, vecTest+sizeVec*13, vecTest+sizeVec*14);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
    }
}

void testSort15Vec_pair(){
    std::cout << "Start testSort15Vec_pair int...\n";
    {
        static const int Size = 15;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[Size*16];

            createRandVec(vecTest, Size*16);

            int values[Size*16];
            for(int idxval = 0 ; idxval < Size*16 ; ++idxval){
                values[idxval] = vecTest[idxval]*100+1;
            }

            Checker<int> checker(vecTest, vecTest, Size*16);
            Sort512kv::CoreSmallSort15(vecTest, values);
            assertNotSorted(vecTest, Size*16, "testSortVec_Core_Equal");

            for(int idxval = 0 ; idxval < Size*16 ; ++idxval){
                if(values[idxval] != vecTest[idxval]*100+1){
                    std::cout << "Error in testSortVec_pair "
                                 " is " << values[idxval] <<
                                 " should be " << vecTest[idxval]*100+1 << std::endl;
                    test_res = 1;
                }
            }
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
            Sort512::CoreSmallSort16(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
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
            Sort512::CoreSmallSort16(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                     vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12, vecTest+sizeVec*13, vecTest+sizeVec*14, vecTest+sizeVec*15);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
    }
}

void testSort16Vec_pair(){
    std::cout << "Start testSort16Vec_pair int...\n";
    {
        static const int Size = 16;
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[Size*16];

            createRandVec(vecTest, Size*16);

            int values[Size*16];
            for(int idxval = 0 ; idxval < Size*16 ; ++idxval){
                values[idxval] = vecTest[idxval]*100+1;
            }

            Checker<int> checker(vecTest, vecTest, Size*16);
            Sort512kv::CoreSmallSort16(vecTest, values);
            assertNotSorted(vecTest, Size*16, "testSortVec_Core_Equal");

            for(int idxval = 0 ; idxval < Size*16 ; ++idxval){
                if(values[idxval] != vecTest[idxval]*100+1){
                    std::cout << "Error in testSortVec_pair "
                                 " is " << values[idxval] <<
                                 " should be " << vecTest[idxval]*100+1 << std::endl;
                    test_res = 1;
                }
            }
        }
    }
}

template <class NumType>
void testQs512(){
    std::cout << "Start Sort512 sort...\n";
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        Sort512::Sort<NumType,size_t>(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
#if defined(_OPENMP)
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        Sort512::SortOmp<NumType,size_t>(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
#endif
}

template <class NumType>
void testQs512_pair(){
    std::cout << "Start testQs512_pair...\n";
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        std::unique_ptr<NumType[]> values(new NumType[idx]);
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            values[idxval] = array[idxval]*100+1;
        }
        Sort512kv::Sort<NumType,size_t>(array.get(), values.get(), idx);
        assertNotSorted(array.get(), idx, "");
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            if(values[idxval] != array[idxval]*100+1){
                std::cout << "Error in testNewPartition512V2_pair, pair/key do not match" << std::endl;
                test_res = 1;
            }
        }
    }
#if defined(_OPENMP)
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        std::unique_ptr<NumType[]> values(new NumType[idx]);
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            values[idxval] = array[idxval]*100+1;
        }
        Sort512kv::SortOmp<NumType,size_t>(array.get(), values.get(), idx);
        assertNotSorted(array.get(), idx, "");
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            if(values[idxval] != array[idxval]*100+1){
                std::cout << "Error in testNewPartition512V2_pair, pair/key do not match" << std::endl;
                test_res = 1;
            }
        }
    }
#endif
}


template <class NumType>
void testPartition(){
    std::cout << "Start Sort512::Partition512...\n";
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        const NumType pivot = NumType(idx/2);
        size_t limite = Sort512::Partition512<size_t>(&array[0], 0, idx-1, pivot);
        assertNotPartitioned(array.get(), idx, pivot, limite, "");
    }
    for(size_t idx = 1 ; idx <= 1000; ++idx){
        if(idx%100 == 0) std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        const NumType pivot = NumType(idx/2);
        size_t limite = Sort512::Partition512<size_t>(&array[0], 0, idx-1, pivot);
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
        size_t limite = Sort512::Partition512<size_t>(&array[0], 0, idx-1, pivot);
        assertNotPartitioned(array.get(), idx, pivot, limite, "");
    }
}

template <class NumType>
void testPartition_pair(){
    std::cout << "Start testPartition_pair...\n";
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        std::unique_ptr<NumType[]> values(new NumType[idx]);
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            values[idxval] = array[idxval]*100+1;
        }
        const NumType pivot = NumType(idx/2);
        size_t limite = Sort512kv::Partition512<size_t>(&array[0], &values[0], 0, idx-1, pivot);
        assertNotPartitioned(array.get(), idx, pivot, limite, "");
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            if(values[idxval] != array[idxval]*100+1){
                std::cout << "Error in testNewPartition512V2_pair, pair/key do not match" << std::endl;
                test_res = 1;
            }
        }
    }
    for(size_t idx = 1 ; idx <= 1000; ++idx){
        if(idx%100 == 0) std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        std::unique_ptr<NumType[]> values(new NumType[idx]);
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            values[idxval] = array[idxval]*100+1;
        }
        const NumType pivot = NumType(idx/2);
        size_t limite = Sort512kv::Partition512<size_t>(&array[0], &values[0], 0, idx-1, pivot);
        assertNotPartitioned(array.get(), idx, pivot, limite, "");
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            if(values[idxval] != array[idxval]*100+1){
                std::cout << "Error in testNewPartition512V2_pair, pair/key do not match" << std::endl;
                test_res = 1;
            }
        }
    }
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);

        for(size_t idxVal = 0 ; idxVal < idx ; ++idxVal){
            array[idxVal] = NumType(idx);
        }

        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        std::unique_ptr<NumType[]> values(new NumType[idx]);
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            values[idxval] = array[idxval]*100+1;
        }
        const NumType pivot = NumType(idx/2);
        size_t limite = Sort512kv::Partition512<size_t>(&array[0], &values[0], 0, idx-1, pivot);
        assertNotPartitioned(array.get(), idx, pivot, limite, "");
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            if(values[idxval] != array[idxval]*100+1){
                std::cout << "Error in testNewPartition512V2_pair, pair/key do not match" << std::endl;
                test_res = 1;
            }
        }
    }
}


template <class NumType>
void testSmallVecSort(){
    std::cout << "Start Sort512::SmallSort16V...\n";
    {
        const int SizeVec = 64/sizeof(NumType);
        const int MaxSizeAllVec = SizeVec * 16;
        for(size_t idx = 1 ; idx <= MaxSizeAllVec; idx++){
            std::cout << "   " << idx << std::endl;
            std::unique_ptr<NumType[]> array(new NumType[idx]);
            for(int idxTest = 0 ; idxTest < 100 ; ++idxTest){
                createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
                Sort512::SmallSort16V(array.get(), idx);
                assertNotSorted(array.get(), idx, "");
            }
        }
    }
}

template <class NumType>
void testSmallVecSort_pair(){
    std::cout << "Start testSmallVecSort_pair bitfull...\n";
    {
        const int SizeVec = 64/sizeof(NumType);
        const int MaxSizeAllVec = SizeVec * 16;
        for(size_t idx = 1 ; idx <= MaxSizeAllVec; idx++){
            std::cout << "   " << idx << std::endl;
            std::unique_ptr<NumType[]> array(new NumType[idx]);
            std::unique_ptr<NumType[]> values(new NumType[idx]);
            for(int idxTest = 0 ; idxTest < 100 ; ++idxTest){
                createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);

                for(int idxval = 0 ; idxval < idx ; ++idxval){
                    values[idxval] = array[idxval]*100+1;
                }

                Sort512kv::SmallSort16V(array.get(), values.get(), idx);
                assertNotSorted(array.get(), idx, "");

                for(int idxval = 0 ; idxval < idx ; ++idxval){
                    if(values[idxval] != array[idxval]*100+1){
                        std::cout << "Error in testSortVec_pair "
                                     " is " << values[idxval] <<
                                     " should be " << array[idxval]*100+1 << std::endl;
                        test_res = 1;
                    }
                }
            }
        }
    }
}


int main(){
    testPopcount();

    testSortVec();
    testSortVec_pair();
    testSort2Vec();
    testSort2Vec_pair();
    testSort3Vec();
    testSort3Vec_pair();
    testSort4Vec();
    testSort4Vec_pair();
    testSort5Vec();
    testSort5Vec_pair();
    testSort6Vec();
    testSort6Vec_pair();
    testSort7Vec();
    testSort7Vec_pair();
    testSort8Vec();
    testSort8Vec_pair();

    testSort9Vec();
    testSort9Vec_pair();
    testSort10Vec();
    testSort10Vec_pair();
    testSort11Vec();
    testSort11Vec_pair();
    testSort12Vec();
    testSort12Vec_pair();
    testSort13Vec();
    testSort13Vec_pair();
    testSort14Vec();
    testSort14Vec_pair();
    testSort15Vec();
    testSort15Vec_pair();
    testSort16Vec();
    testSort16Vec_pair();

    testSmallVecSort<int>();
    testSmallVecSort<double>();
    testSmallVecSort_pair<int>();

    testQs512<double>();
    testQs512<int>();
    testQs512_pair<int>();

    testPartition<int>();
    testPartition<double>();
    testPartition_pair<int>();

    if(test_res != 0){
        std::cout << "Test failed!" << std::endl;
    }

    return test_res;
}
