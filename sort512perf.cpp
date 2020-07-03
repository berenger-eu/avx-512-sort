////////////////////////////////////////////////////////////
/// Berenger Bramas - 2016
/// berenger.bramas@mpcdf.mpg.de
/// MIT Licence
/// AVX 512 Sorting algorithm
////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
/// Compilation :
/// Gcc : g++ -DNDEBUG -O3 -funroll-loops -faggressive-loop-optimizations -std=c++11 -mavx512f -mavx512pf -mavx512er -mavx512cd -fopenmp sort512perf.cpp -o sort512perf.gcc.exe
/// Intel : icpc -DNDEBUG -O3 -std=c++11 -xCOMMON-AVX512 -xMIC-AVX512 -qopenmp sort512perf.cpp -o sort512perf.intel.exe
///
///
/// SKL:
/// Gcc : g++ -DNDEBUG -O3 -funroll-loops -faggressive-loop-optimizations -std=c++11 -mavx512f -mavx512cd -mavx512vl -mavx512bw -mavx512dq -fopenmp sort512perf.cpp -o sort512perf.gcc.exe
/// Intel : icpc -DNDEBUG -O3 -std=c++11 -xCOMMON-AVX512 -xCORE-AVX512 -qopenmp sort512perf.cpp -o sort512perf.intel.exe
///
/// With Ipp:
/// Gcc : g++ -DNDEBUG -DUSE_IPP -O3 -funroll-loops -faggressive-loop-optimizations -std=c++11 -mavx512f -mavx512pf -mavx512er -mavx512cd -fopenmp sort512perf.cpp -o sort512perf.gcc.exe -I $IPPROOT/include -L $IPPROOT/lib/intel64 -lippi -lipps -lippcore -Wl,-rpath=$IPPROOT/lib/intel64
/// Intel : icpc -DNDEBUG -DUSE_IPP -O3 -std=c++11 -xCOMMON-AVX512 -xMIC-AVX512 -qopenmp sort512perf.cpp -o sort512perf.intel.exe  -I $IPPROOT/include -L $IPPROOT/lib/intel64 -lippi -lipps -lippcore -Wl,-rpath,$IPPROOT/lib/intel64
///
/// SKL IPP:
/// Gcc : g++ -DNDEBUG -DUSE_IPP -O3 -funroll-loops -faggressive-loop-optimizations -std=c++11 -mavx512f -mavx512cd -mavx512vl -mavx512bw -mavx512dq -fopenmp sort512perf.cpp -o sort512perf.gcc.exe  -I $IPPROOT/include -L $IPPROOT/lib/intel64 -lippi -lipps -lippcore -Wl,-rpath=$IPPROOT/lib/intel64
/// Intel : icpc -DNDEBUG -DUSE_IPP -O3 -std=c++11 -xCOMMON-AVX512 -xCORE-AVX512 -qopenmp sort512perf.cpp -o sort512perf.intel.exe  -I $IPPROOT/include -L $IPPROOT/lib/intel64 -lippi -lipps -lippcore -Wl,-rpath,$IPPROOT/lib/intel64
///
/// Numa:
/// In Flat mode list with : numactl --hardware
/// Then run with : numactl --physcpubind=8 --membind=1  EXEC
////////////////////////////////////////////////////////////
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
#include <cmath>
#include <cstdlib>

#include "sort512.hpp"
#include "sort512kv.hpp"

// Default alignement for the complete application by redirecting the new operator
static const int DefaultMemAlignement = 128;

namespace aligned_malloc {

template <std::size_t AlignementValue>
inline void* malloc(const std::size_t inSize){
    if(inSize == 0){
        return nullptr;
    }

    // Ensure it is a power of 2
    static_assert(AlignementValue != 0 && ((AlignementValue-1)&AlignementValue) == 0, "Alignement must be a power of 2");
    // We will need to store the adress of the real blocks
    const std::size_t sizeForAddress = (AlignementValue < sizeof(unsigned char*)? sizeof(unsigned char*) : AlignementValue);

    unsigned char* allocatedMemory      = reinterpret_cast<unsigned char*>(std::malloc(inSize + AlignementValue-1 + sizeForAddress));
    unsigned char* alignedMemoryAddress = reinterpret_cast<unsigned char*>((reinterpret_cast<std::size_t>(allocatedMemory) + AlignementValue-1 + sizeForAddress) & ~static_cast<std::size_t>(AlignementValue-1));
    unsigned char* ptrForAddress        = (alignedMemoryAddress - sizeof(unsigned char*));

    // Save allocated adress
    *reinterpret_cast<unsigned char**>(ptrForAddress) = allocatedMemory;
    // Return aligned address
    return reinterpret_cast<void*>(alignedMemoryAddress);
}

inline void free(void* ptrToFree){
    if( ptrToFree ){
        unsigned char** storeRealAddress = reinterpret_cast<unsigned char**>(reinterpret_cast<unsigned char*>(ptrToFree) - sizeof(unsigned char*));
        std::free(*storeRealAddress);
    }
}
}


// Regular scalar new
void* operator new(std::size_t n) {
    void* const allocated = aligned_malloc::malloc<DefaultMemAlignement>(n);
    if(allocated){
        return allocated;
    }
    throw std::bad_alloc();
    return allocated;
}

void* operator new[]( std::size_t n ) {
    void* const allocated = aligned_malloc::malloc<DefaultMemAlignement>(n);
    if(allocated){
        return allocated;
    }
    throw std::bad_alloc();
    return allocated;
}

void* operator new  ( std::size_t n, const std::nothrow_t& tag){
    void* const allocated = aligned_malloc::malloc<DefaultMemAlignement>(n);
    return allocated;
}

void* operator new[] ( std::size_t n, const std::nothrow_t& tag){
    void* const allocated = aligned_malloc::malloc<DefaultMemAlignement>(n);
    return allocated;
}

// Regular scalar delete
void operator delete(void* p) {
    aligned_malloc::free(p);
}

void operator delete[](void* p) {
    aligned_malloc::free(p);
}

void operator delete  ( void* p, const std::nothrow_t& /*tag*/) {
    aligned_malloc::free(p);
}

void operator delete[]( void* p, const std::nothrow_t& /*tag*/) {
    aligned_malloc::free(p);
}

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////


#ifdef USE_IPP

#include <ipp.h>
#include <ipps_l.h>
#include <ippbase.h>
#include <memory>


template <class NumType>
struct GetIppDataType;

// https://software.intel.com/en-us/node/501992

template <>
struct GetIppDataType<double>{
    static const IppDataType type = ipp64f;
};

template <>
struct GetIppDataType<int>{
    static const IppDataType type = ipp32s;
};

template <class NumType>
class IppSort{
    static void ipp_sort(Ipp64f* pSrcDst, IppSizeL len, Ipp8u* pBuffer){
        IppStatus status = ippsSortRadixAscend_64f_I(pSrcDst, len, pBuffer);// TODO cannot find ippsSortRadixAscend_64f_I_L
        if( status != ippStsNoErr ) {
            std::cout << "ippsSortRadixAscend() Error, at line " << __LINE__ << ":\n";
            std::cout << ippGetStatusString(status) << std::endl;
            exit(1);
        }
    }

    static void ipp_sort(Ipp32s* pSrcDst, IppSizeL len, Ipp8u* pBuffer){
        IppStatus status = ippsSortRadixAscend_32s_I_L(pSrcDst, len, pBuffer);
        if( status != ippStsNoErr ) {
            std::cout << "ippsSortRadixAscend() Error, at line " << __LINE__ << ":\n";
            std::cout << ippGetStatusString(status) << std::endl;
            exit(1);
        }
    }


    const size_t sizeToSort;
    std::unique_ptr<Ipp8u[]> buffer;


public:
    explicit IppSort(const IppSizeL inSizeToSort)
        : sizeToSort(inSizeToSort){
        IppSizeL lenghtBuffer = 0;
        IppStatus status = ippsSortRadixGetBufferSize_L(inSizeToSort, GetIppDataType<NumType>::type, &lenghtBuffer);
        if( status != ippStsNoErr ) {
            std::cout << "ippsSortRadixGetBufferSize_L() Error, at line " << __LINE__ << ":\n";
            std::cout << ippGetStatusString(status) << std::endl;
            exit(1);
        }
        buffer.reset(new Ipp8u[lenghtBuffer]());
    }

    void sort(NumType* array){
        ipp_sort(array, sizeToSort, buffer.get());
    }
};

#else
#warning "IPP is disabled"
#endif

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

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


////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

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
////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
/// Timing functions
////////////////////////////////////////////////////////////

#include <fstream>

template <class NumType>
void timeAll(std::ostream& fres){
    const size_t MaxSize = 1073741824;//10L*1024L*1024L*1024L;//10*1024*1024*1024;262144*8;//
    const int NbLoops = 5;

    std::unique_ptr<NumType[]> array(new NumType[MaxSize]);

    fres << "#size\tstdsort\tsort512";
#ifdef USE_IPP
    fres << "\tipp\tipplogn";
#endif
    fres << "\n";

    for(size_t currentSize = 64 ; currentSize <= MaxSize ; currentSize *= 8 ){
        std::cout << "currentSize " << currentSize << std::endl;


        double allTimes[3][3] = {{ std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                            { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                            { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. }};
#ifdef USE_IPP
        IppSort<NumType> ippsort(currentSize);
#endif

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
                Sort512::Sort<NumType, size_t>(array.get(), currentSize);
                timer.stop();
                std::cout << "    Sort512 " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 1;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
#ifdef USE_IPP
            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                dtimer timer;
                ippsort.sort(array.get());
                timer.stop();
                std::cout << "    IPPSORT " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 2;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
#endif
        }

        std::cout << currentSize << ",\"stdsort\"," << allTimes[0][0] << "," << allTimes[0][1] << "," << allTimes[0][2] << "\n";
        std::cout << currentSize << ",\"sort512\"," << allTimes[1][0] << "," << allTimes[1][1] << "," << allTimes[1][2] << "\n";
        std::cout << currentSize << ",\"ipp\"," << allTimes[2][0] << "," << allTimes[2][1] << "," << allTimes[2][2] << "\n";


        fres << currentSize << "\t"
             << allTimes[0][2] << "\t" << allTimes[0][2]/(currentSize*std::log(currentSize)) << "\t"
             << allTimes[1][2] << "\t" << allTimes[1][2]/(currentSize*std::log(currentSize));

#ifdef USE_IPP
        fres << "\t" << allTimes[2][2] << "\t" <<
                allTimes[2][2]/(currentSize*std::log(currentSize));
#endif
        fres << "\n";
    }

}


template <class NumType>
void timeAll_pair(std::ostream& fres){
    const size_t MaxSize = 1073741824;//10L*1024L*1024L*1024L;//10*1024*1024*1024;262144*8;//
    const int NbLoops = 5;

    std::unique_ptr<NumType[]> array(new NumType[MaxSize]);
    std::unique_ptr<NumType[]> values(new NumType[MaxSize]());

    std::unique_ptr<std::array<NumType,2>[]> arrayStruct(new std::array<NumType,2>[MaxSize]());

    fres << "#size\tstdsort\tsort512";
    fres << "\n";

    for(size_t currentSize = 64 ; currentSize <= MaxSize ; currentSize *= 8 ){
        std::cout << "currentSize " << currentSize << std::endl;

        double allTimes[2][3] = {{ std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                            { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. }};

        for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            std::cout << "  idxLoop " << idxLoop << std::endl;
            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                for(int idxItem = 0 ; idxItem < currentSize ; ++idxItem){
                    arrayStruct[idxItem][0] = array[idxItem];
                }
                dtimer timer;
                std::sort(&arrayStruct[0], &arrayStruct[currentSize], [&](const std::array<NumType,2>& v1, const std::array<NumType,2>& v2){
                    return v1[0] < v2[0];
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
                Sort512kv::Sort<NumType, size_t>(array.get(), values.get(), currentSize);
                timer.stop();
                std::cout << "    sort512 " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 1;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
        }

        std::cout << currentSize << ",\"stdsort\"," << allTimes[0][0] << "," << allTimes[0][1] << "," << allTimes[0][2] << "\n";
        std::cout << currentSize << ",\"sort512\"," << allTimes[1][0] << "," << allTimes[1][1] << "," << allTimes[1][2] << "\n";

        fres << currentSize << "\t"
             << allTimes[0][2] << "\t" << allTimes[0][2]/(currentSize*std::log(currentSize)) << "\t"
             << allTimes[1][2] << "\t" << allTimes[1][2]/(currentSize*std::log(currentSize));
        fres << "\n";
    }

}

#if defined(_OPENMP)

template <class NumType>
void timeAllOmp(std::ostream& fres, const std::string prefix){
    const size_t MaxSize = 1073741824;//10L*1024L*1024L*1024L;//10*1024*1024*1024;262144*8;//
    const int NbLoops = 5;

    std::unique_ptr<NumType[]> array(new NumType[MaxSize]);

    for(size_t currentSize = 64 ; currentSize <= MaxSize ; currentSize *= 8 ){
        std::cout << "currentSize " << currentSize << std::endl;


        double allTimes[4][3] = {{ std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                            { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                            { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. }};

        for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            std::cout << "  idxLoop " << idxLoop << std::endl;

            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                dtimer timer;
                Sort512::SortOmpPartition<NumType, size_t>(array.get(), currentSize);
                timer.stop();
                std::cout << "    SortOmpPartition " << timer.getElapsed() << std::endl;
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
                Sort512::SortOmpMerge<NumType, size_t>(array.get(), currentSize);
                timer.stop();
                std::cout << "    SortOmpMerge " << timer.getElapsed() << std::endl;
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
                Sort512::SortOmpMergeDeps<NumType, size_t>(array.get(), currentSize);
                timer.stop();
                std::cout << "    SortOmpMergeDeps " << timer.getElapsed() << std::endl;
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
                Sort512::SortOmpParMerge<NumType, size_t>(array.get(), currentSize);
                timer.stop();
                std::cout << "    SortOmpParMerge " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 3;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
        }

        fres << prefix << currentSize << ",\"SortOmpPartition\"," << allTimes[0][0] << "," << allTimes[0][1] << "," << allTimes[0][2] << "\n";
        fres << prefix << currentSize << ",\"SortOmpMerge\"," << allTimes[1][0] << "," << allTimes[1][1] << "," << allTimes[1][2] << "\n";
        fres << prefix << currentSize << ",\"SortOmpMergeDeps\"," << allTimes[2][0] << "," << allTimes[2][1] << "," << allTimes[2][2] << "\n";
        fres << prefix << currentSize << ",\"SortOmpParMerge\"," << allTimes[3][0] << "," << allTimes[3][1] << "," << allTimes[3][2] << "\n";
        fres.flush();
    }

}

#endif

template <class NumType>
void timeSmall(std::ostream& fres){
    const size_t MaxSizeV2 = 16*64/sizeof(NumType);
    const int NbLoops = 10000;

    std::unique_ptr<NumType[]> array(new NumType[MaxSizeV2*NbLoops]);

    double allTimes[3] = {0};

        fres << "#size\tstdsort\tstdsortlogn\tsort512\tsort512logn";
#ifdef USE_IPP
        fres << "\tipp\tipplogn";
#endif
        fres << "\n";

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
                Sort512::SmallSort16V(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "    sort512 " << timer.getElapsed() << std::endl;
            const int idxType = 1;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
#ifdef USE_IPP
        std::cout << "    ipp " << std::endl;
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            IppSort<NumType> ippsort(currentSize);
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                ippsort.sort(&array[idxLoop*currentSize]);
            }
            timer.stop();
            std::cout << "    ipp " << timer.getElapsed() << std::endl;
            const int idxType = 2;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
#endif
        fres << currentSize << "\t" << allTimes[0] << "\t" <<
                allTimes[0]/(currentSize*std::log(currentSize)) << "\t" << allTimes[1] << "\t" <<
                allTimes[1]/(currentSize*std::log(currentSize));
#ifdef USE_IPP
        fres << "\t" << allTimes[2] << "\t" <<
                allTimes[2]/(currentSize*std::log(currentSize));
#endif
        fres << "\n";
    }

}



template <class NumType>
void timeSmall_pair(std::ostream& fres){
    const size_t MaxSizeV2 = 16*64/sizeof(NumType);
    const int NbLoops = 10000;

    std::unique_ptr<NumType[]> array(new NumType[MaxSizeV2*NbLoops]);
    std::unique_ptr<NumType[]> indexes(new NumType[MaxSizeV2*NbLoops]());

    std::unique_ptr<std::array<NumType,2>[]> arrayStruct(new std::array<NumType,2>[MaxSizeV2*NbLoops]());


    double allTimes[3] = {0};

        fres << "#size\tstdsort\tstdsortlogn\tsort512\tsort512logn";
        fres << "\n";

    for(size_t currentSize = 1 ; currentSize <= MaxSizeV2 ; currentSize++ ){
        std::cout << "currentSize " << currentSize << std::endl;
        std::cout << "    std::sort " << std::endl;
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
                for(int idxItem = 0 ; idxItem < currentSize ; ++idxItem){
                    arrayStruct[idxLoop*currentSize+idxItem][0] = array[idxLoop*currentSize+idxItem];
                }
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                std::sort(&arrayStruct[idxLoop*currentSize], &arrayStruct[(idxLoop+1)*currentSize], [&](const std::array<NumType,2>& v1, const std::array<NumType,2>& v2){
                    return v1[0] < v2[0];
                });
            }
            timer.stop();
            std::cout << "    std::sort " << timer.getElapsed() << std::endl;
            const int idxType = 0;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        std::cout << "    sort512 " << std::endl;
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
                Sort512kv::SmallSort16V(&array[idxLoop*currentSize], &indexes[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "    sort512 " << timer.getElapsed() << std::endl;
            const int idxType = 1;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        fres << currentSize << "\t" << allTimes[0] << "\t" <<
                allTimes[0]/(currentSize*std::log(currentSize)) << "\t" << allTimes[1] << "\t" <<
                allTimes[1]/(currentSize*std::log(currentSize));
        fres << "\n";
    }

}


template <class NumType>
void timePartitionAll(std::ostream& fres){
    const size_t MaxSize = 1073741824;//10L*1024L*1024L*1024L;//10*1024*1024*1024;
    const int NbLoops = 20;

    std::unique_ptr<NumType[]> array(new NumType[MaxSize]);

    fres << "#size\tstdpart\tstdpartn\tpartition512\tpartition512n";
    fres << "\n";

    for(size_t currentSize = 64 ; currentSize <= MaxSize ; currentSize *= 8 ){
        std::cout << "currentSize " << currentSize << std::endl;


        double allTimes[2][3] = {{ std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
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
                Sort512::Partition512<size_t>(array.get(), 0, currentSize-1, pivot);
                timer.stop();
                std::cout << "    partition512 " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 1;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
        }

        std::cout << currentSize << ",\"stdpartion\"," << allTimes[0][0] << "," << allTimes[0][1] << "," << allTimes[0][2] << "\n";
        std::cout << currentSize << ",\"partition512\"," << allTimes[1][0] << "," << allTimes[1][1] << "," << allTimes[1][2] << "\n";

        fres << currentSize << "\t"
             << allTimes[0][2] << "\t" << allTimes[0][2]/(currentSize) << "\t"
             << allTimes[1][2] << "\t" << allTimes[1][2]/(currentSize) << "\n";
    }
}

template <class NumType>
void timePartitionAll_pair(std::ostream& fres){
    const size_t MaxSize = 1073741824;//10L*1024L*1024L*1024L;//10*1024*1024*1024;
    const int NbLoops = 20;

    std::unique_ptr<NumType[]> array(new NumType[MaxSize]);
    std::unique_ptr<NumType[]> values(new NumType[MaxSize]());


    std::unique_ptr<std::array<NumType,2>[]> arrayStruct(new std::array<NumType,2>[MaxSize]());

    fres << "#size\tstdpart\tstdpartn\tpartition512\tpartition512n";
    fres << "\n";

    for(size_t currentSize = 64 ; currentSize <= MaxSize ; currentSize *= 8 ){
        std::cout << "currentSize " << currentSize << std::endl;


        double allTimes[2][3] = {{ std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                            { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. }};

        for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            std::cout << "  idxLoop " << idxLoop << std::endl;
            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                for(int idxItem = 0 ; idxItem < currentSize ; ++idxItem){
                    arrayStruct[idxItem][0] = array[idxItem];
                }
                const NumType pivot = array[(idxLoop*currentSize/NbLoops)];
                dtimer timer;
                std::partition(&arrayStruct[0], &arrayStruct[currentSize], [&](const std::array<NumType,2>& v){
                    return v[0] < pivot;
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
                Sort512kv::Partition512<size_t>(array.get(), values.get(), 0, currentSize-1, pivot);
                timer.stop();
                std::cout << "    partition512 " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 1;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
        }

        std::cout << currentSize << ",\"stdpartion\"," << allTimes[0][0] << "," << allTimes[0][1] << "," << allTimes[0][2] << "\n";
        std::cout << currentSize << ",\"partition512V2\"," << allTimes[1][0] << "," << allTimes[1][1] << "," << allTimes[1][2] << "\n";

        fres << currentSize << "\t"
             << allTimes[0][2] << "\t" << allTimes[0][2]/(currentSize) << "\t"
             << allTimes[1][2] << "\t" << allTimes[1][2]/(currentSize) << "\n";
    }
}


////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

int main(){
    #ifdef USE_IPP
    IppStatus status=ippInit();
    if( status != ippStsNoErr ) {
            std::cout << "IppInit() Error:\n";
            std::cout << ippGetStatusString(status) << std::endl;
            return -1;
    }
    #endif

    {
        std::ofstream fres("smallres-int.data");
        timeSmall<int>(fres);
    }
    {
        std::ofstream fres("smallres-double.data");
        timeSmall<double>(fres);
    }
    {
        std::ofstream fres("smallres-pair-int.data");
        timeSmall_pair<int>(fres);
    }
    {
        std::ofstream fres("partitions-int.data");
        timePartitionAll<int>(fres);
    }
    {
        std::ofstream fres("partitions-double.data");
        timePartitionAll<double>(fres);
    }
    {
        std::ofstream fres("partitions-pair-int.data");
        timePartitionAll_pair<int>(fres);
    }
    {
        std::ofstream fres("res-int.data");
        timeAll<int>(fres);
    }
    {
        std::ofstream fres("res-double.data");
        timeAll<double>(fres);
    }
    {
        std::ofstream fres("res-pair-int.data");
        timeAll_pair<int>(fres);
    }
#if defined(_OPENMP)
    {
        std::ofstream fres("res-int-openmp.data");
        timeAllOmp<int>(fres, "max-threads");
    }
    {
        std::ofstream fres("res-double-openmp.data");
        timeAllOmp<double>(fres, "max-threads");
    }
#endif
    return 0;
}
