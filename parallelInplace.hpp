//////////////////////////////////////////////////////////
/// Code to sort merge two arrays in parallel taken
/// from an existing study.
///
/// By berenger.bramas@inria.fr 2020.
/// Licence is MIT.
/// Comes without any warranty.
///
/// Please refer to:
/// https://gitlab.inria.fr/bramas/inplace-merge
/// https://hal.inria.fr/hal-02613668
//////////////////////////////////////////////////////////
#ifndef PARALLELINPLACE_HPP
#define PARALLELINPLACE_HPP

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <sys/time.h>
#include <omp.h>
#include <utility>
#include <algorithm>
#include <cassert>
#include <array>
#include <memory>
#include <iostream>// TODO

#include <omp.h>

namespace ParallelInplace {


// Reorder an array in place with extra moves :
// if we have [0 1 2 3 4 5 ; A B C]
// it creates [A B C ; 0 1 2 3 4 5]
template <class NumType>
inline void reorderShifting(NumType array[], const int lengthLeftPart, const int totalSize){
    const int lengthRightPart = totalSize - lengthLeftPart;
    // if one of the size is zero just return
    if(lengthLeftPart == 0 || lengthRightPart == 0){
        // do nothing
        return;
    }

    // size of the partitions at first iteration
    int workingLeftLength  = lengthLeftPart;
    int workingRightLength = lengthRightPart;
    // while the partitions have different sizes and none of them are null
    while(workingLeftLength != workingRightLength && workingLeftLength && workingRightLength){
        // if the left partition is the smallest
        if(workingLeftLength < workingRightLength){
            // move the left parition in the correct place
            for(int idx = 0 ; idx < workingLeftLength ; ++idx){
                std::swap(array[idx], array[idx + workingRightLength]);
            }
            // the new left partition is now the values that have been swaped
            workingRightLength = workingRightLength - workingLeftLength;
            //workingLeftLength = workingLeftLength;
        }
        // if right partition is the smallest
        else{
            // move the right partition in the correct place
            for(int idx = 0 ; idx < workingRightLength ; ++idx){
                std::swap(array[idx], array[idx + workingLeftLength]);
            }
            // shift the pointer to skip the correct values
            array = (array + workingRightLength);
            // the new left partition is the previous right minus the swaped values
            //workingRightLength = workingRightLength;
            workingLeftLength  = workingLeftLength - workingRightLength;
        }
    }
    // if partitions have the same size
    for(int idx = 0 ; idx < workingLeftLength ; ++idx){
        std::swap(array[idx], array[idx + workingLeftLength]);
    }
}

////////////////////////////////////////////////////////////////
// Merge functions
////////////////////////////////////////////////////////////////

template <class NumType>
void FindMedian(NumType array[],  int centerPosition, const int sizeArray,
                            int* middleA, int* middleB){
    if(centerPosition == 0 || centerPosition == sizeArray || array[centerPosition-1] <= array[centerPosition]){
        *middleA = centerPosition;
        *middleB = 0;
        return;
    }
    if(!(array[0] <= array[sizeArray-1])){
        *middleA = 0;
        *middleB = sizeArray-centerPosition;
    }

    int leftStart = 0;
    int leftLimite = centerPosition;
    int leftPivot = (leftLimite-leftStart)/2 + leftStart;

    int rightStart = centerPosition;
    int rightLimte = sizeArray;
    int rightPivot = (rightLimte-rightStart)/2 + rightStart;

    while(leftStart < leftLimite && rightStart < rightLimte
            && !(array[leftPivot] == array[rightPivot])){
        assert( leftPivot <  leftLimite);
        assert( rightPivot <  rightLimte);
        assert( leftPivot <  centerPosition);
        assert( rightPivot <  sizeArray);

        const int A0 = leftPivot-0;
        const int A1 = centerPosition-leftPivot;
        const int B0 = rightPivot-centerPosition;
        const int B1 = sizeArray-rightPivot;

        if(array[leftPivot] < array[rightPivot]){
            if(A0+B0 < A1+B1){
                leftStart = leftPivot+1;
                leftPivot = (leftLimite-leftStart)/2 + leftStart;
            }
            else{
                rightLimte = rightPivot;
                rightPivot = (rightLimte-rightStart)/2 + rightStart;
            }
        }
        else{
            if(A0+B0 < A1+B1){
                rightStart = rightPivot+1;
                rightPivot = (rightLimte-rightStart)/2 + rightStart;
            }
            else{
                leftLimite = leftPivot;
                leftPivot = (leftLimite-leftStart)/2 + leftStart;
            }
        }
    }

    *middleA = leftPivot;
    *middleB = rightPivot-centerPosition;
}


template <class NumType>
struct WorkingInterval{
    NumType* array;
    int currentStart;
    int currentMiddle;
    int currentEnd;
    int level;
    int depthLimite;
};

template <class NumType>
inline void parallelMergeInPlaceCore(NumType array[], int currentStart, int currentMiddle, int currentEnd,
                                    int level, const int depthLimite,
                                    WorkingInterval<NumType> intervals[], int barrier[]){

    assert(0 <= currentStart);
    assert(currentStart <= currentMiddle);
    assert(currentMiddle <= currentEnd);

    if(currentStart != currentMiddle && currentMiddle != currentEnd){
        while(level != depthLimite && (currentEnd-currentStart) > 256){
            int middleA = 0;
            int middleB = 0;

            FindMedian(array + currentStart,  currentMiddle - currentStart, currentEnd-currentStart,
                        &middleA, &middleB);

            const int sizeRestA = currentMiddle-currentStart-middleA;
            const int sizeRestB = currentEnd-currentMiddle-middleB;

            reorderShifting(array + middleA + currentStart, sizeRestA, middleB+sizeRestA);

            const int targetThread = (1 << (depthLimite - level - 1)) + omp_get_thread_num();

#pragma omp critical(out)
std::cout << omp_get_thread_num() << "] Ok for " << targetThread << std::endl;// TODO

            // Should be #pragma omp atomic write
            intervals[targetThread] = WorkingInterval<NumType>{array,
                                     currentStart+middleA+middleB,
                                     currentStart+middleA+middleB+sizeRestA,
                                     currentEnd,
                                     level+1, depthLimite};
            #pragma omp atomic write
            barrier[targetThread] = 1;

            currentEnd = currentStart+middleA+middleB;
            currentMiddle = currentStart+middleA;

            assert(0 <= currentStart);
            assert(currentStart <= currentMiddle);
            assert(currentMiddle <= currentEnd);

            level += 1;
        }

        std::inplace_merge(array + currentStart, array+currentMiddle, array+currentEnd);
    }

    while(level != depthLimite){
        const int targetThread = (1 << (depthLimite - level - 1)) + omp_get_thread_num();

#pragma omp critical(out)
std::cout << omp_get_thread_num() << "] Ok no work for " << targetThread << std::endl;// TODO

        // Should be #pragma omp atomic write
        intervals[targetThread] = WorkingInterval<NumType>{array,
                                 currentEnd,
                                 currentEnd,
                                 currentEnd,
                                 level+1, depthLimite};
        #pragma omp atomic write
        barrier[targetThread] = 1;

        level += 1;
    }
}

template <class NumType>
inline void parallelMergeInPlace(NumType array[], const int sizeArray, int centerPosition,
                                 const long int numThreadsInvolved, const long int firstThread,
                                 WorkingInterval<NumType> intervals[], int barrier[]){
    const int numThread = omp_get_thread_num();

    for(int idxThread = 0 ; idxThread < numThreadsInvolved ; ++idxThread){
        if(idxThread + firstThread == numThread){
#pragma omp atomic write
            barrier[idxThread + firstThread] = -1;
        }
        while(true){
            int dataAreReady;
#pragma omp atomic read
            dataAreReady = barrier[idxThread + firstThread];
            if(dataAreReady == -1){
                break;
            }
        }
    }

    // Already in good shape
    if(centerPosition == 0 || centerPosition == sizeArray || array[centerPosition-1] <= array[centerPosition]){
#pragma omp critical(out)
std::cout << omp_get_thread_num() << "] nothing to do array " << array << " centerPosition " << centerPosition << " " <<
          array[centerPosition-1] << " " << array[centerPosition] << std::endl;// TODO

        for(int idxThread = 0 ; idxThread < numThreadsInvolved ; ++idxThread){
            if(idxThread + firstThread == numThread){
        #pragma omp atomic write
                barrier[idxThread + firstThread] = 0;
            }
            while(true){
                int dataAreReady;
        #pragma omp atomic read
                dataAreReady = barrier[idxThread + firstThread];
                if(dataAreReady == 0){
                    break;
                }
            }
        }

        return;
    }

#pragma omp critical(out)
std::cout << omp_get_thread_num() << "] work to do array " << array << " centerPosition " << centerPosition << " " <<
          array[centerPosition-1] << " " << array[centerPosition] << std::endl;// TODO

    for(int idxThread = 0 ; idxThread < numThreadsInvolved ; ++idxThread){
        if(idxThread + firstThread == numThread){
    #pragma omp atomic write
            barrier[idxThread + firstThread] = -2;
        }
        while(true){
            int dataAreReady;
    #pragma omp atomic read
            dataAreReady = barrier[idxThread + firstThread];
            if(dataAreReady == -2){
                break;
            }
        }
    }

    if(numThread == firstThread){
#pragma omp critical(out)
std::cout << omp_get_thread_num() << "] LEAD" << std::endl;// TODO
        const int depthLimite = ffs(numThreadsInvolved) - 1;
#pragma omp atomic write
        barrier[numThread] = 1;

        parallelMergeInPlaceCore<NumType>(array, 0, centerPosition, sizeArray, 0, depthLimite,
                                          intervals, barrier);
    }
    else{
//#pragma omp critical(out)
//std::cout << omp_get_thread_num() << "] wait my turn" << std::endl;// TODO

        while(true){
            int myDataAreReady;
#pragma omp atomic read
            myDataAreReady = barrier[numThread];
            if(myDataAreReady == 1){
                break;
            }
        }

        #pragma omp critical(out)
        std::cout << omp_get_thread_num() << "] GOO" << std::endl;// TODO
//        #pragma omp critical(out)
//        std::cout << omp_get_thread_num() << "] intervals[numThread].array = " << intervals[numThread].array << std::endl;// TODO
//        #pragma omp critical(out)
//        std::cout << omp_get_thread_num() << "] intervals[numThread].currentStart = " << intervals[numThread].currentStart << std::endl;// TODO
//        #pragma omp critical(out)
//        std::cout << omp_get_thread_num() << "] intervals[numThread].currentMiddle = " << intervals[numThread].currentMiddle << std::endl;// TODO
//        #pragma omp critical(out)
//        std::cout << omp_get_thread_num() << "] intervals[numThread].currentEnd = " << intervals[numThread].currentEnd << std::endl;// TODO
//        #pragma omp critical(out)
//        std::cout << omp_get_thread_num() << "] intervals[numThread].level = " << intervals[numThread].level << std::endl;// TODO
//        #pragma omp critical(out)
//        std::cout << omp_get_thread_num() << "] intervals[numThread].depthLimite = " << intervals[numThread].depthLimite << std::endl;// TODO

        parallelMergeInPlaceCore<NumType>(intervals[numThread].array,
                                          intervals[numThread].currentStart,
                                          intervals[numThread].currentMiddle,
                                          intervals[numThread].currentEnd,
                                          intervals[numThread].level,
                                          intervals[numThread].depthLimite,
                                          intervals, barrier);
    }

#pragma omp critical(out)
std::cout << omp_get_thread_num() << "] done wait other" << std::endl;// TODO

    for(int idxThread = 0 ; idxThread < numThreadsInvolved ; ++idxThread){
        if(idxThread + firstThread == numThread){
#pragma omp atomic write
            barrier[idxThread + firstThread] = 0;
        }
        while(true){
            int dataAreReady;
#pragma omp atomic read
            dataAreReady = barrier[idxThread + firstThread];
            if(dataAreReady == 0){
                break;
            }
        }
    }

#pragma omp critical(out)
std::cout << omp_get_thread_num() << "] leave" << std::endl;// TODO
}

}

#endif
