#ifndef _MY_OPENMP
#define _MY_OPENMP
#ifdef _OPENMP
  #include <omp.h>
#else
  // for machines with compilers void of openmp support
  #define omp_get_num_threads()  1
  #define omp_set_num_threads(a)     //
  #define omp_get_thread_num()   0
  #define omp_get_max_threads()  1
  #define omp_get_thread_limit() 1
  #define omp_get_num_procs()    1
  #define omp_set_nested(a)   // empty statement to remove the call
  #define omp_get_wtime()        0
  #define MY_OPENMP              0
#endif
#endif
