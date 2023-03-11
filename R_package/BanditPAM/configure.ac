dnl Hacked from original source at https://github.com/dmlc/xgboost
AC_INIT(banditpam, 1.0)
AC_PREREQ(2.71)

: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  echo "could not determine R_HOME"
  exit 1
fi

CXX17=`"${R_HOME}/bin/R" CMD config CXX17`
CXX17STD=`"${R_HOME}/bin/R" CMD config CXX17STD`
CXX="${CXX17} ${CXX17STD}"
CXXFLAGS=`"${R_HOME}/bin/R" CMD config CXXFLAGS`

CC=`"${R_HOME}/bin/R" CMD config CC`
CFLAGS=`"${R_HOME}/bin/R" CMD config CFLAGS`
CPPFLAGS=`"${R_HOME}/bin/R" CMD config CPPFLAGS`

LDFLAGS=`"${R_HOME}/bin/R" CMD config LDFLAGS`
AC_LANG(C++)

OPENMP_CXXFLAGS=""
R_HAS_OPENMP='TRUE'

if test `uname -s` = "Linux"
then
  OPENMP_CXXFLAGS="\$(SHLIB_OPENMP_CXXFLAGS)"
fi

if test `uname -s` = "Darwin"
then
  if command -v brew &> /dev/null
  then
    HOMEBREW_LIBOMP_PREFIX=`brew --prefix libomp`
  else
    # Homebrew not found
    HOMEBREW_LIBOMP_PREFIX=''
  fi
  OPENMP_CXXFLAGS="-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include"
  OPENMP_LIB="-lomp -L${HOMEBREW_LIBOMP_PREFIX}/lib"
  ac_pkg_openmp=no
  AC_MSG_CHECKING([whether OpenMP will work in a package])
  AC_LANG_CONFTEST([AC_LANG_PROGRAM([[#include <omp.h>]], [[ return (omp_get_max_threads() <= 1); ]])])
  ${CXX} -o conftest conftest.cpp ${CPPFLAGS} ${LDFLAGS} ${OPENMP_LIB} ${OPENMP_CXXFLAGS} 2>/dev/null && ./conftest && ac_pkg_openmp=yes
  AC_MSG_RESULT([${ac_pkg_openmp}])
  if test "${ac_pkg_openmp}" = no; then
    OPENMP_CXXFLAGS=''
    OPENMP_LIB=''
    R_HAS_OPENMP='FALSE'
    echo '*****************************************************************************************'
    echo '         OpenMP is unavailable on this Mac OSX system. Speed may be suboptimal.          '
    echo '         To use all CPU cores, you should install OpenMP by running\n                    '
    echo '             brew install libomp                                                         '
    echo '*****************************************************************************************'
  fi
fi

echo ".bpam_has_openmp <- $R_HAS_OPENMP" > R/autoconfig.R

AC_SUBST(OPENMP_CXXFLAGS)
AC_SUBST(R_HAS_OPENMP)
AC_SUBST(OPENMP_LIB)
AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT()
