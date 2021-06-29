#include <iostream>
#include <armadillo>

using namespace arma;

int
main()
  {
  std::cout << "*** smoke test start" << std::endl;
  
  uword N = 5;
  
  mat A = reshape(regspace(1, N*N), N, N);
  
  A.diag() += randu<vec>(N);
  
  mat B;
  
  bool status = expmat(B,A);
  
  A.print("A:");
  B.print("B:");
  
  std::cout << ((status) ? "*** smoke test okay" : "*** smoke test failed") << std::endl;
  
  return (status) ? 0 : -1;
  }
