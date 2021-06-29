// Copyright 2015 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2015 National ICT Australia (NICTA)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------


#include <armadillo>
#include <complex>
#include "catch.hpp"

using namespace arma;
using namespace std;



TEST_CASE("fn_hess_non_square")
  {
  mat A(5, 6, fill::ones);
  mat U, H;
  REQUIRE_THROWS( hess(U, H, A) );
  }


/*****************  tests for real matrix  ****************/

TEST_CASE("fn_hess_empty")
  {
  mat A(1, 1);
  A.reset();
  mat U, H, H1, H2;
  
  hess(U, H, A);
  H1 = hess(A);
  hess(H2, A);
  
  REQUIRE(  U.is_empty() == true );
  REQUIRE(  H.is_empty() == true );
  REQUIRE( H1.is_empty() == true );
  REQUIRE( H2.is_empty() == true );
  }



TEST_CASE("fn_hess_1")
  {
  mat A(1, 1);
  A(0, 0) = 0.061198;
  mat U, H, H1, H2;
  
  hess(U, H, A);
  H1 = hess(A);
  hess(H2, A);
  
  REQUIRE( U(0, 0) == Approx(1.0) );
  REQUIRE( H(0, 0) == Approx(0.061198) );
  REQUIRE( H1(0, 0) == Approx(0.061198) );
  REQUIRE( H2(0, 0) == Approx(0.061198) );
  }



TEST_CASE("fn_hess_2")
  {
  mat A =
    "\
     0.061198   0.201990;\
     0.437242   0.058956;\
    ";
  mat U, H, H1, H2;
  
  hess(U, H, A);
  H1 = hess(A);
  hess(H2, A);
  
  REQUIRE( U(0, 0) == Approx(1.0) );
  REQUIRE( U(0, 1) == Approx(0.0) );
  REQUIRE( U(1, 0) == Approx(0.0) );
  REQUIRE( U(1, 1) == Approx(1.0) );
  
  REQUIRE( H(0, 0) == Approx(0.061198) );
  REQUIRE( H(0, 1) == Approx(0.201990) );
  REQUIRE( H(1, 0) == Approx(0.437242) );
  REQUIRE( H(1, 1) == Approx(0.058956) );
  
  REQUIRE( H1(0, 0) == Approx(0.061198) );
  REQUIRE( H1(0, 1) == Approx(0.201990) );
  REQUIRE( H1(1, 0) == Approx(0.437242) );
  REQUIRE( H1(1, 1) == Approx(0.058956) );
  
  REQUIRE( H2(0, 0) == Approx(0.061198) );
  REQUIRE( H2(0, 1) == Approx(0.201990) );
  REQUIRE( H2(1, 0) == Approx(0.437242) );
  REQUIRE( H2(1, 1) == Approx(0.058956) );
  }



TEST_CASE("fn_hess_3")
  {
  mat A =
    "\
      0.061198   0.201990   0.019678;\
      0.437242   0.058956  -0.149362;\
     -0.492474  -0.031309   0.314156;\
    ";
  mat U, H, H1, H2;
  
  hess(U, H, A);
  H1 = hess(A);
  hess(H2, A);
  
  REQUIRE( U(0, 0) == Approx(1.0) );
  REQUIRE( U(0, 1) == Approx(0.0) );
  REQUIRE( U(0, 2) == Approx(0.0) );
  
  REQUIRE( U(1, 0) == Approx( 0.0) );
  REQUIRE( U(1, 1) == Approx(-0.663928864062532) );
  REQUIRE( U(1, 2) == Approx( 0.747795736457915) );
  
  REQUIRE( U(2, 0) == Approx( 0.0) );
  REQUIRE( U(2, 1) == Approx( 0.747795736457915) );
  REQUIRE( U(2, 2) == Approx( 0.663928864062532) );
  
  
  REQUIRE( H(0, 0) == Approx( 0.061198000000000) );
  REQUIRE( H(0, 1) == Approx(-0.119391866749972) );
  REQUIRE( H(0, 2) == Approx( 0.164112052994157) );
  
  REQUIRE( H(1, 0) == Approx(-0.658567541896805) );
  REQUIRE( H(1, 1) == Approx( 0.291363559380149) );
  REQUIRE( H(1, 2) == Approx( 0.175033560375766) );
  
  REQUIRE( H(2, 0) == Approx( 0.0) );
  REQUIRE( H(2, 1) == Approx( 0.056980560375766) );
  REQUIRE( H(2, 2) == Approx( 0.081748440619851) );
  
  
  REQUIRE( H1(0, 0) == Approx( 0.061198000000000) );
  REQUIRE( H1(0, 1) == Approx(-0.119391866749972) );
  REQUIRE( H1(0, 2) == Approx( 0.164112052994157) );
  
  REQUIRE( H1(1, 0) == Approx(-0.658567541896805) );
  REQUIRE( H1(1, 1) == Approx( 0.291363559380149) );
  REQUIRE( H1(1, 2) == Approx( 0.175033560375766) );
  
  REQUIRE( H1(2, 0) == Approx( 0.0) );
  REQUIRE( H1(2, 1) == Approx( 0.056980560375766) );
  REQUIRE( H1(2, 2) == Approx( 0.081748440619851) );
  
  
  REQUIRE( H2(0, 0) == Approx( 0.061198000000000) );
  REQUIRE( H2(0, 1) == Approx(-0.119391866749972) );
  REQUIRE( H2(0, 2) == Approx( 0.164112052994157) );
  
  REQUIRE( H2(1, 0) == Approx(-0.658567541896805) );
  REQUIRE( H2(1, 1) == Approx( 0.291363559380149) );
  REQUIRE( H2(1, 2) == Approx( 0.175033560375766) );
  
  REQUIRE( H2(2, 0) == Approx( 0.0) );
  REQUIRE( H2(2, 1) == Approx( 0.056980560375766) );
  REQUIRE( H2(2, 2) == Approx( 0.081748440619851) );
  }




TEST_CASE("fn_hess_4")
  {
  mat A =
    "\
      0.061198   0.201990   0.019678  -0.493936;\
      0.437242   0.058956  -0.149362  -0.045465;\
     -0.492474  -0.031309   0.314156   0.419733;\
      0.336352   0.411541   0.458476  -0.393139;\
    ";
  mat U, H, H1, H2;
  
  hess(U, H, A);
  H1 = hess(A);
  hess(H2, A);
  
  REQUIRE( U(0, 0) == Approx(1.0) );
  REQUIRE( U(0, 1) == Approx(0.0) );
  REQUIRE( U(0, 2) == Approx(0.0) );
  REQUIRE( U(0, 3) == Approx(0.0) );
  
  REQUIRE( U(1, 0) == Approx( 0.0) );
  REQUIRE( U(1, 1) == Approx(-0.591275924818639) );
  REQUIRE( U(1, 2) == Approx(-0.462981984254642) );
  REQUIRE( U(1, 3) == Approx( 0.660333599770220) );
  
  REQUIRE( U(2, 0) == Approx( 0.0) );
  REQUIRE( U(2, 1) == Approx( 0.665965345962041) );
  REQUIRE( U(2, 2) == Approx( 0.181491258046004) );
  REQUIRE( U(2, 3) == Approx( 0.723568297557693) );
  
  REQUIRE( U(3, 0) == Approx( 0.0) );
  REQUIRE( U(3, 1) == Approx(-0.454843861899358) );
  REQUIRE( U(3, 2) == Approx( 0.867587808529208) );
  REQUIRE( U(3, 3) == Approx( 0.201018545870685) );
  
  
  REQUIRE( H(0, 0) == Approx( 0.061198000000000) );
  REQUIRE( H(0, 1) == Approx( 0.118336799794845) );
  REQUIRE( H(0, 2) == Approx(-0.518479197817449) );
  REQUIRE( H(0, 3) == Approx( 0.048328864303744) );
  
  REQUIRE( H(1, 0) == Approx(-0.739488928344434) );
  REQUIRE( H(1, 1) == Approx(-0.017815019577445) );
  REQUIRE( H(1, 2) == Approx( 0.549585804168668) );
  REQUIRE( H(1, 3) == Approx( 0.001541438669749) );
  
  REQUIRE( H(2, 0) == Approx( 0.0) );
  REQUIRE( H(2, 1) == Approx( 0.268224897826587) );
  REQUIRE( H(2, 2) == Approx(-0.266514530817371) );
  REQUIRE( H(2, 3) == Approx( 0.544078897369960) );
  
  REQUIRE( H(3, 0) == Approx( 0.0) );
  REQUIRE( H(3, 1) == Approx( 0.0) );
  REQUIRE( H(3, 2) == Approx( 0.163125252889179) );
  REQUIRE( H(3, 3) == Approx( 0.264302550394816) );
  
  
  REQUIRE( H1(0, 0) == Approx( 0.061198000000000) );
  REQUIRE( H1(0, 1) == Approx( 0.118336799794845) );
  REQUIRE( H1(0, 2) == Approx(-0.518479197817449) );
  REQUIRE( H1(0, 3) == Approx( 0.048328864303744) );
  
  REQUIRE( H1(1, 0) == Approx(-0.739488928344434) );
  REQUIRE( H1(1, 1) == Approx(-0.017815019577445) );
  REQUIRE( H1(1, 2) == Approx( 0.549585804168668) );
  REQUIRE( H1(1, 3) == Approx( 0.001541438669749) );
  
  REQUIRE( H1(2, 0) == Approx( 0.0) );
  REQUIRE( H1(2, 1) == Approx( 0.268224897826587) );
  REQUIRE( H1(2, 2) == Approx(-0.266514530817371) );
  REQUIRE( H1(2, 3) == Approx( 0.544078897369960) );
  
  REQUIRE( H1(3, 0) == Approx( 0.0) );
  REQUIRE( H1(3, 1) == Approx( 0.0) );
  REQUIRE( H1(3, 2) == Approx( 0.163125252889179) );
  REQUIRE( H1(3, 3) == Approx( 0.264302550394816) );
  
  
  REQUIRE( H2(0, 0) == Approx( 0.061198000000000) );
  REQUIRE( H2(0, 1) == Approx( 0.118336799794845) );
  REQUIRE( H2(0, 2) == Approx(-0.518479197817449) );
  REQUIRE( H2(0, 3) == Approx( 0.048328864303744) );
  
  REQUIRE( H2(1, 0) == Approx(-0.739488928344434) );
  REQUIRE( H2(1, 1) == Approx(-0.017815019577445) );
  REQUIRE( H2(1, 2) == Approx( 0.549585804168668) );
  REQUIRE( H2(1, 3) == Approx( 0.001541438669749) );
  
  REQUIRE( H2(2, 0) == Approx( 0.0) );
  REQUIRE( H2(2, 1) == Approx( 0.268224897826587) );
  REQUIRE( H2(2, 2) == Approx(-0.266514530817371) );
  REQUIRE( H2(2, 3) == Approx( 0.544078897369960) );
  
  REQUIRE( H2(3, 0) == Approx( 0.0) );
  REQUIRE( H2(3, 1) == Approx( 0.0) );
  REQUIRE( H2(3, 2) == Approx( 0.163125252889179) );
  REQUIRE( H2(3, 3) == Approx( 0.264302550394816) );
  }



/*****************  tests for complex matrix  ****************/

TEST_CASE("fn_hess_cx_empty")
  {
  cx_mat A(1, 1);
  A.reset();
  cx_mat U, H, H1, H2;
  
  hess(U, H, A);
  H1 = hess(A);
  hess(H2, A);
  
  REQUIRE(  U.is_empty() == true );
  REQUIRE(  H.is_empty() == true );
  REQUIRE( H1.is_empty() == true );
  REQUIRE( H2.is_empty() == true );
  }



TEST_CASE("fn_hess_cx_1")
  {
  cx_mat A(1, 1);
  A(0, 0) = complex<double>(0.061198, 1.012234);
  cx_mat U, H, H1, H2;
  
  hess(U, H, A);
  H1 = hess(A);
  hess(H2, A);
  
  REQUIRE( U(0, 0).real() == Approx(1.0) );
  REQUIRE( U(0, 0).imag() == Approx(0.0) );
  
  REQUIRE( H(0, 0).real() == Approx(0.061198) );
  REQUIRE( H(0, 0).imag() == Approx(1.012234) );
  
  REQUIRE( H1(0, 0).real() == Approx(0.061198) );
  REQUIRE( H1(0, 0).imag() == Approx(1.012234) );
  
  REQUIRE( H2(0, 0).real() == Approx(0.061198) );
  REQUIRE( H2(0, 0).imag() == Approx(1.012234) );
  }



TEST_CASE("fn_hess_cx_2")
  {
  mat B =
    "\
     0.061198   0.201990;\
     0.437242   0.058956;\
    ";
  cx_mat A(B, B*B);
  cx_mat U, H, H1, H2;
  
  hess(U, H, A);
  H1 = hess(A);
  hess(H2, A);
  
  REQUIRE( U(0, 0).real() == Approx(1.0) );
  REQUIRE( U(0, 0).imag() == Approx(0.0) );
  REQUIRE( U(0, 1).real() == Approx(0.0) );
  REQUIRE( U(0, 1).imag() == Approx(0.0) );
  REQUIRE( U(1, 0).real() == Approx(0.0) );
  REQUIRE( U(1, 0).imag() == Approx(0.0) );
  REQUIRE( U(1, 1).real() == Approx(1.0) );
  REQUIRE( U(1, 1).imag() == Approx(0.0) );
  
  REQUIRE( H(0, 0).real() == Approx( 0.061198000000000) );
  REQUIRE( H(0, 0).imag() == Approx( 0.092063706784000) );
  REQUIRE( H(0, 1).real() == Approx( 0.201990000000000) );
  REQUIRE( H(0, 1).imag() == Approx( 0.024269906460000) );
  REQUIRE( H(1, 0).real() == Approx( 0.437242000000000) );
  REQUIRE( H(1, 0).imag() == Approx( 0.052536375268000) );
  REQUIRE( H(1, 1).real() == Approx( 0.058956000000000) );
  REQUIRE( H(1, 1).imag() == Approx( 0.091794321516000) );
  
  REQUIRE( H1(0, 0).real() == Approx( 0.061198000000000) );
  REQUIRE( H1(0, 0).imag() == Approx( 0.092063706784000) );
  REQUIRE( H1(0, 1).real() == Approx( 0.201990000000000) );
  REQUIRE( H1(0, 1).imag() == Approx( 0.024269906460000) );
  REQUIRE( H1(1, 0).real() == Approx( 0.437242000000000) );
  REQUIRE( H1(1, 0).imag() == Approx( 0.052536375268000) );
  REQUIRE( H1(1, 1).real() == Approx( 0.058956000000000) );
  REQUIRE( H1(1, 1).imag() == Approx( 0.091794321516000) );
  
  REQUIRE( H2(0, 0).real() == Approx( 0.061198000000000) );
  REQUIRE( H2(0, 0).imag() == Approx( 0.092063706784000) );
  REQUIRE( H2(0, 1).real() == Approx( 0.201990000000000) );
  REQUIRE( H2(0, 1).imag() == Approx( 0.024269906460000) );
  REQUIRE( H2(1, 0).real() == Approx( 0.437242000000000) );
  REQUIRE( H2(1, 0).imag() == Approx( 0.052536375268000) );
  REQUIRE( H2(1, 1).real() == Approx( 0.058956000000000) );
  REQUIRE( H2(1, 1).imag() == Approx( 0.091794321516000) );
  }



TEST_CASE("fn_hess_cx_3")
  {
  mat B =
    "\
      0.061198   0.201990   0.019678;\
      0.437242   0.058956  -0.149362;\
     -0.492474  -0.031309   0.314156;\
    ";
  cx_mat A(B, B*B);
  cx_mat U, H, H1, H2;
  
  hess(U, H, A);
  H1 = hess(A);
  hess(H2, A);
  
  REQUIRE( U(0, 0).real() == Approx(1.0) );
  REQUIRE( U(0, 0).imag() == Approx(0.0) );
  REQUIRE( U(0, 1).real() == Approx(0.0) );
  REQUIRE( U(0, 1).imag() == Approx(0.0) );
  REQUIRE( U(0, 2).real() == Approx(0.0) );
  REQUIRE( U(0, 2).imag() == Approx(0.0) );
  
  REQUIRE( U(1, 0).real() == Approx( 0.0) );
  REQUIRE( U(1, 0).imag() == Approx( 0.0) );
  REQUIRE( U(1, 1).real() == Approx(-0.625250908290361) );
  REQUIRE( U(1, 1).imag() == Approx(-0.180311900237219) );
  REQUIRE( U(1, 2).real() == Approx(-0.694923841863332) );
  REQUIRE( U(1, 2).imag() == Approx(-0.305989827159056) );
  
  REQUIRE( U(2, 0).real() == Approx( 0.0) );
  REQUIRE( U(2, 0).imag() == Approx( 0.0) );
  REQUIRE( U(2, 1).real() == Approx( 0.704232017531224) );
  REQUIRE( U(2, 1).imag() == Approx( 0.283912285396078) );
  REQUIRE( U(2, 2).real() == Approx(-0.565610163671470) );
  REQUIRE( U(2, 2).imag() == Approx(-0.321770449912063) );
  
  
  REQUIRE( H(0, 0).real() == Approx( 0.061198000000000) );
  REQUIRE( H(0, 0).imag() == Approx( 0.082372803412000) );
  REQUIRE( H(0, 1).real() == Approx(-0.101702999021493) );
  REQUIRE( H(0, 1).imag() == Approx(-0.061668749553784) );
  REQUIRE( H(0, 2).real() == Approx(-0.151590948501704) );
  REQUIRE( H(0, 2).imag() == Approx(-0.071689748472419) );
  
  REQUIRE( H(1, 0).real() == Approx(-0.699306461138236) );
  REQUIRE( H(1, 0).imag() == Approx( 0.0) );
  REQUIRE( H(1, 1).real() == Approx( 0.298129546829246) );
  REQUIRE( H(1, 1).imag() == Approx( 0.178624769103627) );
  REQUIRE( H(1, 2).real() == Approx(-0.165941859233838) );
  REQUIRE( H(1, 2).imag() == Approx( 0.014927427092653) );
  
  REQUIRE( H(2, 0).real() == Approx( 0.0) );
  REQUIRE( H(2, 0).imag() == Approx( 0.0) );
  REQUIRE( H(2, 1).real() == Approx(-0.061767777059231) );
  REQUIRE( H(2, 1).imag() == Approx( 0.0) );
  REQUIRE( H(2, 2).real() == Approx( 0.074982453170754) );
  REQUIRE( H(2, 2).imag() == Approx( 0.011525391092373) );
  
  
  REQUIRE( H1(0, 0).real() == Approx( 0.061198000000000) );
  REQUIRE( H1(0, 0).imag() == Approx( 0.082372803412000) );
  REQUIRE( H1(0, 1).real() == Approx(-0.101702999021493) );
  REQUIRE( H1(0, 1).imag() == Approx(-0.061668749553784) );
  REQUIRE( H1(0, 2).real() == Approx(-0.151590948501704) );
  REQUIRE( H1(0, 2).imag() == Approx(-0.071689748472419) );
  
  REQUIRE( H1(1, 0).real() == Approx(-0.699306461138236) );
  REQUIRE( H1(1, 0).imag() == Approx( 0.0) );
  REQUIRE( H1(1, 1).real() == Approx( 0.298129546829246) );
  REQUIRE( H1(1, 1).imag() == Approx( 0.178624769103627) );
  REQUIRE( H1(1, 2).real() == Approx(-0.165941859233838) );
  REQUIRE( H1(1, 2).imag() == Approx( 0.014927427092653) );
  
  REQUIRE( H1(2, 0).real() == Approx( 0.0) );
  REQUIRE( H1(2, 0).imag() == Approx( 0.0) );
  REQUIRE( H1(2, 1).real() == Approx(-0.061767777059231) );
  REQUIRE( H1(2, 1).imag() == Approx( 0.0) );
  REQUIRE( H1(2, 2).real() == Approx( 0.074982453170754) );
  REQUIRE( H1(2, 2).imag() == Approx( 0.011525391092373) );
  
  
  REQUIRE( H2(0, 0).real() == Approx( 0.061198000000000) );
  REQUIRE( H2(0, 0).imag() == Approx( 0.082372803412000) );
  REQUIRE( H2(0, 1).real() == Approx(-0.101702999021493) );
  REQUIRE( H2(0, 1).imag() == Approx(-0.061668749553784) );
  REQUIRE( H2(0, 2).real() == Approx(-0.151590948501704) );
  REQUIRE( H2(0, 2).imag() == Approx(-0.071689748472419) );
  
  REQUIRE( H2(1, 0).real() == Approx(-0.699306461138236) );
  REQUIRE( H2(1, 0).imag() == Approx( 0.0) );
  REQUIRE( H2(1, 1).real() == Approx( 0.298129546829246) );
  REQUIRE( H2(1, 1).imag() == Approx( 0.178624769103627) );
  REQUIRE( H2(1, 2).real() == Approx(-0.165941859233838) );
  REQUIRE( H2(1, 2).imag() == Approx( 0.014927427092653) );
  
  REQUIRE( H2(2, 0).real() == Approx( 0.0) );
  REQUIRE( H2(2, 0).imag() == Approx( 0.0) );
  REQUIRE( H2(2, 1).real() == Approx(-0.061767777059231) );
  REQUIRE( H2(2, 1).imag() == Approx( 0.0) );
  REQUIRE( H2(2, 2).real() == Approx( 0.074982453170754) );
  REQUIRE( H2(2, 2).imag() == Approx( 0.011525391092373) );
  }




TEST_CASE("fn_hess_cx_4")
  {
  mat B =
    "\
      0.061198   0.201990   0.019678  -0.493936;\
      0.437242   0.058956  -0.149362  -0.045465;\
     -0.492474  -0.031309   0.314156   0.419733;\
      0.336352   0.411541   0.458476  -0.393139;\
    ";
  cx_mat A(B, B*B);
  cx_mat U, H, H1, H2;
  
  hess(U, H, A*A);
  H1 = hess(A*A);
  hess(H2, A*A);
  
  REQUIRE( U(0, 0).real() == Approx(1.0) );
  REQUIRE( U(0, 0).imag() == Approx(0.0) );
  REQUIRE( U(0, 1).real() == Approx(0.0) );
  REQUIRE( U(0, 1).imag() == Approx(0.0) );
  REQUIRE( U(0, 2).real() == Approx(0.0) );
  REQUIRE( U(0, 2).imag() == Approx(0.0) );
  REQUIRE( U(0, 3).real() == Approx(0.0) );
  REQUIRE( U(0, 3).imag() == Approx(0.0) );
  
  REQUIRE( U(1, 0).real() == Approx( 0.0) );
  REQUIRE( U(1, 0).imag() == Approx( 0.0) );
  REQUIRE( U(1, 1).real() == Approx(-0.310409361344421) );
  REQUIRE( U(1, 1).imag() == Approx( 0.134965522927510) );
  REQUIRE( U(1, 2).real() == Approx( 0.368370931495079) );
  REQUIRE( U(1, 2).imag() == Approx( 0.620286967761253) );
  REQUIRE( U(1, 3).real() == Approx(-0.461565151978241) );
  REQUIRE( U(1, 3).imag() == Approx( 0.389788251419862) );
  
  REQUIRE( U(2, 0).real() == Approx( 0.0) );
  REQUIRE( U(2, 0).imag() == Approx( 0.0) );
  REQUIRE( U(2, 1).real() == Approx( 0.090510343531288) );
  REQUIRE( U(2, 1).imag() == Approx( 0.435448214446087) );
  REQUIRE( U(2, 2).real() == Approx(-0.629572243863963) );
  REQUIRE( U(2, 2).imag() == Approx( 0.277252591049466) );
  REQUIRE( U(2, 3).real() == Approx( 0.331725833624923) );
  REQUIRE( U(2, 3).imag() == Approx( 0.467889401534022) );
  
  REQUIRE( U(3, 0).real() == Approx( 0.0) );
  REQUIRE( U(3, 0).imag() == Approx( 0.0) );
  REQUIRE( U(3, 1).real() == Approx( 0.662749913792672) );
  REQUIRE( U(3, 1).imag() == Approx(-0.498383003349854) );
  REQUIRE( U(3, 2).real() == Approx(-0.073218600583049) );
  REQUIRE( U(3, 2).imag() == Approx(-0.030915392543373) );
  REQUIRE( U(3, 3).real() == Approx(-0.297561059397637) );
  REQUIRE( U(3, 3).imag() == Approx( 0.466387847936125) );
  
  
  REQUIRE( H(0, 0).real() == Approx(-0.059498334460944) );
  REQUIRE( H(0, 0).imag() == Approx( 0.187834910202221) );
  REQUIRE( H(0, 1).real() == Approx(-0.017930467829804) );
  REQUIRE( H(0, 1).imag() == Approx(-0.366928547670200) );
  REQUIRE( H(0, 2).real() == Approx(-0.021913405453089) );
  REQUIRE( H(0, 2).imag() == Approx(-0.128142818524165) );
  REQUIRE( H(0, 3).real() == Approx( 0.012590549436907) );
  REQUIRE( H(0, 3).imag() == Approx(-0.036787529849029) );
  
  REQUIRE( H(1, 0).real() == Approx(-0.212856818153491) );
  REQUIRE( H(1, 0).imag() == Approx( 0.0) );
  REQUIRE( H(1, 1).real() == Approx( 0.173480548915683) );
  REQUIRE( H(1, 1).imag() == Approx(-0.119570582029397) );
  REQUIRE( H(1, 2).real() == Approx(-0.098222486822866) );
  REQUIRE( H(1, 2).imag() == Approx(-0.073492477972392) );
  REQUIRE( H(1, 3).real() == Approx(-0.088126641335837) );
  REQUIRE( H(1, 3).imag() == Approx( 0.107905518898551) );
  
  REQUIRE( H(2, 0).real() == Approx( 0.0) );
  REQUIRE( H(2, 0).imag() == Approx( 0.0) );
  REQUIRE( H(2, 1).real() == Approx( 0.125544511009417) );
  REQUIRE( H(2, 1).imag() == Approx( 0.0) );
  REQUIRE( H(2, 2).real() == Approx( 0.374057080595739) );
  REQUIRE( H(2, 2).imag() == Approx( 0.061223114296791) );
  REQUIRE( H(2, 3).real() == Approx( 0.231175819260595) );
  REQUIRE( H(2, 3).imag() == Approx(-0.224564151240434) );
  
  REQUIRE( H(3, 0).real() == Approx( 0.0) );
  REQUIRE( H(3, 0).imag() == Approx( 0.0) );
  REQUIRE( H(3, 1).real() == Approx( 0.0) );
  REQUIRE( H(3, 1).imag() == Approx( 0.0) );
  REQUIRE( H(3, 2).real() == Approx(-0.238973358869022) );
  REQUIRE( H(3, 2).imag() == Approx( 0.0) );
  REQUIRE( H(3, 3).real() == Approx(-0.101771291133878) );
  REQUIRE( H(3, 3).imag() == Approx( 0.212030655387598) );
  
  
  REQUIRE( H1(0, 0).real() == Approx(-0.059498334460944) );
  REQUIRE( H1(0, 0).imag() == Approx( 0.187834910202221) );
  REQUIRE( H1(0, 1).real() == Approx(-0.017930467829804) );
  REQUIRE( H1(0, 1).imag() == Approx(-0.366928547670200) );
  REQUIRE( H1(0, 2).real() == Approx(-0.021913405453089) );
  REQUIRE( H1(0, 2).imag() == Approx(-0.128142818524165) );
  REQUIRE( H1(0, 3).real() == Approx( 0.012590549436907) );
  REQUIRE( H1(0, 3).imag() == Approx(-0.036787529849029) );
  
  REQUIRE( H1(1, 0).real() == Approx(-0.212856818153491) );
  REQUIRE( H1(1, 0).imag() == Approx( 0.0) );
  REQUIRE( H1(1, 1).real() == Approx( 0.173480548915683) );
  REQUIRE( H1(1, 1).imag() == Approx(-0.119570582029397) );
  REQUIRE( H1(1, 2).real() == Approx(-0.098222486822866) );
  REQUIRE( H1(1, 2).imag() == Approx(-0.073492477972392) );
  REQUIRE( H1(1, 3).real() == Approx(-0.088126641335837) );
  REQUIRE( H1(1, 3).imag() == Approx( 0.107905518898551) );
  
  REQUIRE( H1(2, 0).real() == Approx( 0.0) );
  REQUIRE( H1(2, 0).imag() == Approx( 0.0) );
  REQUIRE( H1(2, 1).real() == Approx( 0.125544511009417) );
  REQUIRE( H1(2, 1).imag() == Approx( 0.0) );
  REQUIRE( H1(2, 2).real() == Approx( 0.374057080595739) );
  REQUIRE( H1(2, 2).imag() == Approx( 0.061223114296791) );
  REQUIRE( H1(2, 3).real() == Approx( 0.231175819260595) );
  REQUIRE( H1(2, 3).imag() == Approx(-0.224564151240434) );
  
  REQUIRE( H1(3, 0).real() == Approx( 0.0) );
  REQUIRE( H1(3, 0).imag() == Approx( 0.0) );
  REQUIRE( H1(3, 1).real() == Approx( 0.0) );
  REQUIRE( H1(3, 1).imag() == Approx( 0.0) );
  REQUIRE( H1(3, 2).real() == Approx(-0.238973358869022) );
  REQUIRE( H1(3, 2).imag() == Approx( 0.0) );
  REQUIRE( H1(3, 3).real() == Approx(-0.101771291133878) );
  REQUIRE( H1(3, 3).imag() == Approx( 0.212030655387598) );
  
  
  REQUIRE( H2(0, 0).real() == Approx(-0.059498334460944) );
  REQUIRE( H2(0, 0).imag() == Approx( 0.187834910202221) );
  REQUIRE( H2(0, 1).real() == Approx(-0.017930467829804) );
  REQUIRE( H2(0, 1).imag() == Approx(-0.366928547670200) );
  REQUIRE( H2(0, 2).real() == Approx(-0.021913405453089) );
  REQUIRE( H2(0, 2).imag() == Approx(-0.128142818524165) );
  REQUIRE( H2(0, 3).real() == Approx( 0.012590549436907) );
  REQUIRE( H2(0, 3).imag() == Approx(-0.036787529849029) );
  
  REQUIRE( H2(1, 0).real() == Approx(-0.212856818153491) );
  REQUIRE( H2(1, 0).imag() == Approx( 0.0) );
  REQUIRE( H2(1, 1).real() == Approx( 0.173480548915683) );
  REQUIRE( H2(1, 1).imag() == Approx(-0.119570582029397) );
  REQUIRE( H2(1, 2).real() == Approx(-0.098222486822866) );
  REQUIRE( H2(1, 2).imag() == Approx(-0.073492477972392) );
  REQUIRE( H2(1, 3).real() == Approx(-0.088126641335837) );
  REQUIRE( H2(1, 3).imag() == Approx( 0.107905518898551) );
  
  REQUIRE( H2(2, 0).real() == Approx( 0.0) );
  REQUIRE( H2(2, 0).imag() == Approx( 0.0) );
  REQUIRE( H2(2, 1).real() == Approx( 0.125544511009417) );
  REQUIRE( H2(2, 1).imag() == Approx( 0.0) );
  REQUIRE( H2(2, 2).real() == Approx( 0.374057080595739) );
  REQUIRE( H2(2, 2).imag() == Approx( 0.061223114296791) );
  REQUIRE( H2(2, 3).real() == Approx( 0.231175819260595) );
  REQUIRE( H2(2, 3).imag() == Approx(-0.224564151240434) );
  
  REQUIRE( H2(3, 0).real() == Approx( 0.0) );
  REQUIRE( H2(3, 0).imag() == Approx( 0.0) );
  REQUIRE( H2(3, 1).real() == Approx( 0.0) );
  REQUIRE( H2(3, 1).imag() == Approx( 0.0) );
  REQUIRE( H2(3, 2).real() == Approx(-0.238973358869022) );
  REQUIRE( H2(3, 2).imag() == Approx( 0.0) );
  REQUIRE( H2(3, 3).real() == Approx(-0.101771291133878) );
  REQUIRE( H2(3, 3).imag() == Approx( 0.212030655387598) );
  }
