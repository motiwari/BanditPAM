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
#include "catch.hpp"

using namespace arma;


TEST_CASE("decomp_eig_pair_1")
  {
  mat A1 =
    "\
     0.840375529753905  -0.600326562133734  -2.138355269439939   0.124049800003193   2.908008030729362;\
    -0.888032082329010   0.489965321173948  -0.839588747336614   1.436696622718939   0.825218894228491;\
     0.100092833139322   0.739363123604474   1.354594328004644  -1.960899999365033   1.378971977916614;\
    -0.544528929990548   1.711887782981555  -1.072155288384252  -0.197698225974150  -1.058180257987362;\
     0.303520794649354  -0.194123535758265   0.960953869740567  -1.207845485259799  -0.468615581100624;\
    ";

  mat A2 =
    "\
    -0.272469409250187  -0.353849997774433   0.033479882244451   0.022889792751630  -0.979206305167302;\
     1.098424617888623  -0.823586525156853  -1.333677943428106  -0.261995434966092  -1.156401655664002;\
    -0.277871932787639  -1.577057022799202   1.127492278341590  -1.750212368446790  -0.533557109315987;\
     0.701541458163284   0.507974650905946   0.350179410603312  -0.285650971595330  -2.002635735883060;\
    -2.051816299911149   0.281984063670556  -0.299066030332982  -0.831366511567624   0.964229422631627;\
    ";
  
  cx_vec eigvals1 =
    {
    cx_double(-2.467066249890401, +0.000000000000000),
    cx_double( 1.483137782196390, +0.595028644066690),
    cx_double( 1.483137782196390, -0.595028644066690),
    cx_double(-0.646831879916377, +0.000000000000000),
    cx_double( 0.099992295916005, +0.000000000000000)
    };
  
  cx_vec eigvals2 = eig_pair(A1, A2);
  
  cx_vec eigvals3;
  bool status = eig_pair(eigvals3, A1, A2);
  
  cx_vec eigvals4;
  cx_mat eigvecs4;
  eig_pair(eigvals4, eigvecs4, A1, A2);
  
  cx_vec  eigvals5;
  cx_mat leigvecs5;
  cx_mat reigvecs5;
  eig_pair(eigvals5, leigvecs5, reigvecs5, A1, A2);
  
  cx_mat B  = A2 *            eigvecs4   * diagmat(eigvals4) *   inv( eigvecs4)     ;
  
  cx_mat Cl =      inv(trans(leigvecs5)) * diagmat(eigvals5) * trans(leigvecs5) * A2;
  cx_mat Cr = A2 *           reigvecs5   * diagmat(eigvals5) *   inv(reigvecs5)     ;
  
  REQUIRE( status == true );
  REQUIRE( accu(abs(eigvals2 - eigvals1)) == Approx(0.0).epsilon(0.0001) );
  REQUIRE( accu(abs(eigvals3 - eigvals1)) == Approx(0.0).epsilon(0.0001) );
  REQUIRE( accu(abs(eigvals4 - eigvals1)) == Approx(0.0).epsilon(0.0001) );
  REQUIRE( accu(abs(eigvals5 - eigvals1)) == Approx(0.0).epsilon(0.0001) );
  REQUIRE( accu(abs(A1       - B       )) == Approx(0.0).epsilon(0.0001) );
  REQUIRE( accu(abs(A1       - Cl      )) == Approx(0.0).epsilon(0.0001) );
  REQUIRE( accu(abs(A1       - Cr      )) == Approx(0.0).epsilon(0.0001) );
  }



TEST_CASE("decomp_eig_pair_2")
  {
  cx_mat A1 =
    {
    { cx_double( 0.520060101455458, +0.451679418928238), cx_double(-0.133217479507735, -1.361694470870754), cx_double(-0.293753597735416, +1.039090653504956), cx_double( 0.307535159238252, -0.195221197898754), cx_double(-1.332004421315247, +0.826062790211595) },
    { cx_double(-0.020027851642538, -0.130284653145721), cx_double(-0.714530163787158, +0.455029556444334), cx_double(-0.847926243637934, -1.117638683265208), cx_double(-1.257118359352053, -0.217606350143192), cx_double(-2.329867155805076, +1.526976686733373) },
    { cx_double(-0.034771086028483, +0.183689095861942), cx_double( 1.351385768426657, -0.848709379933659), cx_double(-1.120128301243728, +1.260658709120896), cx_double(-0.865468030554804, -0.303107621351741), cx_double(-1.449097292838739, +0.466914435684700) },
    { cx_double(-0.798163584564142, -0.476153016619074), cx_double(-0.224771056052584, -0.334886938964048), cx_double( 2.525999692118309, +0.660143141046978), cx_double(-0.176534114231451, +0.023045624425105), cx_double( 0.333510833065806, -0.209713338388737) },
    { cx_double( 1.018685282128575, +0.862021611556922), cx_double(-0.589029030720801, +0.552783345944550), cx_double( 1.655497592887346, -0.067865553542687), cx_double( 0.791416061628634, +0.051290355848775), cx_double( 0.391353604432901, +0.625190357087626) }
    };

  cx_mat A2 =
    {
    { cx_double( 0.183227263001437, -0.444627816446985), cx_double( 0.515246335524849, +0.391894209432449), cx_double(-0.532011376808821, -0.320575506600239), cx_double(-1.174212331456816, -1.066701398984750), cx_double(-1.064213412889327, -1.565056014150725) },
    { cx_double(-1.029767543566621, -0.155941035724769), cx_double( 0.261406324055383, -1.250678906826407), cx_double( 1.682103594663179, +0.012469041361618), cx_double(-0.192239517539275, +0.933728162671238), cx_double( 1.603457298120044, -0.084539479817724) },
    { cx_double( 0.949221831131023, +0.276068253931536), cx_double(-0.941485770955434, -0.947960922331432), cx_double(-0.875729346160017, -3.029177341404146), cx_double(-0.274070229932602, +0.350321001356112), cx_double( 1.234679146890778, +1.603946350602880) },
    { cx_double( 0.307061919146703, -0.261163645776479), cx_double(-0.162337672803828, -0.741106093940411), cx_double(-0.483815050110121, -0.457014640871583), cx_double( 1.530072514424096, -0.029005763708726), cx_double(-0.229626450963180, +0.098347774640108) },
    { cx_double( 0.135174942099456, +0.443421912904091), cx_double(-0.146054634331526, -0.507817550278174), cx_double(-0.712004549027422, +1.242448406390738), cx_double(-0.249024742513714, +0.182452167505983), cx_double(-1.506159703979719, +0.041373613489615) }
    };

  cx_vec eigvals1 =
    {
    cx_double(-0.567948485992280, -1.314594536444777),
    cx_double( 1.051873153748018, -0.162676262480913),
    cx_double( 0.610087089288344, +0.562148335263468),
    cx_double(-0.890023643973463, -0.352930772605452),
    cx_double(-0.365476750045154, +0.305826583179225)
    };

  cx_vec eigvals2 = eig_pair(A1, A2);

  cx_vec eigvals3;
  bool status = eig_pair(eigvals3, A1, A2);

  cx_vec eigvals4;
  cx_mat eigvecs4;
  eig_pair(eigvals4, eigvecs4, A1, A2);

  cx_vec  eigvals5;
  cx_mat leigvecs5;
  cx_mat reigvecs5;
  eig_pair(eigvals5, leigvecs5, reigvecs5, A1, A2);

  cx_mat B  = A2 *            eigvecs4   * diagmat(eigvals4) *   inv( eigvecs4)     ;

  cx_mat Cl =      inv(trans(leigvecs5)) * diagmat(eigvals5) * trans(leigvecs5) * A2;
  cx_mat Cr = A2 *           reigvecs5   * diagmat(eigvals5) *   inv(reigvecs5)     ;

  REQUIRE( status == true );
  REQUIRE( accu(abs(eigvals2 - eigvals1)) == Approx(0.0).epsilon(0.0001) );
  REQUIRE( accu(abs(eigvals3 - eigvals1)) == Approx(0.0).epsilon(0.0001) );
  REQUIRE( accu(abs(eigvals4 - eigvals1)) == Approx(0.0).epsilon(0.0001) );
  REQUIRE( accu(abs(eigvals5 - eigvals1)) == Approx(0.0).epsilon(0.0001) );
  REQUIRE( accu(abs(A1       - B       )) == Approx(0.0).epsilon(0.0001) );
  REQUIRE( accu(abs(A1       - Cl      )) == Approx(0.0).epsilon(0.0001) );
  REQUIRE( accu(abs(A1       - Cr      )) == Approx(0.0).epsilon(0.0001) );
  }
