//
// Created by TESTER on 03/11/2023.
//

#ifndef TFG_FUNCTIONS_H
#define TFG_FUNCTIONS_H
#include <complex>
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include "functions.h"
#include <Eigen/Dense>
#include <chrono>
#include <unsupported/Eigen/MatrixFunctions>

using namespace std;
using namespace Eigen;

#define DEBUG false
#define PI M_PI
#define eu std::exp(1.0)
const complex<double> I(0.0,1.0);
#define vms vector<chrono::microseconds>

double SNR = 1; // positive
// vector<complex<double>> X = {1,1,1,1};
// vector<complex<double>> X = {1,2};
int sizeX = pow(2,6);
vector<double> Qq;

vector<complex<double>> X;
//vector<complex<double>> X;
// complex<double> I1 (0,2 * PI * 1 / 4), I2 (0,2 * PI * 2 / 4), I3 (0,2 * PI * 3 / 4);
// vector<complex<double>> X = {1,exp(I1), exp(I2), exp(I3)};
/* vector<complex<double>> X = {complex<double> (-1/sqrt(2),-1/sqrt(2) ),
                             complex<double> ( 1/sqrt(2),-1/sqrt(2) ),
                             complex<double> (-1/sqrt(2), 1/sqrt(2) ),
                             complex<double> ( 1/sqrt(2), 1/sqrt(2) )   };*/

double R;
unordered_map<int, vector<double>> all_hweights;
unordered_map<int, vector<double>> all_roots;
unordered_map<int, vector<double>> all_multhweights;

int n = 15; // todo: warning: temporary
void setN(int n_){n = n_;}

// -- MATRIX DEFINITIONS --
VectorXd Q_mat;
MatrixXd PI_mat;
MatrixXd W_mat;
VectorXcd X_mat(sizeX);
MatrixXd D_mat;

double low = n; // todo: warning: temporary: before was 17.0

vms e0_times;// = {std::chrono::microseconds(1), std::chrono::microseconds(2), std::chrono::microseconds(3)};
vms NAG_cc_times;
vms NAG_iid_times;
vms NAG_times;
vms GD_cc_times;
vms NAG_co_times;
vms GD_iid_times;
vms GD_co_times;
vms GD_ccomp_times;
vms gradient_e0_co_times;
vms fa_times;
vms gradient_f_times;
vms gradient_f_co_times;
vms fa_co_times;
vms e02_times;
vms E_0_co_times;
vms gradient_e0_times;
vms compute_hweights_times;
vms mult_newhweights_times;


unordered_map<string, vector<chrono::microseconds>> get_times(){
    return
            {{"e0_times", e0_times},// = {std::chrono::microseconds(1), std::chrono::microseconds(2), std::chrono::microseconds(3)};{NAG_cc_times, "NAG_cc_times",
             {"NAG_iid_times", NAG_iid_times},
             {"NAG_times", NAG_times},
             {"GD_cc_times", GD_cc_times},
             {"NAG_co_times", NAG_co_times},
             {"GD_iid_times", GD_iid_times},
             {"GD_co_times", GD_co_times},
             {"GD_ccomp_times", GD_ccomp_times},
             {"gradient_e0_co_times", gradient_e0_co_times},
             {"fa_times", fa_times},
             {"gradient_f_times", gradient_f_times},
             {"gradient_f_co_times", gradient_f_co_times},
             {"fa_co_times", fa_co_times},
             {"e02_times", e02_times},
             {"E_0_co_times", E_0_co_times},
             {"gradient_e0_times", gradient_e0_times},
             {"compute_hweights_times", compute_hweights_times},
             {"mult_newhweights_times", mult_newhweights_times}};
}

double abs_sq(std::complex<double> a) { return pow(real(a),2) + pow(imag(a),2); }

vector<double> getAllHweights(){
    return all_hweights[n];
}

vector<double> getAllRoots(){
    return all_roots[n];
}

vector<double> getAllMultHweights(){
    return all_multhweights[n];
}


unsigned long long factorial(unsigned int n){
    if (n == 0)
        return 1;
    return n * factorial(n - 1);
}

vector<double> Hroots(int n){ // returns the roots of the Hermite polynomial of degree n up to n = 40 with 16 digits of precision
    vector<double> roots;
    switch(n){ // very small n should not be used to approximate correctly
        case(1):  return {0};
        case(2):  return {-0.7071067811865475, 0.7071067811865475};
        case(3):  return {-1.224744871391589, 0, 1.224744871391589};
        case(4):  return {-1.650680123885785, -0.5246476232752904, 0.5246476232752904, 1.650680123885785};
        case(5):  return {-2.020182870456086, -0.9585724646138185, 0, 0.9585724646138185, 2.020182870456086};
        case(6):  return {-2.350604973674492, -1.3358490740136968, -0.4360774119276165, 0.4360774119276165, 1.3358490740136968, 2.350604973674492};
        case(7):  return {-2.651961356835233, -1.673551628767471, -0.8162878828589647, 0, 0.8162878828589647, 1.673551628767471, 2.651961356835233};
        case(8):  return {-2.930637420257244, -1.981656756695843, -1.1571937124467802, -0.3811869902073221, 0.3811869902073221, 1.1571937124467802, 1.981656756695843, 2.930637420257244};
        case(9):  return {-3.190993201781528, -2.266580584531843, -1.468553289216668, -0.7235510187528376, 0, 0.7235510187528376, 1.468553289216668, 2.266580584531843, 3.190993201781528};
        case(10): return {-3.436159118837738, -2.53273167423279, -1.756683649299882, -1.036610829789514, -0.3429013272237046, 0.3429013272237046, 1.036610829789514, 1.756683649299882, 2.53273167423279, 3.436159118837738};
        case(11): return {-3.668470846559583, -2.783290099781652, -2.025948015825755, -1.326557084494933, -0.6568095668820998, 0, 0.6568095668820998, 1.326557084494933, 2.025948015825755, 2.783290099781652, 3.668470846559583};
        case(12): return {-3.889724897869782, -3.02063702512089, -2.27950708050106, -1.597682635152605, -0.9477883912401637, -0.3142403762543591, 0.3142403762543591, 0.9477883912401637, 1.597682635152605, 2.27950708050106, 3.02063702512089, 3.889724897869782};
        case(13): return {-4.10133759617864, -3.24660897837241, -2.519735685678238, -1.853107651601512, -1.220055036590748, -0.6057638791710601, 0, 0.6057638791710601, 1.220055036590748, 1.853107651601512, 2.519735685678238, 3.24660897837241, 4.10133759617864};
        case(14): return {-4.304448570473632, -3.462656933602271, -2.748470724985403, -2.095183258507717, -1.476682731141141, -0.8787137873293994, -0.2917455106725621, 0.2917455106725621, 0.8787137873293994, 1.476682731141141, 2.095183258507717, 2.748470724985403, 3.462656933602271, 4.304448570473632};
        case(15): return {-4.499990707309392, -3.669950373404453, -2.967166927905603, -2.325732486173858, -1.719992575186489, -1.136115585210921, -0.5650695832555757, 0, 0.5650695832555757, 1.136115585210921, 1.719992575186489, 2.325732486173858, 2.967166927905603, 3.669950373404453, 4.499990707309392};
        case(16): return {-4.688738939305818, -3.869447904860123, -3.176999161979956, -2.546202157847481, -1.951787990916254, -1.380258539198881, -0.8229514491446559, -0.2734810461381525, 0.2734810461381525, 0.8229514491446559, 1.380258539198881, 1.951787990916254, 2.546202157847481, 3.176999161979956, 3.869447904860123, 4.688738939305818};
        case(17): return {-4.871345193674403, -4.061946675875474, -3.378932091141494, -2.757762915703889, -2.173502826666621, -1.612924314221231, -1.067648725743451, -0.5316330013426547, 0, 0.5316330013426547, 1.067648725743451, 1.612924314221231, 2.173502826666621, 2.757762915703889, 3.378932091141494, 4.061946675875474, 4.871345193674403};
        case(18): return {-5.048364008874467, -4.248117873568126, -3.573769068486266, -2.961377505531607, -2.386299089166686, -1.835531604261629, -1.300920858389617, -0.7766829192674117, -0.2582677505190968, 0.2582677505190968, 0.7766829192674117, 1.300920858389617, 1.835531604261629, 2.386299089166686, 2.961377505531607, 3.573769068486266, 4.248117873568126, 5.048364008874467};
        case(19): return {-5.220271690537482, -4.428532806603779, -3.76218735196402, -3.157848818347602, -2.591133789794543, -2.049231709850619, -1.524170619393533, -1.010368387134311, -0.5035201634238882, 0, 0.5035201634238882, 1.010368387134311, 1.524170619393533, 2.049231709850619, 2.591133789794543, 3.157848818347602, 3.76218735196402, 4.428532806603779, 5.220271690537482};
        case(20): return {-5.387480890011233, -4.603682449550744, -3.944764040115625, -3.347854567383216, -2.78880605842813, -2.254974002089276, -1.738537712116586, -1.234076215395323, -0.7374737285453944, -0.2453407083009012, 0.2453407083009012, 0.7374737285453944, 1.234076215395323, 1.7385377121165861, 2.254974002089276, 2.78880605842813, 3.347854567383216, 3.944764040115625, 4.603682449550744, 5.387480890011233};
        case(21): return {-5.550351873264678, -4.773992343411219, -4.12199554749184, -3.531972877137678, -2.979991207704598, -2.453552124512838, -1.944962949186254, -1.448934250650732, -0.961499634418369, -0.4794507070791076, 0, 0.4794507070791076, 0.961499634418369, 1.448934250650732, 1.944962949186254, 2.453552124512838, 2.979991207704598, 3.531972877137678, 4.12199554749184, 4.773992343411219, 5.550351873264678};
        case(22): return {-5.709201353205264, -4.939834131060176, -4.294312480593162, -3.710701532877805, -3.165265909202137, -2.645637441058173, -2.144233592798534, -1.655874373286422, -1.176713958481244, -0.7036860971700069, -0.2341791399309906, 0.2341791399309906, 0.7036860971700069, 1.176713958481244, 1.655874373286422, 2.144233592798534, 2.645637441058173, 3.165265909202137, 3.710701532877805, 4.294312480593162, 4.939834131060176, 5.709201353205264};
        case(23): return {-5.864309498984573, -5.101534610476677, -4.462091173740007, -3.884472708106102, -3.345127159941225, -2.831803787126157, -2.337016211474456, -1.855677037671371, -1.384039585682495, -0.9191514654425638, -0.4585383500681048, 0, 0.4585383500681048, 0.9191514654425638, 1.384039585682495, 1.855677037671371, 2.337016211474456, 2.831803787126157, 3.345127159941225, 3.884472708106102, 4.462091173740007, 5.101534610476677, 5.864309498984573};
        case(24): return {-6.01592556142574, -5.259382927668044, -4.625662756423787, -4.05366440244815, -3.520006813034525, -3.012546137565565, -2.523881017011427, -2.049003573661699, -1.584250010961694, -1.126760817611245, -0.6741711070372122, -0.2244145474725156, 0.2244145474725156, 0.6741711070372122, 1.126760817611245, 1.584250010961694, 2.049003573661699, 2.523881017011427, 3.012546137565565, 3.520006813034525, 4.05366440244815, 4.625662756423787, 5.259382927668044, 6.01592556142574};
        case(25): return {-6.164272434052452, -5.413636355280034, -4.785320367352224, -4.218609444386561, -3.690282876998356, -3.188294924425105, -2.705320237173026, -2.236420130267281, -1.778001124337147, -1.327280702073084, -0.8819827562138214, -0.4401472986453083, 0, 0.4401472986453083, 0.8819827562138214, 1.327280702073084, 1.778001124337147, 2.236420130267281, 2.705320237173026, 3.188294924425105, 3.690282876998356, 4.218609444386561, 4.785320367352224, 5.413636355280034, 6.164272434052452};
        case(26): return {-6.309550385625694, -5.564524981950103, -4.941324957241379, -4.379602662983305, -3.856288419909149, -3.35942718235083, -2.881762219543087, -2.418415764773779, -1.965854785641137, -1.521361516651921, -1.082733011077883, -0.6480952139934483, -0.2157778562434634, 0.2157778562434634, 0.6480952139934483, 1.082733011077883, 1.521361516651921, 1.965854785641137, 2.418415764773779, 2.881762219543087, 3.35942718235083, 3.856288419909149, 4.379602662983305, 4.941324957241379, 5.564524981950103, 6.309550385625694};
        case(27): return {-6.451940140753472, -5.712255552816537, -5.093910003113184, -4.536906663372442, -4.018318670408739, -3.526275340134353, -3.053582419822255, -2.595416338910818, -2.148296645361627, -1.709560739260337, -1.277066817339858, -0.8490113420601031, -0.423807900543853, 0, 0.423807900543853, 0.8490113420601031, 1.277066817339858, 1.709560739260337, 2.148296645361627, 2.595416338910818, 3.053582419822255, 3.526275340134353, 4.018318670408739, 4.536906663372442, 5.093910003113184, 5.712255552816537, 6.451940140753472};
        case(28): return {-6.591605442367743, -5.857014641382851, -5.243285373202936, -4.690756523943118, -4.176636742129268, -3.689134238461679, -3.221112076561456, -2.767795352913594, -2.325749842656441, -1.892360496837685, -1.465537263457409, -1.043535273754208, -0.6248367195052092, -0.2080673826907369, 0.2080673826907369, 0.6248367195052092, 1.043535273754208, 1.465537263457409, 1.892360496837685, 2.325749842656441, 2.767795352913594, 3.221112076561456, 3.689134238461679, 4.176636742129268, 4.690756523943118, 5.243285373202936, 5.857014641382851, 6.591605442367743};
        case(29): return {-6.72869519860885, -5.99897128946382, -5.389640521966752, -4.841363651059164, -4.33147829381915, -3.84826679221362, -3.384645141092214, -2.935882504290126, -2.498585691019404, -2.070181076053428, -1.648622913892316, -1.232215755084753, -0.8194986812709116, -0.4091646363949287, 0, 0.4091646363949287, 0.8194986812709116, 1.232215755084753, 1.648622913892316, 2.070181076053428, 2.498585691019404, 2.935882504290126, 3.384645141092214, 3.84826679221362, 4.33147829381915, 4.841363651059164, 5.389640521966752, 5.99897128946382, 6.72869519860885};
        case(30): return {-6.863345293529892, -6.138279220123935, -5.533147151567496, -4.988918968589944, -4.483055357092518, -4.003908603861229, -3.54444387315535, -3.099970529586442, -2.667132124535617, -2.243391467761504, -1.826741143603688, -1.415527800198189, -1.008338271046723, -0.6039210586255523, -0.2011285765488715, 0.2011285765488715, 0.6039210586255523, 1.008338271046723, 1.415527800198189, 1.826741143603688, 2.243391467761504, 2.667132124535617, 3.099970529586442, 3.54444387315535, 4.003908603861229, 4.483055357092518, 4.988918968589944, 5.533147151567496, 6.138279220123935, 6.863345293529892};
        case(31): return {-6.99568012371854, -6.27507870494286, -5.673961444618588, -5.133595577112381, -4.63155950631286, -4.156271755818145, -3.700743403231469, -3.260320732313541, -2.831680453390205, -2.41231770548042, -2.000258548935639, -1.59388586047214, -1.191826998350046, -0.7928769769153089, -0.3959427364714231, 0, 0.3959427364714231, 0.7928769769153089, 1.191826998350046, 1.59388586047214, 2.000258548935639, 2.41231770548042, 2.831680453390205, 3.260320732313541, 3.700743403231469, 4.156271755818145, 4.63155950631286, 5.133595577112381, 5.673961444618588, 6.27507870494286, 6.99568012371854};
        case(32): return {-7.125813909830728, -6.40949814926966, -5.812225949515914, -5.27555098651588, -4.777164503502596, -4.305547953351198, -3.853755485471445, -3.417167492818571, -2.992490825002374, -2.577249537732317, -2.169499183606112, -1.767654109463202, -1.370376410952872, -0.9765004635896828, -0.5849787654359324, -0.1948407415693993, 0.1948407415693993, 0.5849787654359324, 0.9765004635896828, 1.370376410952872, 1.767654109463202, 2.169499183606112, 2.577249537732317, 2.992490825002374, 3.417167492818571, 3.853755485471445, 4.305547953351198, 4.777164503502596, 5.27555098651588, 5.812225949515914, 6.40949814926966, 7.125813909830728};
        case(33): return {-7.253851822015201, -6.541655445738077, -5.948071182087144, -5.414929002614193, -4.920028520595008, -4.451911148832827, -4.003671609956931, -3.570721980232718, -3.149796681703825, -2.738445824351355, -2.334751151529515, -1.937154581822207, -1.544348261243122, -1.15520020412679, -0.7687013797588687, -0.3839260145084091, 0, 0.3839260145084091, 0.7687013797588687, 1.15520020412679, 1.544348261243122, 1.937154581822207, 2.334751151529515, 2.738445824351355, 3.149796681703825, 3.570721980232718, 4.003671609956931, 4.451911148832827, 4.920028520595008, 5.414929002614193, 5.948071182087144, 6.541655445738077, 7.253851822015201};
        case(34): return {-7.379890950481246, -6.67165913607017, -6.081616993936316, -5.551861330988778, -5.060296018605762, -4.59551974810817, -4.150665602970781, -3.721175232476153, -3.303808431564416, -2.896138943174432, -2.496271940816547, -2.102673690467333, -1.714062553387338, -1.329335551884786, -0.9475164580334473, -0.5677172685548746, -0.1891080605271425, 0.1891080605271425, 0.5677172685548746, 0.9475164580334473, 1.329335551884786, 1.714062553387338, 2.102673690467333, 2.496271940816547, 2.896138943174432, 3.303808431564416, 3.721175232476153, 4.150665602970781, 4.59551974810817, 5.060296018605762, 5.551861330988778, 6.081616993936316, 6.67165913607017, 7.379890950481246};
        case(35): return {-7.504021146448936, -6.79960941328413, -6.212973747633717, -5.686468948090442, -5.198099346197753, -4.736518477413211, -4.294895814492763, -3.868700730969154, -3.454716495751991, -3.050538420430447, -2.654292781197172, -2.264467501042569, -1.879803988730917, -1.49922448861173, -1.121780990720303, -0.746617639879867, -0.3729417170496169, 0, 0.3729417170496169, 0.746617639879867, 1.121780990720303, 1.49922448861173, 1.879803988730917, 2.264467501042569, 2.654292781197172, 3.050538420430447, 3.454716495751991, 3.868700730969154, 4.294895814492763, 4.736518477413211, 5.198099346197753, 5.686468948090442, 6.212973747633717, 6.79960941328413, 7.504021146448936};
        case(36): return {-7.626325754003894, -6.925598990259942, -6.342243330994412, -5.818863279505577, -5.333560107113064, -4.875039972467084, -4.436506970192857, -4.01345656774947, -3.602693857148476, -3.201833945788159, -2.809022235131104, -2.422766042053562, -2.04182718355442, -1.665150001843414, -1.291810958820924, -0.920981801570753, -0.5519014332904228, -0.1838533671058128, 0.1838533671058128, 0.5519014332904228, 0.920981801570753, 1.291810958820924, 1.665150001843414, 2.04182718355442, 2.422766042053562, 2.809022235131104, 3.201833945788159, 3.602693857148476, 4.01345656774947, 4.436506970192857, 4.875039972467084, 5.333560107113064, 5.818863279505577, 6.342243330994412, 6.925598990259942, 7.626325754003894};
        case(37): return {-7.746882249649456, -7.049713855778229, -6.469520036524031, -5.949147217461971, -5.46679033596856, -5.011206138573073, -4.575631748667359, -4.155587281126479, -3.74789820647548, -3.350197894972536, -2.960649181303289, -2.577776858113272, -2.200360934009252, -1.827365248763605, -1.457887646874209, -1.091123764975933, -0.7263396166051203, -0.362849905050658, 0, 0.362849905050658, 0.7263396166051203, 1.091123764975933, 1.457887646874209, 1.827365248763605, 2.200360934009252, 2.577776858113272, 2.960649181303289, 3.350197894972536, 3.74789820647548, 4.155587281126479, 4.575631748667359, 5.011206138573073, 5.46679033596856, 5.949147217461971, 6.469520036524031, 7.049713855778229, 7.746882249649456};
        case(38): return {-7.865762803380041, -7.172033935320031, -6.594891327265494, -6.077416003537561, -5.597893514184678, -5.145129320740823, -4.712392132084888, -4.295225419749605, -3.890473760963341, -3.495787454835627, -3.109345311717942, -2.729687962888326, -2.355611733035508, -1.986097778039066, -1.620262755633014, -1.257323131700713, -0.896568346193136, -0.5373398108709835, -0.1790137232958775, 0.1790137232958775, 0.5373398108709835, 0.896568346193136, 1.257323131700713, 1.620262755633014, 1.986097778039066, 2.355611733035508, 2.729687962888326, 3.109345311717942, 3.495787454835627, 3.890473760963341, 4.295225419749605, 4.712392132084888, 5.145129320740823, 5.597893514184678, 6.077416003537561, 6.594891327265494, 7.172033935320031, 7.865762803380041};
        case(39): return {-7.983034772719781, -7.292633670865721, -6.718438506444093, -6.20375799772811, -5.726965451782105, -5.276913315230426, -4.846900568743526, -4.432492882593037, -4.030552814602468, -3.638746424874536, -3.255267235992229, -2.878670311374955, -2.507766693891319, -2.14155301198688, -1.779162582854313, -1.419830157685736, -1.062865567281179, -0.7076332733485723, -0.3535358469963293, 0, 0.3535358469963293, 0.7076332733485723, 1.062865567281179, 1.419830157685736, 1.779162582854313, 2.14155301198688, 2.507766693891319, 2.878670311374955, 3.255267235992229, 3.638746424874536, 4.030552814602468, 4.432492882593037, 4.846900568743526, 5.276913315230426, 5.726965451782105, 6.20375799772811, 6.718438506444093, 7.292633670865721, 7.983034772719781};
        case(40): return {-8.09876113925085, -7.411582531485469, -6.840237305249355, -6.328255351220082, -5.8540950560304, -5.406654247970128, -4.979260978545256, -4.567502072844395, -4.1682570668325, -3.779206753435223, -3.398558265859628, -3.024879883901284, -2.656995998442896, -2.293917141875083, -1.934791472282296, -1.578869894931614, -1.225480109046289, -0.8740066123570881, -0.5238747138322772, -0.1745372145975824, 0.1745372145975824, 0.5238747138322772, 0.8740066123570881, 1.225480109046289, 1.578869894931614, 1.934791472282296, 2.293917141875083, 2.656995998442896, 3.024879883901284, 3.398558265859628, 3.779206753435223, 4.1682570668325, 4.567502072844395, 4.979260978545256, 5.406654247970128, 5.8540950560304, 6.328255351220082, 6.840237305249355, 7.411582531485469, 8.09876113925085};
        default: return {};
    }
}

// todo maybe put it below directly?
inline double hermite_w(int n, double root, unsigned long long fact){ // w_i
    double h = hermitel(n-1,root);
    double out = (fact / (h*h)) * (pow(2,n-1)*sqrt(PI) / (n*n)); // since fact and h^2 are big
    return out;
}

vector<double> Hweights(int my_n) { // with big my_n (>20), we have very little Hweights
    my_n++;
    vector<double> weights(my_n, 0);
    vector<double> roots = Hroots(my_n);
    unsigned long long fact = factorial(my_n);
    for (int i = 0; i < my_n; i++) {
        weights[i] = hermite_w(my_n, roots[i], fact);
    }
    return weights;
}


void setQ(){
    Q_mat = VectorXd::Zero(sizeX);
    for(int i = 0; i < sizeX; i++){
        Q_mat(i) = 1.0/double(sizeX);
    }
    //cout << endl << "Q" << endl << Q_mat << endl;
}

void setPI(){
    vector<double> hweights = Hweights(n-1); // todo change n
    vector<double> windows;

    PI_mat = MatrixXd::Zero(sizeX, n*n*sizeX);
    //for(auto h: hweights){  cout << "weights: " << h << endl; }

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            windows.push_back(hweights[j] * hweights[i]); // windows is size n*n
        }
    }

    //for(auto w: windows){  cout << "windows: " << w << endl; }

    for(int i = 0; i < sizeX; i++){
        int a = 0;
        for(int j = i*n*n; j < i*n*n+n*n; j++){
            PI_mat(i,j) = windows[a];//all_hweights[n][i]*all_hweights[n][j]
            a++;
        }
    }
    //cout << endl << "PI: " << endl << PI_mat << endl;
    //cout << PI_mat.rows() << " " << PI_mat.cols();
}

void setW(){
    W_mat = MatrixXd::Zero(sizeX, n*n*sizeX);
    vector<double> roots = Hroots(n);
    vector<complex<double>> complexroots; // n*n
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            complexroots.push_back(complex<double>(roots[i], roots[j]));
        }
    }
    /*
    cout << endl << "X: " << endl;
    for(auto x: X_mat){
        cout << x << endl;
    }


    cout << endl << "Z: " << endl;
    for(auto z: complexroots){
        cout << z << endl;
    }
    */
    VectorXcd Y(n*n*sizeX); // todo revise
    int a = 0;
    for(int i = 0; i < n*n*sizeX; i += n*n){
        for(int j = 0; j < n*n; j++){
            Y(i+j) = X_mat(a);
            Y(i+j) += complexroots[j];
        }
        a++;
    }
    // cout << endl << "Y: " << endl << Y << endl;

    //MatrixXd D_mat(sizeX, n*n*sizeX);
    D_mat = MatrixXd::Zero(sizeX, n*n*sizeX);
    for(int i = 0; i < sizeX; i++){
        for(int j = 0; j < n*n*sizeX; j++){
            D_mat(i,j) = abs_sq(Y(j)-X_mat(i));
        }
    }
    // cout << endl << "D: " << endl << D_mat << endl;
    // cout << D_mat.rows() << " " << D_mat.cols();

    //W_mat = expNonSquare(-D_mat);

    // cout << endl << "W: " << endl <<W_mat << endl;
    // cout << endl << endl;

}

void setX(int npoints, string xmode){
    sizeX = npoints;
    X.resize(npoints);
    X_mat = VectorXd::Zero(sizeX);
    if(xmode == "pam"){ // pam
        float delta = sqrt(3/(pow(npoints,2)-1));
        for(int n = 0; n < npoints/2; n++){
            X[n+npoints/2] = (2*n+1)*delta;
            X_mat(n+npoints/2) = (2*n+1)*delta;
        }

        for(int n = 0; n < npoints/2; n++){
            X[n] = -X[npoints-1-n];
            X_mat(n) = -X_mat(npoints-1-n);
        }
    }
    else if(xmode == "psk"){ // psk
        for(int n = 0; n < npoints; n++){
            X[n] = (cos(2.0*PI*double(n)/npoints) + I*sin(2.0*PI*double(n)/npoints));
            X_mat(n) = (cos(2.0*PI*double(n)/npoints) + I*sin(2.0*PI*double(n)/npoints));
        }
    }
    else if(xmode == "secret"){
        for(int n = 0; n < npoints; n++) X[n] = (double(rand()) + I*double(rand()));
        //X = {-1/sqrt(2)-I*double(1/sqrt(2)), -1/sqrt(2)+I*double(1/sqrt(2)), +1/sqrt(2)-I*double(1/sqrt(2)), 1/sqrt(2)+I*double(1/sqrt(2))};
    }
    else{
        cout << "ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
    }

    //cout << "X: " << X_mat << endl;
    /*
    cout << endl << "XXXXXX";
    for(int n = 0; n < npoints; n++){
        cout << X[n] << " ";
    }
    cout << endl;
     */
    /*
    for(auto x: X){
        qDebug("x,mod is %f+i%f,%d", real(x),imag(x), sizeX);
    }
    */

    /*
    if(DEBUG){
        for(auto x: X){
            qDebug("x,mod is %f+i%f,%d", real(x),imag(x), sizeX);
        }
    }
    */
}

void setR(double r){
    R = r;
}

void setSNR(double snr){
    SNR = snr;
}

void setMod(int mod, string xmode){
    setX(mod, xmode);
}

std::chrono::microseconds sum_(vector<std::chrono::microseconds> vector1) {
    std:chrono::microseconds s = (std::chrono::microseconds) 0;
    for(auto x: vector1){
        s += x;
    }
    return s;
}

string complextostr(complex<double> x){
    return to_string(real(x)) + "+I*" + to_string(imag(x));
}

void initQ(){
    //qDebug() << "sx" << sizeX;
    double r, mysum=0;
    for(int n = 0; n < sizeX; n++){
        r = rand();
        Qq.push_back(r);
        mysum += r;
    }

    for(int n = 0; n < sizeX; n++){
        //qDebug() << "qprev " << Qq[n];
        Qq[n] /= mysum;
        //qDebug() << "qpost " << Qq[n];
    }
    //qDebug() << mysum;
}

double Q(complex<double> x){
    //return Qq[x];
    return 1.0/double(sizeX);
}


inline double cost(complex<double> current_x){ // returns the cost function given x
    double out = abs_sq(current_x);
    int a = 0;
    for (auto x: X){
        out -= Q(a) * abs_sq(x);
        a++;
    }
    //cout << "cost: " << out << endl;
    return out;
}

inline double W(complex<double> y, complex<double> x){
    return 1/PI * exp(-abs_sq(y-sqrt(SNR)*x));
}

inline double G(double alpha, complex<double> xhat, complex<double> y // a
        , double rho){                                  // rho
    return exp(-rho*alpha) * pow(W(y,xhat),1/(1+rho));
}


inline double G_co(double r, complex<double> xhat, complex<double> y // a
        , double rho){                                  // rho
    return exp(-r*rho*cost(xhat)) * pow(W(y,xhat),1/(1+rho));
}

inline double H_co(double r, complex<double> x, complex<double> y, double rho){
    return exp(r*cost(x)) * pow(W(y,x),1/(1+rho));
}

inline double H(double alpha, complex<double> x, complex<double> y, double rho){
    //return exp(alpha(x)) * pow(W(y,x),1/(1+rho));
    return exp(alpha) * pow(W(y,x),1/(1+rho));
}

double fa(complex<double> x, complex<double> y, vector<double> alphas, double rho, int xind){

    auto start_XX = std::chrono::high_resolution_clock::now();

    double f = 0;
    double h = H(alphas[xind],x,y,rho);
    int xindex = 0;
    for(auto xhat: X){
        f += Q(xindex) * G(alphas[xindex],xhat,y,rho) / h;
        xindex++;
    }

    auto stop_XX = std::chrono::high_resolution_clock::now();
    auto duration_XX = std::chrono::duration_cast<std::chrono::microseconds>(stop_XX - start_XX);
    fa_times.push_back(duration_XX);

    return f;
}

double fa_co(complex<double> x, complex<double> y, double r, double rho){

    auto start_XX = std::chrono::high_resolution_clock::now();

    double f = 0;
    double h = H_co(r,x,y,rho);
    int a = 0;
    for(auto xhat: X){
        f += Q(a) * G_co(r,xhat,y,rho) / h; // todo check Q()
        a++;
    }
    auto stop_XX = std::chrono::high_resolution_clock::now();
    auto duration_XX = std::chrono::duration_cast<std::chrono::microseconds>(stop_XX - start_XX);
    fa_co_times.push_back(duration_XX);

    return f;
}


double E_0(double rho, vector<double> alphas, int n){

    auto start_e0 = std::chrono::high_resolution_clock::now();

    /*
    cout << Q_mat.rows() << endl;
    cout << Q_mat.cols() << endl;
    cout << PI_mat.rows() << endl;
    cout << PI_mat.cols() << endl;
    cout << W_mat.rows() << endl;
    cout << W_mat.cols() << endl;
    */
    rho = 0; // todo temp

    MatrixXd W_firstpower =  W_mat.array().pow(-(rho)/(1+rho));
    MatrixXd W_secondpower = W_mat.array().pow(1/(1+rho));
    //MatrixXcd QW2 = ;
    //MatrixXcd QW2_transpose = QW2.transpose();

    //cout << endl << PI_mat << endl;

    cout << "Wf" << endl << W_firstpower << endl;
    cout << "Ws" << endl << W_secondpower << endl;
    cout << "test" << endl << (PI_mat.cwiseProduct(W_firstpower)) << endl;
    cout << "last" << endl << (Q_mat.array().transpose().pow(rho).matrix() * W_secondpower).transpose() << endl;

    VectorXd ones(n*n*sizeX); for(int i = 0; i < n*n*sizeX; i++){ ones(i) = 1; }

    double E0 = Q_mat.transpose() * // 1 x 32
                (PI_mat.cwiseProduct(W_firstpower)) * // 32 x 544
                ones;  // 1 x 32 x 32 x 544
    //(Q_mat.array().transpose() * W_secondpower).pow(rho).transpose();

    //MatrixXd ED = (D_mat*-0.5).array().exp();
    //MatrixXd Wpow = W_mat.array().pow(0.5);


    auto stop_e0 = std::chrono::high_resolution_clock::now();
    auto duration_e0 = std::chrono::duration_cast<std::chrono::microseconds>(stop_e0 - start_e0);

    //cout << endl << ED << endl;

    //double E0 = 0;

    e0_times.push_back(duration_e0);

    // todo change
    E0 = 0;
    return E0;
}

void printrowscols(MatrixXd m, string s){
    //cout << s << " " << m.rows() << " " << m.cols() << endl;
}

double E_0_1_co(){
    double sum = 0;
    for (complex<double> x: X){
        for(complex<double> xhat: X){
            sum += Q(x) * Q(xhat) * exp(-SNR/4 * abs_sq(x-xhat));
        }
    }
    return -log2(sum);
}

double E_0_co(double r, double rho, double& grad_rho, double& grad_2_rho, double& E0, int n, vector<double> hweights, vector<double> multhweights, vector<double> roots){
    //auto start_e0 = std::chrono::high_resolution_clock::now();

    MatrixXd G1 =  exp((rho)/(1+rho)*D_mat.array()).matrix();
    MatrixXd G2 = exp((-1)/(1+rho) *D_mat.array()).matrix();

    MatrixXd qg2 = (Q_mat.transpose() * G2);
    MatrixXd qg2rho = qg2.array().pow(rho).matrix().transpose();
    MatrixXd pig1 = (PI_mat.cwiseProduct(G1));

    double s = 1/(1+rho);
    double F0 = (Q_mat.transpose() * pig1 * qg2rho).value();

    MatrixXd W = exp(-D_mat.array()).matrix();

    MatrixXd second = -s*(
            (
                    (Q_mat.transpose()*
                    (pig1.cwiseProduct(W.array().log().matrix())) *
                    (qg2rho)
            )
            ).array()
            ).matrix();

    MatrixXd Qpig = Q_mat.transpose() * pig1;
    MatrixXd Qlog = (qg2rho.array() * qg2.array().log().transpose()).matrix();
    MatrixXd first = Qpig * Qlog;

    double mp = (first+second).value();

    F0 /= PI;
    double Fder0 = mp/PI;

    E0 = -log2(F0);

    grad_rho = -(Fder0)/((log(2)*(F0)));

     // Compute m2p
    double m2p = (Q_mat.transpose() * pig1 * (qg2rho.array() * Qlog.array().square()).matrix()).sum()
          - (2.0 / (1.0 + rho)) * (Q_mat.transpose() * (pig1.array() * (-D_mat.array())).matrix()
                                   * (qg2rho.array() * Qlog.array()).matrix()).sum()
          + (1.0 / std::pow(1.0 + rho,2)) * (Q_mat.transpose() * (pig1.array() * (-D_mat.array()).square()).matrix() * qg2rho).sum();
    //auto stop_e0 = std::chrono::high_resolution_clock::now();
    //auto duration_e0 = std::chrono::duration_cast<std::chrono::microseconds>(stop_e0 - start_e0);

    //e0_times.push_back(duration_e0);
    /*
    // Compute logqg2
    VectorXd logqg2 = (Q_mat.transpose() * (-1.0 / (1.0 + rho) * D_mat).array().exp().matrix()).array().log();

    // Compute qg2rho
    VectorXd qg2rho = (rho * logqg2.array()).exp();

    // Compute pig1
    MatrixXd pig1 = (PI_mat.array() * (rho / (1.0 + rho) * D_mat.array()).exp()).matrix();

    // Compute m
    double m = (Q_mat.transpose() * pig1 * qg2rho).sum();

    // Compute mp
    double mp = (Q_mat.transpose() * pig1 * (qg2rho.array() * logqg2.array()).matrix()).sum()
         - (1.0 / (1.0 + rho)) * (Q_mat.transpose() * (pig1.array() * (-D_mat.array())).matrix() * qg2rho).sum();

    // Compute m2p
    double m2p = (Q_mat.transpose() * pig1 * (qg2rho.array() * logqg2.array().square()).matrix()).sum()
          - (2.0 / (1.0 + rho)) * (Q_mat.transpose() * (pig1.array() * (-D_mat.array())).matrix()
                                   * (qg2rho.array() * logqg2.array()).matrix()).sum()
          + (1.0 / std::pow(1.0 + rho,2)) * (Q_mat.transpose() * (pig1.array() * (-D_mat.array()).square()).matrix() * qg2rho).sum();

    // Scale m, mp, m2p
    const double scale = 1.0 / M_PI;
    m *= scale;
    mp *= scale;
    m2p *= scale;

    E0 = -log2(m);
    grad_rho = -(mp)/((log(2)*(m)));
    */
    return E0;
}



inline void gradient_f(complex<double> x, complex<double> y, vector<double> alphas, double rho, vector<double>& grads_alpha, double& grad_rho, int xindex){

    auto start_XX = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < sizeX; i++){ grads_alpha[i] = 0; }
    grad_rho = 0;

    double h = H(alphas[xindex],x,y,rho), GHQ;
    complex<double> x_hat_hat;

    int xcounter = 0;
    for(auto xhat: X){
        GHQ = Q(xcounter)*(G(alphas[xcounter],xhat,y,rho) / h);

        for(int c = 0; c < sizeX; c++){ // last alpha not updated
            double aux = 0;
            x_hat_hat = X[c];

            if(xhat == x_hat_hat){
                aux -= rho;
            }

            if(x == x_hat_hat){
                aux -= 1;
            }

            grads_alpha[c] += GHQ * aux;
        }

        grad_rho += GHQ * (-alphas[xcounter]/* - 1/pow(1+rho,2)*log(W(y,xhat))+1/pow(1+rho,2)*log(W(y,x))*/);
        xcounter++;
    }
    auto stop_XX = std::chrono::high_resolution_clock::now();
    auto duration_XX = std::chrono::duration_cast<std::chrono::microseconds>(stop_XX - start_XX);
    gradient_f_times.push_back(duration_XX);
}

inline void gradient_f_co(complex<double> x, complex<double> y, double r, double rho, double& grad_r, double& grad_rho){

    auto start_XX = std::chrono::high_resolution_clock::now();

    grad_rho = 0;
    grad_r = 0;

    double h = H_co(r,x,y,rho), GHQ;
    int a = 0;
    for(auto xhat: X){
        double g = G_co(r,xhat,y,rho);
        GHQ = Q(a)*(g / h);

        grad_rho +=  GHQ * (-r*cost(xhat)/*-1/((1+rho)*(1+rho)) * (loG_co(wyx) - loG_co(W(y,xhat)))*/);

        double c1 = cost(xhat);
        double c2 = cost(x);
        grad_r += GHQ * (-rho*c1 - c2);
        a++;
    }

    auto stop_XX = std::chrono::high_resolution_clock::now();
    auto duration_XX = std::chrono::duration_cast<std::chrono::microseconds>(stop_XX - start_XX);
    gradient_f_co_times.push_back(duration_XX);
}


double e02(int n){

    auto start_XX = std::chrono::high_resolution_clock::now();

    double lhs = 0, rhs = 0, mylog, num_log, qx, wij;
    complex<double> y;
    int m = n;
    vector<double> roots = Hroots(n);
    vector<double> hweights = all_hweights[n];

    int xcounter = 0;
    for(auto x: X){
        qx = Q(xcounter);
        for(int i = 0; i < n; i++){
            for (int j = 0; j < m; j++){

                complex<double> root (roots[i], roots[j]); // wi + imag*wj;
                y = sqrt(SNR)*x + root;
                wij = hweights[j] * hweights[i]; //wi * wj

                num_log = 0;
                int a = 0;
                for(auto xhat: X){
                    float q = Q(a);
                    double w = W(y,xhat);
                    num_log += q*w;
                    a++;
                }
                mylog = log2(num_log / W(y,x));

                rhs += qx * mylog * wij * 1/PI;
                lhs += qx * mylog * mylog * wij * 1/PI;
            }
        }
        xcounter++;
    }

    double out = (rhs * rhs - lhs) * log(2);

    auto stop_XX = std::chrono::high_resolution_clock::now();
    auto duration_XX = std::chrono::duration_cast<std::chrono::microseconds>(stop_XX - start_XX);
    e02_times.push_back(duration_XX);

    return out;
}

inline void gradient_e0(vector<double> alphas, double rho, vector<double>& grads_alpha, double& grad_rho, int my_n, vector<double> hweights, vector<double> mult, vector<double> roots) { // TODO optimize
    // ---------------------------
    // | GRADIENT OF E_0 - rho*R |
    // ---------------------------

    auto start_XX = std::chrono::high_resolution_clock::now();

    double grad_f_rho;
    vector<double> grad_f_alphas(sizeX);
    double denominator = 0, numerator_rho = 0, hwij, hi, aa_f_a_rho;
    vector<double> numerator_alphas(sizeX);

    double Aa, f_a;
    complex<double> y;
    // fa
    for (int i = 0; i < my_n; i++) { //todo change multsize
        hi = hweights[i];
        for (int j = 0; j < my_n; j++) {
            hwij = hi*hweights[j];
            complex<double> root(roots[i], roots[j]);
            int xcounter = 0;
            for (auto x: X) {
                Aa = 1/PI * Q(xcounter) * hwij;
                y = std::sqrt(SNR) * x + root;
                gradient_f(x,y,alphas, rho, grad_f_alphas, grad_f_rho, xcounter); //para cada f(x,...) tenemos derivadas alpha(x_hat_hat)
                f_a = fa(x, y, alphas, rho, xcounter);
                denominator += Aa * std::pow(f_a,rho);
                numerator_rho += Aa * std::pow(f_a, rho) * (log(f_a) + rho * grad_f_rho / f_a);
                for(int c = 0; c < sizeX; c++){
                    numerator_alphas[c] += Aa * rho * std::pow(f_a, rho - 1) * grad_f_alphas[c];
                }
                xcounter++;
            }
        }
    }
    grad_rho = -std::log2(eu) * numerator_rho / denominator - R;
    for(int c = 0; c < sizeX-1; c++){ // last alpha not updated
        grads_alpha[c] = (-std::log2(eu) * numerator_alphas[c] / denominator) +
                         (-std::log2(eu) * numerator_alphas[sizeX-1] / denominator) * (-Q(c) / Q(sizeX-1));
    }


    auto stop_XX = std::chrono::high_resolution_clock::now();
    auto duration_XX = std::chrono::duration_cast<std::chrono::microseconds>(stop_XX - start_XX);
    gradient_e0_times.push_back(duration_XX);
}


inline void gradient_e0_co(double r, double rho, double& grad_r, double& grad_rho, int my_n, vector<double> hweights, vector<double> mult, vector<double> roots) { // TODO optimize
    // ---------------------------
    // | GRADIENT OF E_0 - rho*R |
    // ---------------------------

    auto start_XX = std::chrono::high_resolution_clock::now();

    double grad_f_r, grad_f_rho;
    double numerator_r = 0, denominator = 0, numerator_rho = 0, hwij, hi, aa_f_a_rho;

    vector<chrono::microseconds> inner_times;

    double Aa, f_a;
    complex<double> y;
    // fa
    for (int i = 0; i < my_n; i++) { // todo change multsize
        hi = hweights[i];
        for (int j = 0; j < my_n; j++) {
            hwij = hi*hweights[j];
            complex<double> root(roots[i], roots[j]);
            int a = 0;
            for (auto x: X) {
                Aa = 1/PI * Q(a) * hwij;
                y = std::sqrt(SNR) * x + root;

                auto start_inner = std::chrono::high_resolution_clock::now();

                gradient_f_co(x,y,r,rho, grad_f_r, grad_f_rho); //todo problems in the past
                f_a = fa_co(x, y, r, rho);

                auto stop_inner = std::chrono::high_resolution_clock::now();
                auto duration_inner = std::chrono::duration_cast<std::chrono::microseconds>(stop_inner - start_inner);

                inner_times.push_back(duration_inner);

                aa_f_a_rho =  Aa*std::pow(f_a, rho);
                numerator_r += Aa * rho * std::pow(f_a, rho - 1) * grad_f_r;
                denominator += aa_f_a_rho;
                numerator_rho += aa_f_a_rho * (std::log(f_a) + rho * grad_f_rho / f_a);
                a++;
            }
        }
    }

    grad_r = -std::log2(eu) * numerator_r / denominator;
    grad_rho = -std::log2(eu) * numerator_rho / denominator - R;

    auto stop_XX = std::chrono::high_resolution_clock::now();
    auto duration_XX = std::chrono::duration_cast<std::chrono::microseconds>(stop_XX - start_XX);
    gradient_e0_co_times.push_back(duration_XX - sum_(inner_times));
}


inline vector<double> mult_newhweights(vector<double> hweights, int my_n){
    // we only pick the most significant hweights, those inside the real-imaginary circle, and pairwise multiply and store them.

    auto start_XX = std::chrono::high_resolution_clock::now();

    sort(hweights.begin(), hweights.end());
    double boundary = hweights[my_n-1] - hweights[0];
    double mult, hi;
    vector<double> newhweights;

    int multsize = 0;
    for (int i = 0; i < my_n; i++) {
        hi = hweights[i];
        for (int j = 0; j < my_n; j++) {
            mult = hi * hweights[j];
            if(mult <= boundary){
                newhweights.push_back(mult);
                multsize++;
            }
        }
    }

    auto stop_XX = std::chrono::high_resolution_clock::now();
    auto duration_XX = std::chrono::duration_cast<std::chrono::microseconds>(stop_XX - start_XX);
    mult_newhweights_times.push_back(duration_XX);

    return newhweights;
}


void compute_hweights(int n, int num_iterations){
    /*
     * PRECOMPUTES THE WEIGHTS, ROOTS AND THE MULTIPLICATION OF WEIGHTS
     */
    //initQ();
    auto start_XX = std::chrono::high_resolution_clock::now();

    if (DEBUG){
        cout << endl;
        cout << setw( 4) << left << "it";
        cout << setw(10) << left << "rho";
        cout << setw(10) << left << "r";
        cout << setw(13) << left << "-e0+rho*r";
        cout << setw(13) << left << "-d_rho";
        cout << setw(13) << left << "d_r";
        cout << setw(3) << left << "N";
        cout << endl;
    }

    int my_n;
    low = n; // todo temp
    double increment = (n-low)/num_iterations;

    int prev_n = -1;
    vector<double> hweights, roots, multhweights;
    double msum;

    for (int i = 0; i < num_iterations; ++i) {
        msum = 0;

        my_n = ceil(low + increment*i);
        if(my_n != prev_n){
            all_hweights[my_n]     = Hweights(my_n-1);
            all_roots[my_n]        = Hroots(my_n);
            all_multhweights[my_n] = mult_newhweights(all_hweights[my_n], my_n);
        }
        prev_n = my_n;
    }
    auto stop_XX = std::chrono::high_resolution_clock::now();
    auto duration_XX = std::chrono::duration_cast<std::chrono::microseconds>(stop_XX - start_XX);
    compute_hweights_times.push_back(duration_XX);
}

// Gradient Descent of E0
double GD_ccomp(vector<double>& alphas, double& rho, double learning_rate, int num_iterations, int n) {
    //initQ();

    auto start_XX = std::chrono::high_resolution_clock::now();

    float error = 0.05;

    if (DEBUG){
        cout << endl;
        cout << setw( 4) << left << "it";
        cout << setw(10) << left << "rho";
        for(int c = 0; c < sizeX; c++){
            cout << setw(10) << left << "alpha" + to_string(c + 1);
        }
        cout << setw(13) << left << "-e0+rho*r";
        cout << setw(13) << left << "-d_rho";
        for(int c = 0; c < sizeX; c++){
            cout << setw(13) << left << "d_alpha" + to_string(c + 1);
        }
        cout << setw(3) << left << "N";
        cout << endl;
    }

    int my_n;

    double increment = (n-low)/num_iterations;

    int prev_n = -1;
    vector<double> hweights;
    vector<double> roots;
    vector<double> multhweights;
    double msum;
    /*
    if(DEBUG) {
        cout << "rho: " << rho << endl;
        for(auto x: alphas){
            cout << "alpha: " << x << " ";
        }
        cout << endl;
    }
    */

    double grad_rho = 0, grad_lambda; vector<double> grad_alphas(sizeX);

    for (int i = 0; i < num_iterations; ++i) {
        msum = 0;

        my_n = ceil(low + increment*i);

        // todo put out
        if(my_n != prev_n){
            hweights = all_hweights[my_n];
            roots = all_roots[my_n];
            multhweights = all_multhweights[my_n];
        }

        prev_n = my_n;

        gradient_e0(alphas, rho, grad_alphas, grad_rho, my_n-1, hweights, multhweights, roots);

        // update alphas
        for(int c = 0; c < sizeX - 1; c++){ // alphas from 0 to |X|-1
            alphas[c] += 0.1*grad_alphas[c];
            msum += Q(c) * alphas[c];
        }
        alphas[sizeX-1] = -1 / Q(sizeX-1) * msum; // alpha(|X|)

        // update rho
        rho += learning_rate * grad_rho; if (rho <= 0) rho = 0.00000001; else if (rho >= 1) rho = 0.99999999; // todo revise

        bool alphasder = true;
        //if(i >= 1){
        for(int p = 0; p < sizeX-2; p++){
            if(grad_alphas[p] >= error || grad_alphas[p] <= -error){
                alphasder = false;
                break;
            }
        }

        if(alphasder){
            if(rho == 0.00000001 || rho == 0.99999999 || grad_rho <= error && grad_rho >= -error){
                if(DEBUG) {
                    cout << endl;
                    double e0 =  E_0(rho, alphas, n) - rho * R;
                    cout << endl;
                    cout << setw( 4) << left << i;
                    cout << setw(10) << left << setprecision(6) << rho;
                    for(int c = 0; c < sizeX; c++){
                        cout << setw(10) << left << setprecision(3) << alphas[c] << " ";
                    }
                    cout << setw(13) << left << setprecision(6) << e0;
                    cout << setw(13) << left << setprecision(6) << grad_rho;
                    for(int c = 0; c < sizeX; c++){
                        cout << setw(13) << left << setprecision(3) << grad_alphas[c] << " ";
                    }
                    cout << setw(3) << left << my_n;
                    cout << " ";
                    cout << endl;
                }
                //qDebug() << i;
                return E_0(rho, alphas, n) - rho * R;
            }
        }

        //}
        if(DEBUG) {
            cout << endl;
            cout << setw( 4) << left << i;
            cout << setw(10) << left << setprecision(6) << rho;
            for(int c = 0; c < sizeX; c++){
                cout << setw(10) << left << setprecision(3) << alphas[c] << " ";
            }
            cout << setw(13) << left << setprecision(6) << E_0(rho, alphas, n) - rho * R;
            cout << setw(13) << left << setprecision(6) << grad_rho;
            for(int c = 0; c < sizeX; c++){
                cout << setw(13) << left << setprecision(3) << grad_alphas[c] << " ";
            }
            cout << setw(3) << left << my_n;
            cout << " ";
            cout << endl;
            //qDebug() << num_iterations;
        }
    }
    double out = E_0(rho, alphas, n) - rho * R;

    auto stop_XX = std::chrono::high_resolution_clock::now();
    auto duration_XX = std::chrono::duration_cast<std::chrono::microseconds>(stop_XX - start_XX);
    GD_ccomp_times.push_back(duration_XX);
    //cout << num_iterations << endl;
    return out;
}

// Gradient Descent of E0
/*
double GD_co(double& r, double& rho, double learning_rate, int num_iterations, int n, bool updateR) {
    auto start_XX = std::chrono::high_resolution_clock::now();

    if (DEBUG) cout  << "it |  rho   |  r  |-e0+rho*r| -grad_rho| -grad_r" << endl;

    float error = 0.01;

    int my_n;

    double increment = (n-low)/num_iterations;

    int prev_n = -1;
    vector<double> hweights;
    vector<double> roots;
    vector<double> multhweights;
    double e0;
    double grad_r, grad_rho;


    for (int i = 0; i < num_iterations; ++i) {

        my_n = ceil(low + increment*i);
        if(DEBUG) cout << "my_n: " << my_n << endl;

        if(my_n != prev_n){
            //free(&hweights); free(&roots); free(&multhweights);
            hweights = all_hweights[my_n];
            roots = all_roots[my_n];
            multhweights = all_multhweights[my_n];
        }

        prev_n = my_n;

        gradient_e0_co(r, rho, grad_r, grad_rho, my_n-1, hweights, multhweights, roots);

        if(updateR){ r += .5*grad_r; }
        else{ r = 0; }
        //else r = 0.000001;
        //cout << "lr:" << learning_rate << endl;
        rho += learning_rate * grad_rho; if (rho <= 0) rho = 0.00000001; else if (rho >= 1) rho = 0.99999999;

        if(!updateR || grad_r <= error && grad_r >= -error){
            if(rho == 0.00000001 || rho == 0.99999999 || grad_rho <= error && grad_rho >= -error){
                //cout << rho << " " << grad_rho << " " << grad_r << " " << endl;
                //cout << i << endl;
                return E_0_co(rho, r, -rho*r, n) - rho * R;
            }
        }

        if(DEBUG) cout << fixed << setprecision(6) << i << " " << rho << " " << r << " " << e0 << " " << grad_rho << " " << grad_r << endl;
    }
    //cout << rho << " " << grad_rho << " " << grad_r << " " << endl;
    //cout << num_iterations << endl;
    double out = E_0_co(rho, r, -rho*r, n) - rho * R;

    auto stop_XX = std::chrono::high_resolution_clock::now();
    auto duration_XX = std::chrono::duration_cast<std::chrono::microseconds>(stop_XX - start_XX);
    GD_co_times.push_back(duration_XX);

    return out;
}

*/
/*
double GD_iid(double& r, double& rho, double learning_rate, int num_iterations, int n){
    auto start_XX = std::chrono::high_resolution_clock::now();

    double out = GD_co(r, rho, learning_rate, num_iterations, n, false);

    auto stop_XX = std::chrono::high_resolution_clock::now();
    auto duration_XX = std::chrono::duration_cast<std::chrono::microseconds>(stop_XX - start_XX);
    GD_iid_times.push_back(duration_XX);

    return out;
}
*/
/*
double GD_cc(double& r, double& rho, double learning_rate, int num_iterations, int n){
    auto start_XX = std::chrono::high_resolution_clock::now();

    double out = GD_co(r, rho, learning_rate, num_iterations, n, true);

    auto stop_XX = std::chrono::high_resolution_clock::now();
    auto duration_XX = std::chrono::duration_cast<std::chrono::microseconds>(stop_XX - start_XX);
    GD_cc_times.push_back(duration_XX);

    return out;
}
*/
void NAG_update(double &x_t, double &y_t, double &x_tp1, double &y_tp1, double beta, double grad, double kaux){
    // updates the approximation using Nesterov Accelerated Gradient

    y_tp1 =  x_t - beta * grad;
    x_tp1 = (1+kaux)*y_tp1 - kaux*y_t;

    x_t = x_tp1; y_t = y_tp1;
}

double NAG(vector<double>& alphas, double& rho, int num_iterations, double beta, double k, int n) {

    auto start_NAG = std::chrono::high_resolution_clock::now();

    double kaux = ((sqrt(k)-1)/(sqrt(k)+1));
    vector<double> x_t_r = alphas, y_t_r = alphas, y_tp1_r(sizeX), x_tp1_r(sizeX);
    double x_t_rho = rho, y_t_rho = rho, y_tp1_rho, x_tp1_rho;
    double grad_rho;
    vector<double> grads_alpha (sizeX);
    if (DEBUG) cout  << "it |  rho   |  r  |-e0+rho*r| -grad_rho| -grad_r" << endl;

    int my_n = n;
    double increment = (n-low)/num_iterations;

    int prev_n = -1;
    vector<double> hweights;
    vector<double> roots;
    vector<double> multhweights;

    for (int i = 0; i < num_iterations; ++i) {

        my_n = ceil(low + increment*i);
        if(DEBUG) cout << "my_n: " << my_n << endl;

        if(my_n != prev_n){
            hweights = all_hweights[my_n];
            roots = all_roots[my_n];
            multhweights = all_multhweights[my_n];
        }

        prev_n = my_n;

        //grad_rho was nan in gradient_e0()
        gradient_e0(x_t_r, x_t_rho, grads_alpha, grad_rho, my_n-1, hweights, multhweights, roots);
        for(int c = 0; c < sizeX; c++) grads_alpha[c] = - grads_alpha[c];
        grad_rho = - grad_rho;

        for(int c = 0; c < sizeX; c++) NAG_update(x_t_r[c], y_t_r[c], x_tp1_r[c], y_tp1_r[c], beta, grads_alpha[c], kaux);
        NAG_update(x_t_rho, y_t_rho, x_tp1_rho, y_tp1_rho, beta, grad_rho, kaux);

        if(DEBUG){
            cout << fixed << setprecision(6) << i << " " << x_t_rho;
            for(int c = 0; c < sizeX; c++) cout << " " << x_t_r[c] << " ";
            cout << E_0(x_t_rho,x_t_r, n)-x_t_rho*R << " " << -grad_rho << " ";
            for(int c = 0; c < sizeX; c++) cout << " " << -grads_alpha[c] << " ";
            cout << endl;
        }
    }
    double out = E_0(x_t_rho,x_t_r, n)-x_t_rho*R;

    auto stop_NAG = std::chrono::high_resolution_clock::now();
    auto duration_NAG = std::chrono::duration_cast<std::chrono::microseconds>(stop_NAG - start_NAG);
    NAG_times.push_back(duration_NAG);

    return out;
}

double NAG_co(double& r, double& rho, double learning_rate, int num_iterations, int n, double k, bool updateR) {

    auto start_XX = std::chrono::high_resolution_clock::now();

    if (DEBUG) cout  << "it |  rho   |  r  |-e0+rho*r| -grad_rho| -grad_r" << endl;

    float error = 0.00000000001;

    double kaux = ((sqrt(k)-1)/(sqrt(k)+1));

    int my_n;

    double increment = (n-low)/num_iterations;

    int prev_n = -1;
    vector<double> hweights;
    vector<double> roots;
    vector<double> multhweights;
    double e0;
    double grad_r, grad_rho, grad_2_rho;
    double nextrho, auxrho = rho, nextauxrho;
    double nextr, auxr = rho, nextauxr;
    vms inner_times;

    // vector<double> rhos = {0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1};

    for (int i = 0; i < num_iterations; ++i) {

        my_n = ceil(low + increment*i);
        if(DEBUG) cout << "my_n: " << my_n << endl;

        if(my_n != prev_n){
            //free(&hweights); free(&roots); free(&multhweights);
            hweights = all_hweights[my_n];
            roots = all_roots[my_n];
            multhweights = all_multhweights[my_n];
        }

        prev_n = my_n;

        auto start_inner2 = std::chrono::high_resolution_clock::now();
        // gradient_e0_co(r, rho, grad_r, grad_rho, my_n-1, hweights, multhweights, roots);

        // cout << "gr: " << grad_rho << endl;
        //cout << "rho: " << rho << endl;

        //rho = rhos[i];
        E_0_co(0, rho, grad_rho, grad_2_rho, e0, my_n-1, hweights, multhweights, roots);
        grad_rho -= R;
        //cout << "gr_new: " << grad_rho << endl;

        //cout << "rho: " << rho << endl;

        //rho = rhos[i];
        //gradient_e0_co(0, rho, grad_r, grad_rho, my_n-1, hweights, multhweights, roots);

        //cout << "gr_old: " << grad_rho << endl;
        //cout << "rho: " << rho << endl;

        auto stop_inner2 = std::chrono::high_resolution_clock::now();
        auto duration_inner2 = std::chrono::duration_cast<std::chrono::microseconds>(stop_inner2 - start_inner2);
        inner_times.push_back(duration_inner2);


        grad_rho = -grad_rho;
        grad_r = -grad_r;


        if(updateR){ NAG_update(r, auxr, nextr, nextauxr, learning_rate, grad_r, kaux); }
        else{ r = 0; }
        //else r = 0.000001;
        //cout << "lr:" << learning_rate << endl;

        NAG_update(rho, auxrho, nextrho, nextauxrho, learning_rate, grad_rho, kaux);
        //if (rho <= 0) rho = 0.00000001; else if (rho >= 1) rho = 0.99999999;
        //rho += grad_rho;
        if(rho <= -1) rho = -1;
        if(rho >= 2) rho = 2;

        /*
        //if(!updateR || grad_r <= error && grad_r >= -error){
            if(rho == 0.00000001 || rho == 0.99999999 || grad_rho <= error && grad_rho >= -error){
                //cout << rho << " " << grad_rho << " " << grad_r << " " << endl;
                //cout << i << endl;
                return e0 - rho*R;
            }
        //}
        */
        if(DEBUG) cout << fixed <<   setprecision(17) << i << " " << rho << " " << e0 << " " << e0-rho*R << " " << grad_rho << endl;
    }
    //cout << rho << " " << grad_rho << " " << grad_r << " " << endl;
    //cout << num_iterations << endl;

    auto start_inner = std::chrono::high_resolution_clock::now();

    auto stop_inner = std::chrono::high_resolution_clock::now();
    auto duration_inner = std::chrono::duration_cast<std::chrono::microseconds>(stop_inner - start_inner);

    inner_times.push_back(duration_inner);

    auto stop_XX = std::chrono::high_resolution_clock::now();
    auto duration_XX = std::chrono::duration_cast<std::chrono::microseconds>(stop_XX - start_XX);
    NAG_co_times.push_back(duration_XX - sum_(inner_times));

    return e0 - rho*R;
}

double NAG_iid(double& r, double& rho, double learning_rate, int num_iterations, int n, double k){
    auto start_NAG_iid = std::chrono::high_resolution_clock::now();
    //cout << endl << "cooking" << endl;
    double out = NAG_co(r, rho, learning_rate, num_iterations, n, k, false);

    auto stop_NAG_iid = std::chrono::high_resolution_clock::now();
    auto duration_NAG_iid = std::chrono::duration_cast<std::chrono::microseconds>(stop_NAG_iid - start_NAG_iid);
    NAG_iid_times.push_back(chrono::microseconds(0));

    return out;
}

double NAG_cc(double& r, double& rho, double learning_rate, int num_iterations, int n, double k){
    auto start_NAG_cc = std::chrono::high_resolution_clock::now();

    double out = NAG_co(r, rho, learning_rate, num_iterations, n, k, true);

    auto stop_NAG_cc = std::chrono::high_resolution_clock::now();
    auto duration_NAG_cc = std::chrono::duration_cast<std::chrono::microseconds>(stop_NAG_cc - start_NAG_cc);
    NAG_cc_times.push_back(duration_NAG_cc);

    return out;
}


#endif //TFG_FUNCTIONS_H