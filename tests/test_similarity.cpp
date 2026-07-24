// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <DataStructs/BitOps.h>
#include <DataStructs/ExplicitBitVect.h>
#include <gmock/gmock.h>
#include <GraphMol/Fingerprints/MorganFingerprints.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <string>
#include <tuple>

#include "src/similarity.h"
#include "src/utils/device.h"

using namespace RDKit;
using namespace nvMolKit;

namespace {

// --------------------------------
// Test data and fixtures
// --------------------------------

const std::array<std::string, 85> rdkitTestSmiles = {
  "Brc1cccc(Nc2ncnc3ccncc23)c1NCCN1CCOCC1",
  "COc1c(O)cc(O)c(C(=N)Cc2ccc(O)cc2)c1O",
  "CCOC(=O)c1cc2cc(C(=O)O)ccc2[nH]1",
  "C[C@H](NC(=O)OCc1ccccc1)C(=O)N[C@@H](C)C(=O)NN(CC(N)=O)C(=O)/C=C/C(=O)N(Cc1ccco1)Cc1ccco1",
  "CO[C@H]1C[C@H](COC[C@H]2[C@@H](OC)C[C@H](O[C@H]3CC[C@@]4(C)C(=CC[C@]5(O)[C@@H]4C[C@@H](OC(=O)/C=C/c4ccccc4)[C@]4(C)[C@](O)(C(C)=O)CC[C@]54O)C3)O[C@@H]2C)O[C@@H](C)[C@H]1COC[C@H]1C[C@H](OC)[C@H](COC[C@H]2C[C@@H](OC)[C@@H](O[C@H]3O[C@@H](CO)[C@H](O)[C@@H](O)[C@@H]3O)[C@H](C)O2)[C@@H](C)O1",
  "COc1ccc2c(c1OC)C(CC1(C)C=Cc3c(c4cccc(OC)c4n(C)c3=O)O1)N(C)c1c-2ccc2cc3c(cc12)OCO3",
  "CC(C)C[C@H](NC(=O)[C@H](Cc1cnc[nH]1)NC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(=O)[C@@H]1CCCN1C(=O)[C@@H](N)CS)C(=O)N[C@@H](CC(C)C)C(=O)N1CCC[C@H]1C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CS)C(=O)O",
  "CC(C)=C[C@H]1C[C@](C)(O)[C@@H]2[C@H]3CC[C@@H]4[C@@]5(C)CC[C@H](O[C@@H]6OC[C@H](O)[C@H](O[C@@H]7O[C@H](CO)[C@@H](O)[C@H](O)[C@H]7O)[C@H]6O[C@@H]6O[C@@H](COC(=O)CC(=O)O)[C@H](O)[C@H]6O)C(C)(C)[C@@H]5CC[C@@]4(C)[C@@]34CO[C@@]2(C4)O1",
  "CCn1c2ccc3cc2c2cc(ccc21)C(=O)c1ccc(cc1)Cn1cc[n+](c1)Cc1ccc(cc1)-c1cccc(c1C(=O)O)-c1ccc(cc1)C[n+]1ccn(c1)Cc1ccc(cc1)C3=O.[Br-].[Br-]",
  "COc1cccc(CNC2(CCC(C)(C)C)C(=O)C(C3=NS(=O)(=O)c4cc(NS(C)(=O)=O)ccc4N3)=C(O)c3ccccc32)c1",
  "CCCCC[C@H]1CCCCCCCCCC(=O)O[C@@H]2[C@@H](O[C@@H]3O[C@H](C)[C@@H](OC(=O)C(C)CC)[C@H](O)[C@H]3O)[C@H](C)O[C@@H](O[C@H]3[C@@H](O[C@H](CO)[C@@H](O)[C@@H]3O)O[C@@H]3[C@@H](O)[C@H](O)[C@@H](C)O[C@H]3O1)[C@@H]2OC(=O)C(C)CC",
  "C[C@H]1CN(CCC(=O)N[C@@H](CCc2ccccc2)C(=O)O)CC[C@@]1(C)c1cccc(O)c1",
  "O=C(NO)c1ccc(I)cc1",
  "C=CCc1ccc(OC(=O)C23CC4CC(CC(C4)C2)C3)c(OC)c1",
  "Cc1cccc(N2CCN(CCCON3C(=O)c4ccccc4C3=O)CC2)c1",
  "N=c1sc2ccccc2n1CCN1CCC(c2ccc(F)cc2)CC1",
  "CCCCCCCCCCCCCC(=O)NCc1ccc(C(=O)N[C@H](C(=O)O)[C@@H](C)CC)cc1",
  "O=C(Nc1ncc(F)s1)[C@H](CC1CCOCC1)c1ccc(S(=O)(=O)C2CC2)cc1",
  "Oc1ccc(/N=C(/Cc2ccc(Cl)cc2)c2ccc(O)c(O)c2O)cc1",
  "COC1=C2C[C@@H](O)CC[C@]2(C)[C@H]2CC[C@@]3(C)[C@@H](CC(=O)[C@]3(O)[C@H](C)[C@H](CCC(C)C)O[C@@H]3OC[C@H](O)[C@H](O[C@@H]4OC[C@@H](O)[C@H](O)[C@H]4OC(=O)c4ccc(OC)cc4)[C@H]3OC(C)=O)[C@@H]2C1",
  "CO[C@H]1[C@H](OC[C@H]2[C@@H]3O[C@@H]3/C=C/C(=O)[C@H](C)CC[C@H](O[C@@H]3O[C@H](C)C[C@H](O)[C@H]3O)[C@@H](C)/C=C/C(=O)O[C@@H]2C)O[C@H](C)[C@@H](O)[C@H]1OC",
  "COc1cc(C(O)C(COC(=O)/C=C/c2ccc(O)cc2)Oc2c(OC)cc(/C=C/COC(=O)/C=C/c3ccc(O)cc3)cc2OC)ccc1O",
  "COc1cc(O)c(-c2cc(-c3cc(-c4c(O)cc(O)c5c4C[C@@H](C)N[C@@H]5C)c4cc(C)cc(OC)c4c3O)c(O)c3c(OC)cc(C)cc23)c2c1C(C)=N[C@H](C)C2",
  "C[C@H]1CO[C@]2(C[C@@H]1O)O[C@H]1C[C@H]3[C@@H]4CC=C5C[C@@H](O)C[C@@H](O[C@@H]6OC[C@H](O)[C@H](O[C@@H]7OC[C@@H](O)[C@H](O)[C@H]7O)[C@H]6O[C@@H]6O[C@@H](C)[C@H](O)[C@@H](O)[C@H]6O)[C@]5(C)[C@H]4CC[C@]3(C)[C@H]1[C@@H]2C",
  "COC(=O)C[C@H]1C(C)(C)[C@H](OC(C)=O)[C@H]2C(=O)[C@]1(C)[C@H]1CC[C@@]3(C)[C@H](c4ccoc4)OC(=O)C[C@H]3[C@]13O[C@H]23",
  "CC(C)=CCC/C(C)=C/Cc1c2c(c3oc4c(c(=O)c3c1O)CC1c3c(c(O)cc(O)c3-4)OC1(C)C)C=CC(C)(C)O2",
  "CC(C)(C)OC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@@H]1CCCN1C(=O)[C@@H](N)Cc1ccc(O)cc1",
  "COc1ccccc1OCCN1CCN(c2ccc(=O)n(Cn3nc(N4CCN(CCOc5ccccc5OC)CC4)ccc3=O)n2)CC1",
  "C[C@@H](O)[C@H](N)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(N)=O",
  "CCCCCC[C@H](N)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](Cc1ccccc1)C(N)=O",
  "CC[C@H](C)[C@H](NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@@H](N)Cc1ccccc1)C(=O)NCC(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CC(C)C)C(=O)O",
  "CC(C)C[C@H](NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CCCCN)NC(=O)[C@H](CCCCN)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](C)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@@H]1CCCN1C(=O)[C@@H](N)[C@@H](C)O)C(N)=O",
  "CC(C)C[C@@H](C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CS)C(=O)O)N(C)C(=O)CN(C)C(=O)CNC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccsc1)NC(=O)CNC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@@H](N)CCCNC(=N)N)C(C)(C)S)[C@@H](C)O",
  "O=S(=O)(CCO)c1no[n+]([O-])c1-c1ccccc1",
  "CN1C(=O)[C@H](NC(=O)Nc2cccc(OCCNC(=O)COCC(=O)NNC(=O)COCC(=O)N[C@H]3CC[C@@]4(O)[C@H]5Cc6cc(O)cc7c6[C@@]4(CCN5C)[C@H]3O7)c2)N=C(c2ccccc2)c2ccccc21",
  "C=C1CC[C@@]2(OC1)O[C@H]1C[C@H]3[C@@H]4CC[C@@H]5C[C@@H](O[C@@H]6O[C@H](CO)[C@H](O)[C@H](O)[C@H]6O[C@@H]6O[C@H](CO)[C@@H](O)[C@H](O)[C@H]6O)[C@@H](O)C[C@]5(C)[C@H]4CC[C@]3(C)[C@H]1[C@@H]2C",
  "CC(C)C[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CO)NC(=O)[C@@H]1CCCN1C(=O)[C@H](CCC(N)=O)NC(=O)[C@@H](N)CS)C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CS)C(=O)O",
  "CCC(=O)[C@@H]1C[C@@H](C)[C@]2(CC[C@@]3(C)C4=C(CC[C@@]32C)[C@@]2(C)CC[C@H](O[C@@H]3O[C@H](CO[C@@H]5OC[C@H](O)[C@H](O)[C@H]5O[C@@H]5O[C@H](CO)[C@@H](O)[C@H](O[C@@H]6OC[C@H](O)[C@H](O)[C@H]6O[C@@H]6OC[C@@H](O)[C@H](O)[C@H]6O)[C@H]5O[C@@H]5O[C@@H](C)[C@H](O)[C@@H](O)[C@H]5O)[C@@H](O)[C@H](O)[C@H]3O)[C@](C)(CO)[C@@H]2CC4)O1",
  "CCCCCCCCCCCC[C@@H](O)[C@H]1CC[C@H]([C@H](O)CCC(O)CCCC(O)CCC[C@@H](O)CC2=C[C@H](C)OC2=O)O1",
  "CNc1nc2sc(SC)nc2c2c1ncn2C",
  "O=C(NO)[C@H]1C[C@]1(Cc1ccc(OCc2cc(-c3ccccc3)nc3ccccc23)cc1)C(=O)N1CCC[C@H]1CO",
  "CC(=O)OC[C@]12[C@H](OC(C)=O)C(=O)[C@@H]3[C@@H](O)[C@]1(OC3(C)C)[C@](C)(O)C[C@H](OC(C)=O)[C@@H]2OC(C)=O",
  "C[C@@H](c1ccccc1)[C@@H]1NC(=O)CNC(=O)[C@H](CO)NC(=O)[C@@H]([C@@H](O)[C@@H]2CN=C(N)N2[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]2O)NC(=O)[C@H]([C@H](O)[C@@H]2CN=C(N)N2)NC(=O)[C@@H](Cc2ccc3nc(-c4ccc(C(=O)O)cc4)oc3c2)NC1=O",
  "CCCC1(C)CC(=O)N(CCCCN2CCN(c3nsc4ccccc34)CC2)C(=O)C1.Cl",
  "CSCC[C@H](NC(=O)[C@@H](CC(C)C)NC(=O)CNC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@@H](CCC(N)=O)NC(=O)[C@@H](CCC(N)=O)NC(=O)[C@@H]1CCCN1C(=O)[C@@H](CCCCN)NC(=O)[C@@H]1CCCN1C(=O)[C@@H](N)CCCN=C(N)N)C(N)=O",
  "CC(=O)Nc1ccc(C[C@H](NC(=O)[C@H](CO)NC(=O)[C@@H](Cc2cccnc2)NC(=O)[C@@H](Cc2ccc(Cl)cc2)NC(=O)[C@@H](Cc2ccc3ccccc3c2)NC(C)=O)C(=O)N[C@@H](Cc2ccc(CNC(N)=O)cc2)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCCNC(C)C)C(=O)N2CCC[C@H]2C(=O)N[C@H](C)C(N)=O)cc1",
  "CO[C@]1(CC[C@H](C)CO[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)O[C@H]2C[C@H]3[C@@H]4CC=C5C[C@@H](O[C@@H]6O[C@H](CO)[C@@H](O)[C@H](O[C@@H]7O[C@@H](C)[C@H](O[C@@H]8O[C@H](CO)[C@@H](O)[C@H](O)[C@H]8O)[C@@H](O)[C@H]7O)[C@H]6O[C@@H]6O[C@@H](C)[C@H](O)[C@@H](O)[C@H]6O)CC[C@]5(C)[C@H]4CC[C@]3(C)[C@H]2[C@@H]1C",
  "Cl.O=C(Nc1ccc2c(c1)c1ccccc1n2CCCCN1CCN(C/C=C/c2ccccc2)CC1)c1ccccc1",
  "CC(C)=CCC/C(C)=C/C=C/C(=O)N1CCCC1",
  "CSCC[C@H](NC(=O)[C@H](CC(C)C)NC(=O)CNC(=O)[C@H](Cc1ccccc1)NC(=O)[C@@H](Cc1ccccc1)NC(=O)[C@@H](CCC(N)=O)NC(=O)[C@@H](CCC(N)=O)NC(=O)[C@@H]1CCCN1C(=O)[C@@H](CCCCN)NC(=O)[C@H]1CCCN1C(=O)[C@H](N)CCCN=C(N)N)C(N)=O",
  "C[C@@H](c1ccccc1)[C@@H]1NC(=O)CNC(=O)[C@H](CO)NC(=O)[C@@H]([C@@H](O)[C@@H]2CN=C(N)N2[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]2O)NC(=O)[C@H]([C@@H](O)[C@@H]2CN=C(N)N2)NC(=O)[C@@H](Cc2ccc(O[C@H]3O[C@H](CO)[C@@H](O[C@H]4O[C@H](COCc5cccs5)[C@@H](O)[C@H](O)[C@@H]4O)[C@H](O)[C@@H]3O)cc2)NC1=O",
  "CNC(=O)c1cc(C(O)CNC(C)CCc2ccc3c(c2)OCO3)ccc1O.Cl",
  "CNC(=O)[C@H](CCCNC(=O)OC(C)(C)C)NC(=O)[C@H](CCCc1ccccc1)[C@@](C)(O)C(=O)NO",
  "COc1ccccc1N1CCN(CCCNC(=O)c2ccccc2I)CC1.O=C(O)C(=O)O",
  "Cc1cc(NC(=O)c2cc(Cl)cc(Cl)c2O)ccc1Sc1nc2ccccc2s1",
  "CN(C/C=C/c1ccc(C#N)cc1)Cc1ccc2c(c1)CCO2",
  "COc1ccc(CNc2nnc(N3CCC(O)CC3)c3ccc(C#N)cc23)cc1Cl",
  "Cc1cccc(NC(=S)Nc2ccc3c(c2)C(=O)OC3)c1",
  "C=CCn1cc(C[C@@H]2NC(=O)[C@@H]3CCCN3C2=O)c2ccc(OC)cc21",
  "Cc1ccc(Oc2nc3ccccc3nc2N2CCN(C)CC2)cc1",
  "O=C1CC(c2ccc(-c3ccccc3)cc2)c2c(ccc3ccccc23)N1",
  "CC(=O)NC(C(=O)N1CCSCC1)[C@H]1CC(C(=O)O)C[C@@H]1N=C(N)N",
  "CCCc1cnnn1-c1c(Cl)cc(C(F)(F)F)cc1Cl",
  "O=C(O)C(=O)Nc1nc(-c2cc(Cl)no2)cs1",
  "CCc1c(C)[nH]c2c1/C(=N/OC(=O)NCCc1cccnc1)CCC2",
  "CCC[C@@H]1NC(=O)[C@@H](C(C)C)NC(=O)[C@@H](Cc2ccc(O)cc2)NCc2ccccc2CCCCNC1=O",
  "COc1cc2c(cc1OC)CC(=O)N(CCCN(C)CCc1ccccn1)CC2",
  "CCN(NC(=O)[C@H]1CCCN1C(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@@H](NC(=O)[C@H](CC(=O)O)NC(C)=O)[C@@H](C)O)C(C)C)C(C)C)C(=O)Oc1ccc([N+](=O)[O-])cc1",
  "C[C@H]1CC[C@@]2(OC1)O[C@H]1CC3[C@@H]4CC=C5C[C@@H](O[C@@H]6O[C@H](CO)[C@@H](O[C@H]7O[C@H](C)[C@@H](OCCNC(=O)CCCCl)[C@H](O)[C@@H]7O)[C@H](O)[C@H]6O[C@H]6O[C@H](C)[C@@H](O)[C@H](O)[C@@H]6O)CC[C@]5(C)[C@H]4CC[C@]3(C)[C@H]1[C@@H]2C",
  "CC[C@H](C)[C@H](NC(=O)[C@H](Cc1ccc(OP(=O)(O)O)cc1)NC(=O)[C@H](CC(N)=O)NC(=O)[C@H](CC(C)C)NC(C)=O)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](CC(C)C)C(=O)N[C@H](C(N)=O)C(C)C",
  "O=C(NC[C@@H]1CCCNC1)[C@H]1CCCN1C(=O)[C@@H]1C[C@@H](O)CN1C(=O)CC(c1ccc(F)cc1)(c1ccc(F)cc1)c1ccc(F)cc1",
  "CCC(C)CNc1nc(C#N)nc2c1NCN2Cc1ccccc1",
  "CCOC(=O)COC(=O)CCCNC(=O)NC12CC3CC(CC(C3)C1)C2",
  "CC[C@@H](c1ccc(C(F)(F)F)cc1)N1CCN(C2(C)CCN(C(=O)c3c(C)ncnc3C)CC2)C[C@@H]1C",
  "CCCCCCC[C@H]1OC(=O)C[C@@H](O)[C@H](Cc2ccccc2)N(C)C(=O)[C@H](Cc2ccccc2)OC(=O)[C@H]1C",
  "CCCN1c2cc(C)nc3c(-c4ccc(Cl)cc4Cl)nn(c23)C[C@@H]1CC",
  "CN(C)[C@H]1[C@@H](O[C@H]2O[C@H](CO)[C@@H](O)[C@H](N)[C@H]2O)[C@H](NC(=O)[C@@H](O)CCN)C[C@H](N)[C@H]1O[C@H]1O[C@H](CN)CC[C@H]1N",
  "N=C(N)NC(=O)OCc1cccc(N2CCC(Oc3cccnc3)CC2)c1F",
  "CC[C@H](C)[C@H](NC(=O)CNC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CO)NC(=O)[C@H](CC(N)=O)NC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(=O)[C@H](CC(N)=O)NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@H](CC(N)=O)NC(=O)[C@@H]1CCCN1C(=O)[C@@H](N)CC(C)C)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](Cc1ccccc1)C(N)=O",
  "CCCCc1oc2ccccc2c1Cc1cccc(/C(C)=C/Cn2oc(=O)[nH]c2=O)c1",
  "C[C@H](NC(=O)[C@H]1Cc2c(sc3ccccc23)CN1)c1ccccc1.Cl",
  "Oc1cc(Cl)ccc1Oc1ccc(Cl)cc1CN1CCN(C(c2ccccc2)c2ccccc2)CC1",
  "COC(=O)c1cc(CCc2ccc(OC)cc2OC)ccc1O",
  "C=C(CO)[C@]12C=C3C(=O)C[C@@H](C(C)C)[C@]3(C)C[C@@H](O)[C@@]1(C)CC(=O)O2",
  "C=C(CC(O)C(C)(O)[C@H]1CC[C@H]2C3=C[C@H](OC(C)=O)[C@H]4[C@@H](OC(C)=O)[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C)C(C)C"};

void InitializeMols(std::vector<std::unique_ptr<ROMol>>& mols) {
  for (const auto& smi : rdkitTestSmiles) {
    ROMol* mol = SmilesToMol(smi);
    assert(mol != nullptr);
    mols.emplace_back(mol);
  }
}

template <typename ParamType> class SimilarityParamTestFixture : public testing::TestWithParam<ParamType> {
 protected:
  SimilarityParamTestFixture() { InitializeMols(mols_); }

  std::vector<std::unique_ptr<ROMol>> mols_;
};

}  // namespace

// --------------------------------
// Supplementary functions
// --------------------------------

// TODO: Move this to a helper utility or consolidate with other similar functions
template <typename T, typename OtherT>
cuda::std::span<const OtherT> castAsSpanOfSmallerType(const AsyncDeviceVector<T>& vec) {
  static_assert(sizeof(OtherT) <= sizeof(T), "Size of smaller type must be less than or equal to the larger type.");
  static_assert(sizeof(T) % sizeof(OtherT) == 0, "Size of smaller type must be a divisor of the larger type.");
  constexpr int sizeMultiplier = sizeof(T) / sizeof(OtherT);
  assert(vec.size() > 0);
  return cuda::std::span<OtherT>(reinterpret_cast<OtherT*>(vec.data()), vec.size() * sizeMultiplier);
}

// --------------------------------
// Parameterized cross- and self-similarity tests with metric selection
// --------------------------------

enum class SimMetric {
  Tanimoto,
  Cosine
};

// Cross: N, M, bits, metric
using CrossParams = std::tuple<int, int, int, SimMetric>;

class CrossSimilarityParamTestFixture : public testing::TestWithParam<CrossParams> {
 protected:
  CrossSimilarityParamTestFixture() { InitializeMols(mols_); }

  std::vector<std::unique_ptr<ROMol>> mols_;
};
static std::string SimMetricToString(SimMetric m) {
  switch (m) {
    case SimMetric::Tanimoto:
      return "Tan";
    case SimMetric::Cosine:
      return "Cos";
  }
  return "Unknown";
}

static std::string CrossTestName(const testing::TestParamInfo<CrossParams>& info) {
  const auto [n, m, bits, metric] = info.param;
  return SimMetricToString(metric) + std::string("_N") + std::to_string(n) + "_M" + std::to_string(m) + "_B" +
         std::to_string(bits);
}

TEST_P(CrossSimilarityParamTestFixture, CrossCpuGpuAgree) {
  const auto [N, M, kNumBits, metric] = GetParam();

  std::vector<std::unique_ptr<ExplicitBitVect>> explicitVectsA;
  std::vector<std::unique_ptr<ExplicitBitVect>> explicitVectsB;
  explicitVectsA.reserve(N);
  explicitVectsB.reserve(M);

  for (int i = 0; i < N; ++i) {
    const size_t idx = static_cast<size_t>(i) % mols_.size();
    explicitVectsA.emplace_back(MorganFingerprints::getFingerprintAsBitVect(*mols_[idx], 3, kNumBits));
  }
  for (int j = 0; j < M; ++j) {
    const size_t idx = static_cast<size_t>(j) % mols_.size();
    explicitVectsB.emplace_back(MorganFingerprints::getFingerprintAsBitVect(*mols_[idx], 3, kNumBits));
  }

  std::vector<double> wantSimilarities;
  wantSimilarities.reserve(static_cast<size_t>(N) * static_cast<size_t>(M));
  for (const auto& a : explicitVectsA) {
    for (const auto& b : explicitVectsB) {
      if (metric == SimMetric::Tanimoto) {
        wantSimilarities.push_back(TanimotoSimilarity(*a, *b));
      } else {
        wantSimilarities.push_back(CosineSimilarity(*a, *b));
      }
    }
  }

  std::vector<std::uint64_t> bitsA;
  std::vector<std::uint64_t> bitsB;
  for (const auto& bitVec : explicitVectsA) {
    boost::to_block_range(*bitVec->dp_bits, std::back_inserter(bitsA));
  }
  for (const auto& bitVec : explicitVectsB) {
    boost::to_block_range(*bitVec->dp_bits, std::back_inserter(bitsB));
  }

  AsyncDeviceVector<std::uint64_t> bitsGpuA(bitsA.size());
  AsyncDeviceVector<std::uint64_t> bitsGpuB(bitsB.size());
  bitsGpuA.copyFromHost(bitsA);
  bitsGpuB.copyFromHost(bitsB);
  auto bitsGpuSpanA = castAsSpanOfSmallerType<std::uint64_t, std::uint32_t>(bitsGpuA);
  auto bitsGpuSpanB = castAsSpanOfSmallerType<std::uint64_t, std::uint32_t>(bitsGpuB);

  AsyncDeviceVector<double> gotSimilaritiesGpu =
    (metric == SimMetric::Tanimoto) ? crossTanimotoSimilarityGpuResult(bitsGpuSpanA, bitsGpuSpanB, kNumBits) :
                                      crossCosineSimilarityGpuResult(bitsGpuSpanA, bitsGpuSpanB, kNumBits);
  std::vector<double> gotSimilaritiesCpu(gotSimilaritiesGpu.size());
  gotSimilaritiesGpu.copyToHost(gotSimilaritiesCpu);
  cudaDeviceSynchronize();
  EXPECT_THAT(gotSimilaritiesCpu, testing::Pointwise(testing::DoubleNear(1e-5), wantSimilarities));
}

INSTANTIATE_TEST_SUITE_P(CrossSimilarity,
                         CrossSimilarityParamTestFixture,
                         testing::Combine(testing::Values(1, 10, 100),
                                          testing::Values(1, 10, 100),
                                          testing::Values(128, 2048),
                                          testing::Values(SimMetric::Tanimoto, SimMetric::Cosine)),
                         CrossTestName);

// --------------------------------
// Parameterized self-similarity tests (N, bits, metric)
// --------------------------------

using SelfParams = std::tuple<int, int, SimMetric>;  // N, bits, metric

class SelfSimilarityParamTestFixture : public testing::TestWithParam<SelfParams> {
 protected:
  SelfSimilarityParamTestFixture() { InitializeMols(mols_); }

  std::vector<std::unique_ptr<ROMol>> mols_;
};

static std::string SelfTestName(const testing::TestParamInfo<SelfParams>& info) {
  const auto [n, bits, metric] = info.param;
  return SimMetricToString(metric) + std::string("_N") + std::to_string(n) + "_B" + std::to_string(bits);
}

TEST_P(SelfSimilarityParamTestFixture, SelfCpuGpuAgree) {
  const auto [N, kNumBits, metric] = GetParam();

  std::vector<std::unique_ptr<ExplicitBitVect>> explicitVects;
  explicitVects.reserve(N);
  for (int i = 0; i < N; ++i) {
    const size_t idx = static_cast<size_t>(i) % mols_.size();
    explicitVects.emplace_back(MorganFingerprints::getFingerprintAsBitVect(*mols_[idx], 3, kNumBits));
  }

  std::vector<double> wantSimilarities;
  wantSimilarities.reserve(static_cast<size_t>(N) * static_cast<size_t>(N));
  for (const auto& a : explicitVects) {
    for (const auto& b : explicitVects) {
      if (metric == SimMetric::Tanimoto) {
        wantSimilarities.push_back(TanimotoSimilarity(*a, *b));
      } else {
        wantSimilarities.push_back(CosineSimilarity(*a, *b));
      }
    }
  }

  std::vector<std::uint64_t> bits;
  for (const auto& bitVec : explicitVects) {
    boost::to_block_range(*bitVec->dp_bits, std::back_inserter(bits));
  }

  AsyncDeviceVector<std::uint64_t> bitsGpu(bits.size());
  bitsGpu.copyFromHost(bits);
  auto bitsGpuSpan = castAsSpanOfSmallerType<std::uint64_t, std::uint32_t>(bitsGpu);

  AsyncDeviceVector<double> gotSimilaritiesGpu = (metric == SimMetric::Tanimoto) ?
                                                   crossTanimotoSimilarityGpuResult(bitsGpuSpan, kNumBits) :
                                                   crossCosineSimilarityGpuResult(bitsGpuSpan, kNumBits);
  std::vector<double>       gotSimilaritiesCpu(gotSimilaritiesGpu.size());
  gotSimilaritiesGpu.copyToHost(gotSimilaritiesCpu);
  cudaDeviceSynchronize();
  EXPECT_THAT(gotSimilaritiesCpu, testing::Pointwise(testing::DoubleNear(1e-5), wantSimilarities));
}

INSTANTIATE_TEST_SUITE_P(SelfSimilarity,
                         SelfSimilarityParamTestFixture,
                         testing::Combine(testing::Values(1, 10, 100),
                                          testing::Values(128, 2048),
                                          testing::Values(SimMetric::Tanimoto, SimMetric::Cosine)),
                         SelfTestName);

// --------------------------------
// Cross similarity CPU result tests (parameterized)
// --------------------------------

// Cross CPU params: N, M, bits
using CrossCpuParams = std::tuple<int, int, int>;

class CrossCpuSimilarityParamTestFixture : public testing::TestWithParam<CrossCpuParams> {
 protected:
  CrossCpuSimilarityParamTestFixture() { InitializeMols(mols_); }

  std::vector<std::unique_ptr<ROMol>> mols_;
};

static std::string CrossCpuTestName(const testing::TestParamInfo<CrossCpuParams>& info) {
  const auto [n, m, bits] = info.param;
  return std::string("CPU_Tan_N") + std::to_string(n) + "_M" + std::to_string(m) + "_B" + std::to_string(bits);
}

// Default memory path (may compute in a single shot)
TEST_P(CrossCpuSimilarityParamTestFixture, DefaultMemoryAgrees) {
  const auto [N, M, kNumBits] = GetParam();

  std::vector<std::unique_ptr<ExplicitBitVect>> explicitVectsA;
  std::vector<std::unique_ptr<ExplicitBitVect>> explicitVectsB;
  explicitVectsA.reserve(N);
  explicitVectsB.reserve(M);
  for (int i = 0; i < N; ++i) {
    const size_t idx = static_cast<size_t>(i) % mols_.size();
    explicitVectsA.emplace_back(MorganFingerprints::getFingerprintAsBitVect(*mols_[idx], 3, kNumBits));
  }
  for (int j = 0; j < M; ++j) {
    const size_t idx = static_cast<size_t>(j) % mols_.size();
    explicitVectsB.emplace_back(MorganFingerprints::getFingerprintAsBitVect(*mols_[idx], 3, kNumBits));
  }

  std::vector<double> wantSimilarities;
  wantSimilarities.reserve(static_cast<size_t>(N) * static_cast<size_t>(M));
  for (const auto& a : explicitVectsA) {
    for (const auto& b : explicitVectsB) {
      wantSimilarities.push_back(TanimotoSimilarity(*a, *b));
    }
  }

  std::vector<std::uint64_t> bitsA;
  std::vector<std::uint64_t> bitsB;
  for (const auto& bitVec : explicitVectsA) {
    boost::to_block_range(*bitVec->dp_bits, std::back_inserter(bitsA));
  }
  for (const auto& bitVec : explicitVectsB) {
    boost::to_block_range(*bitVec->dp_bits, std::back_inserter(bitsB));
  }

  AsyncDeviceVector<std::uint64_t> bitsGpuA(bitsA.size());
  AsyncDeviceVector<std::uint64_t> bitsGpuB(bitsB.size());
  bitsGpuA.copyFromHost(bitsA);
  bitsGpuB.copyFromHost(bitsB);
  auto bitsGpuSpanA = castAsSpanOfSmallerType<std::uint64_t, std::uint32_t>(bitsGpuA);
  auto bitsGpuSpanB = castAsSpanOfSmallerType<std::uint64_t, std::uint32_t>(bitsGpuB);

  const auto got = crossTanimotoSimilarityCPUResult(bitsGpuSpanA, bitsGpuSpanB, kNumBits);
  cudaDeviceSynchronize();
  EXPECT_THAT(got, testing::Pointwise(testing::DoubleNear(1e-5), wantSimilarities));
}

// Constrained memory: attempt to force segmentation when feasible (e.g., N>=100)
TEST_P(CrossCpuSimilarityParamTestFixture, ConstrainedMemoryAgrees) {
  const auto [N, M, kNumBits] = GetParam();

  std::vector<std::unique_ptr<ExplicitBitVect>> explicitVectsA;
  std::vector<std::unique_ptr<ExplicitBitVect>> explicitVectsB;
  explicitVectsA.reserve(N);
  explicitVectsB.reserve(M);
  for (int i = 0; i < N; ++i) {
    const size_t idx = static_cast<size_t>(i) % mols_.size();
    explicitVectsA.emplace_back(MorganFingerprints::getFingerprintAsBitVect(*mols_[idx], 3, kNumBits));
  }
  for (int j = 0; j < M; ++j) {
    const size_t idx = static_cast<size_t>(j) % mols_.size();
    explicitVectsB.emplace_back(MorganFingerprints::getFingerprintAsBitVect(*mols_[idx], 3, kNumBits));
  }

  std::vector<double> wantSimilarities;
  wantSimilarities.reserve(static_cast<size_t>(N) * static_cast<size_t>(M));
  for (const auto& a : explicitVectsA) {
    for (const auto& b : explicitVectsB) {
      wantSimilarities.push_back(TanimotoSimilarity(*a, *b));
    }
  }

  std::vector<std::uint64_t> bitsA;
  std::vector<std::uint64_t> bitsB;
  for (const auto& bitVec : explicitVectsA) {
    boost::to_block_range(*bitVec->dp_bits, std::back_inserter(bitsA));
  }
  for (const auto& bitVec : explicitVectsB) {
    boost::to_block_range(*bitVec->dp_bits, std::back_inserter(bitsB));
  }

  AsyncDeviceVector<std::uint64_t> bitsGpuA(bitsA.size());
  AsyncDeviceVector<std::uint64_t> bitsGpuB(bitsB.size());
  bitsGpuA.copyFromHost(bitsA);
  bitsGpuB.copyFromHost(bitsB);
  auto bitsGpuSpanA = castAsSpanOfSmallerType<std::uint64_t, std::uint32_t>(bitsGpuA);
  auto bitsGpuSpanB = castAsSpanOfSmallerType<std::uint64_t, std::uint32_t>(bitsGpuB);

  const size_t           totalBytes   = static_cast<size_t>(N) * static_cast<size_t>(M) * sizeof(double);
  // Minimal bytes to support two rotating buffers for a batch of 32 rows
  // Derived from kernel's buffer math: freeBytes >= ceil( (32*M) * (20*sizeof(double)) / 9 )
  const size_t           minThreshold = ((size_t)32 * (size_t)M * (size_t)20 * sizeof(double) + 8) / 9;
  const size_t           margin       = 64;
  CrossSimilarityOptions opts;
  if (totalBytes > minThreshold + margin) {
    opts.maxDeviceMemoryBytes = minThreshold + margin;  // forces segmented path
  } else {
    opts.maxDeviceMemoryBytes = totalBytes + margin;  // not enough headroom to segment; compute single-shot
  }

  const auto got = crossTanimotoSimilarityCPUResult(bitsGpuSpanA, bitsGpuSpanB, kNumBits, opts);
  cudaDeviceSynchronize();
  EXPECT_THAT(got, testing::Pointwise(testing::DoubleNear(1e-5), wantSimilarities));
}

INSTANTIATE_TEST_SUITE_P(CrossCpuSimilarity,
                         CrossCpuSimilarityParamTestFixture,
                         testing::Combine(testing::Values(1, 10, 100),
                                          testing::Values(1, 10, 100),
                                          testing::Values(128, 2048)),
                         CrossCpuTestName);

// --------------------------------
// Cosine cross CPU tests (default and constrained memory)
// --------------------------------

class CrossCpuCosineSimilarityParamTestFixture : public testing::TestWithParam<CrossCpuParams> {
 protected:
  CrossCpuCosineSimilarityParamTestFixture() { InitializeMols(mols_); }

  std::vector<std::unique_ptr<ROMol>> mols_;
};

static std::string CrossCpuCosTestName(const testing::TestParamInfo<CrossCpuParams>& info) {
  const auto [n, m, bits] = info.param;
  return std::string("CPU_Cos_N") + std::to_string(n) + "_M" + std::to_string(m) + "_B" + std::to_string(bits);
}

TEST_P(CrossCpuCosineSimilarityParamTestFixture, DefaultMemoryAgrees) {
  const auto [N, M, kNumBits] = GetParam();

  std::vector<std::unique_ptr<ExplicitBitVect>> explicitVectsA;
  std::vector<std::unique_ptr<ExplicitBitVect>> explicitVectsB;
  explicitVectsA.reserve(N);
  explicitVectsB.reserve(M);
  for (int i = 0; i < N; ++i) {
    const size_t idx = static_cast<size_t>(i) % mols_.size();
    explicitVectsA.emplace_back(MorganFingerprints::getFingerprintAsBitVect(*mols_[idx], 3, kNumBits));
  }
  for (int j = 0; j < M; ++j) {
    const size_t idx = static_cast<size_t>(j) % mols_.size();
    explicitVectsB.emplace_back(MorganFingerprints::getFingerprintAsBitVect(*mols_[idx], 3, kNumBits));
  }

  std::vector<double> wantSimilarities;
  wantSimilarities.reserve(static_cast<size_t>(N) * static_cast<size_t>(M));
  for (const auto& a : explicitVectsA) {
    for (const auto& b : explicitVectsB) {
      wantSimilarities.push_back(CosineSimilarity(*a, *b));
    }
  }

  std::vector<std::uint64_t> bitsA;
  std::vector<std::uint64_t> bitsB;
  for (const auto& bitVec : explicitVectsA) {
    boost::to_block_range(*bitVec->dp_bits, std::back_inserter(bitsA));
  }
  for (const auto& bitVec : explicitVectsB) {
    boost::to_block_range(*bitVec->dp_bits, std::back_inserter(bitsB));
  }

  ScopedStream                     inputStream;
  AsyncDeviceVector<std::uint64_t> bitsGpuA(bitsA.size(), inputStream.stream());
  AsyncDeviceVector<std::uint64_t> bitsGpuB(bitsB.size(), inputStream.stream());
  bitsGpuA.copyFromHost(bitsA);
  bitsGpuB.copyFromHost(bitsB);
  auto bitsGpuSpanA = castAsSpanOfSmallerType<std::uint64_t, std::uint32_t>(bitsGpuA);
  auto bitsGpuSpanB = castAsSpanOfSmallerType<std::uint64_t, std::uint32_t>(bitsGpuB);

  const auto got = crossCosineSimilarityCPUResult(bitsGpuSpanA, bitsGpuSpanB, kNumBits, {}, inputStream.stream());
  cudaDeviceSynchronize();
  EXPECT_THAT(got, testing::Pointwise(testing::DoubleNear(1e-5), wantSimilarities));
}

TEST_P(CrossCpuCosineSimilarityParamTestFixture, ConstrainedMemoryAgrees) {
  const auto [N, M, kNumBits] = GetParam();

  std::vector<std::unique_ptr<ExplicitBitVect>> explicitVectsA;
  std::vector<std::unique_ptr<ExplicitBitVect>> explicitVectsB;
  explicitVectsA.reserve(N);
  explicitVectsB.reserve(M);
  for (int i = 0; i < N; ++i) {
    const size_t idx = static_cast<size_t>(i) % mols_.size();
    explicitVectsA.emplace_back(MorganFingerprints::getFingerprintAsBitVect(*mols_[idx], 3, kNumBits));
  }
  for (int j = 0; j < M; ++j) {
    const size_t idx = static_cast<size_t>(j) % mols_.size();
    explicitVectsB.emplace_back(MorganFingerprints::getFingerprintAsBitVect(*mols_[idx], 3, kNumBits));
  }

  std::vector<double> wantSimilarities;
  wantSimilarities.reserve(static_cast<size_t>(N) * static_cast<size_t>(M));
  for (const auto& a : explicitVectsA) {
    for (const auto& b : explicitVectsB) {
      wantSimilarities.push_back(CosineSimilarity(*a, *b));
    }
  }

  std::vector<std::uint64_t> bitsA;
  std::vector<std::uint64_t> bitsB;
  for (const auto& bitVec : explicitVectsA) {
    boost::to_block_range(*bitVec->dp_bits, std::back_inserter(bitsA));
  }
  for (const auto& bitVec : explicitVectsB) {
    boost::to_block_range(*bitVec->dp_bits, std::back_inserter(bitsB));
  }

  ScopedStream                     inputStream;
  AsyncDeviceVector<std::uint64_t> bitsGpuA(bitsA.size(), inputStream.stream());
  AsyncDeviceVector<std::uint64_t> bitsGpuB(bitsB.size(), inputStream.stream());
  bitsGpuA.copyFromHost(bitsA);
  bitsGpuB.copyFromHost(bitsB);
  auto bitsGpuSpanA = castAsSpanOfSmallerType<std::uint64_t, std::uint32_t>(bitsGpuA);
  auto bitsGpuSpanB = castAsSpanOfSmallerType<std::uint64_t, std::uint32_t>(bitsGpuB);

  const size_t           totalBytes   = static_cast<size_t>(N) * static_cast<size_t>(M) * sizeof(double);
  const size_t           minThreshold = ((size_t)32 * (size_t)M * (size_t)20 * sizeof(double) + 8) / 9;
  const size_t           margin       = 64;
  CrossSimilarityOptions opts;
  if (totalBytes > minThreshold + margin) {
    opts.maxDeviceMemoryBytes = minThreshold + margin;
  } else {
    opts.maxDeviceMemoryBytes = totalBytes + margin;
  }

  const auto got = crossCosineSimilarityCPUResult(bitsGpuSpanA, bitsGpuSpanB, kNumBits, opts, inputStream.stream());
  cudaDeviceSynchronize();
  EXPECT_THAT(got, testing::Pointwise(testing::DoubleNear(1e-5), wantSimilarities));
}

INSTANTIATE_TEST_SUITE_P(CrossCpuCosineSimilarity,
                         CrossCpuCosineSimilarityParamTestFixture,
                         testing::Combine(testing::Values(1, 10, 100),
                                          testing::Values(1, 10, 100),
                                          testing::Values(128, 2048)),
                         CrossCpuCosTestName);
