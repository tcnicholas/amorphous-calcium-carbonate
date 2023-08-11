/*
 *           C++ implementation the three-body correlation function
 *
 *           Implemented by Sergey Sukhomlinov (2019)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated files, without restriction to use, copy,
 * modify, merge and/or distribute copies of this software.
 */

#include <cmath>
#include <iostream>
#include <string>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <vector>

using namespace std;

string ltrim(const string& s){
  size_t start = s.find_first_not_of(" ");
  return (start == string::npos) ? "" : s.substr(start);
}

string rtrim(const string& s){
  size_t end = s.find_last_not_of(" ");
  return (end == string::npos) ? "" : s.substr(0, end + 1);
}

string trim(const string& s){
  return rtrim(ltrim(s));
}

const double pi2i=1./M_PI/2;

// main parameters

// declare variables and assign default values
// while reading in the config file, the values are rewritten
int nAtom = 1620, nType = 1, nConf = 1;
int nrGrid = 201, naGrid = 201;
double Rmin = 0, Rmax = 12, LRcut = 30.0;
int TypeC = 1, TypeE = 1; // atomic type of the Central and of the End atom
string inputFile = "ca_coords.txt", outputFile = "g3distr_custom.dat";

double aveVolC, aveVolE;
double LRcut2, dr, dr1; // grid in r
double da, da1;         // grid in cos
double dr2da1;
double LRcutMod1;       // for binning
int nRnA;
int rBinMin, rBinMax;

ifstream inCoords;
vector <int> Type;
vector < vector <double> > g3;

//----------------------------------------
void InputData(std::string filename);
void setParameters();
void g3Calc(vector<int>);
void OutputData(vector< vector< double > >);

void pbc(double [], double []);
