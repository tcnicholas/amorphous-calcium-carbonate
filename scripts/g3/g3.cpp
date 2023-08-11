/*
 *           C++ implementation of the three-body correlation function
 *
 *           Implemented by Sergey Sukhomlinov (2019)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated files, without restriction to use, copy,
 * modify, merge and/or distribute copies of this software.
 *
 * Note from @tcnicholas: I have modified the main function to load an input
 *                        data file from the first command line argument.
 */


#include "g3.h"


int main(int argc, char** argv){
  if (argc < 2) {
    std::cout << "Usage: ./g3 <filename>" << std::endl;
    return 1;
  }
  std::string filename = argv[1];  

  InputData(filename);
  inCoords.open(inputFile);
  setParameters();
  double time_conf_old, time_conf;
  for (int iConf = 0; iConf < nConf; ++iConf){
    g3Calc(Type);
  }
  inCoords.close();
  inCoords.clear();
  OutputData(g3);
}

//****************************************************************************//

void InputData(std::string filename){
  ifstream Input(filename);
  string keyword, var;
  while (true){
    if (Input.eof()) break;
    Input >> var; getline(Input, keyword);
    string ROLtrimmed = trim(keyword);
    if (ROLtrimmed == "#nAtom") nAtom = stoi(var);
    else if (ROLtrimmed == "#nType") nType = stoi(var);
    else if (ROLtrimmed == "#nConf") nConf = stoi(var);
    else if (ROLtrimmed == "#nrGrid") nrGrid = stoi(var);
    else if (ROLtrimmed == "#naGrid") naGrid = stoi(var);
    else if (ROLtrimmed == "#Rmin") Rmin = stod(var);
    else if (ROLtrimmed == "#Rmax") Rmax = stod(var);
    else if (ROLtrimmed == "#LRcut") LRcut = stod(var);
    else if (ROLtrimmed == "#TypeC") TypeC = stoi(var);
    else if (ROLtrimmed == "#TypeE") TypeE = stoi(var);
    else if (ROLtrimmed == "#input") inputFile = var;
    else if (ROLtrimmed == "#output") outputFile = var;
  }
  Input.close();
  Input.clear();
}

//******************************************************************************//


void setParameters(){
  aveVolC=0.; aveVolE=0.;
  LRcut2 = LRcut*LRcut;
  dr = LRcut/(nrGrid-1); dr1 = 1./dr; // grid in r
  da = 2./(naGrid-1); da1 = 1./da;    // grid in cos
  dr2da1 = 1./(dr*dr*da);
  LRcutMod1=1./(LRcut*1.01); // for binning
  nRnA = nrGrid*naGrid;

  // Rmin and Rmax convert to rBinMin and rBinMax
  rBinMin = static_cast<int>(Rmin/dr);
  rBinMax = static_cast<int>(Rmax/dr);
  
  Type.resize(nAtom);
  g3.resize(naGrid);
  for (int i=0; i<naGrid; ++i) g3[i].resize(nrGrid);
}

//****************************************************************************//

void g3Calc(vector<int> Type){

  // to store the general g3 data locally
  vector < double > Pr(naGrid*nrGrid*nrGrid, 0.0);

  string dummy; int tmp;
  for (int i=0; i<3; ++i) getline(inCoords, dummy);
  inCoords >> tmp; getline(inCoords, dummy);
  if (tmp != nAtom){ 
    cerr << "Check number of atoms!\n"; cout << tmp << endl;
  }
  getline(inCoords, dummy);
  double box[3];
  for (int i=0; i<3; ++i){
    double tmp1,tmp2;
    inCoords >> tmp1 >> tmp2;
    box[i]=tmp2-tmp1;
  }
  getline(inCoords, dummy);
  getline(inCoords, dummy);
  
  // create bins according to the cutoff distance
  int nbins[3]; double rbins[3];
  for (int i=0; i<3; ++i){
    nbins[i]=static_cast<int>(box[i]*LRcutMod1);
    if (nbins[i]==0) {cout << "\nToo large cutoff.\n"; exit(1);}
    rbins[i]=box[i]/nbins[i];
  }
  
  int nBinsTot=nbins[0]*nbins[1]*nbins[2];
  vector <vector <int>> binAtoms (nBinsTot);
  vector <int> nAtBin(nBinsTot, 0.0);

  double x[nAtom],y[nAtom],z[nAtom];
  for (int iAtom=0; iAtom<nAtom; ++iAtom){
    int iat, it;
    int tmpCoor[3], tmpCoor1[3];
    double tCoor[3], vol;
    inCoords >> iat >> it >> tCoor[0] >> tCoor[1] >> tCoor[2];
    // move all atoms into the 0,0,0 unit cell
    for (int j=0; j<3; ++j){
      tmpCoor[j] = static_cast<int>(tCoor[j]/box[j]);
      tmpCoor1[j] = static_cast<int>((tCoor[j]+box[j])/box[j]);
      if (tmpCoor[j]>0) tCoor[j]-=tmpCoor[j]*box[j];
      else if (tmpCoor[j]<0) tCoor[j]-=(tmpCoor[j]-1)*box[j];
      else if ((tmpCoor[j]==0)&&(tmpCoor1[j]==0)) tCoor[j]+=box[j];
    }
    x[iat-1]=tCoor[0];
    y[iat-1]=tCoor[1];
    z[iat-1]=tCoor[2];
    Type[iat-1]=it;
    int rLocBin[3];
    for (int i=0; i<3; ++i){
      rLocBin[i]=tCoor[i]/rbins[i];
      if (rLocBin[i]==nbins[i]) rLocBin[i]-=1;
    }
    int locBin = rLocBin[0]*nbins[1]*nbins[2]+rLocBin[1]*nbins[2]+rLocBin[2];
    binAtoms[locBin].push_back(iat-1);
    nAtBin[locBin]+=1;
  }
  getline(inCoords, dummy);
    
  int nTypeC = 0, nTypeE = 0;
  for (int iAtom=0; iAtom<nAtom; ++iAtom){
    if (Type[iAtom] == TypeC) nTypeC+=1;
    if (Type[iAtom] == TypeE) nTypeE+=1;
  }
  
  // not efficient
  // TODO: keep track of the earlier comminucated bins
  long long int count=0; // counting triangles
  for (int iAtom=0; iAtom<nAtom; ++iAtom){
    if (Type[iAtom]!=TypeC) continue;
      
    int irbin[3];
    irbin[0]=x[iAtom]/rbins[0];
    irbin[1]=y[iAtom]/rbins[1];
    irbin[2]=z[iAtom]/rbins[2];
    for (int ir=0; ir<3; ++ir){
      if (irbin[ir]==nbins[ir]) irbin[ir]-=1;
    }
      
    vector <int> iAtoms;
    int iNeigh=0;
    // go +- 1 bin in each direction
    // for a given atom (central), search for all atoms within 1 bin reach
    for (int ix=-1; ix<2; ++ix){
      int jxbin=irbin[0]+ix;
      if (jxbin<0) jxbin+=nbins[0]; // PBC
      else if (jxbin==nbins[0]) jxbin=0; // PBC
      for (int iy=-1; iy<2; ++iy){
        int jybin=irbin[1]+iy;
        if (jybin<0) jybin+=nbins[1]; // PBC
        else if (jybin==nbins[1]) jybin=0; // PBC
        for (int iz=-1; iz<2; ++iz){
          int jzbin=irbin[2]+iz;
          if (jzbin<0) jzbin+=nbins[2]; // PBC
          else if (jzbin==nbins[2]) jzbin=0; // PBC
          int locBin = jxbin*nbins[1]*nbins[2]+jybin*nbins[2]+jzbin;
          for (int iat=0; iat<nAtBin[locBin]; ++iat){
            int locAtom = binAtoms[locBin][iat];
            if (locAtom == iAtom) continue;
            iAtoms.push_back(locAtom);
            ++iNeigh;
          }
        }
      }
    }
      
    int locIndex;
    for (int j=0; j<iNeigh-1; ++j){
      int jAtom=iAtoms[j];
      if (Type[jAtom]!=TypeE) continue;

      double rij[3]{x[iAtom]-x[jAtom],y[iAtom]-y[jAtom],z[iAtom]-z[jAtom]};
      pbc(rij, box); // PBC
      double rij2=rij[0]*rij[0]+rij[1]*rij[1]+rij[2]*rij[2];
      if (rij2<=LRcut2){
        for (int k=j+1; k<iNeigh; ++k){
          int kAtom=iAtoms[k];
          if (Type[kAtom]!=TypeE) continue;

          double rik[3]{x[iAtom]-x[kAtom],y[iAtom]-y[kAtom],z[iAtom]-z[kAtom]};
          pbc(rik, box); // PBC
          double rik2=rik[0]*rik[0]+rik[1]*rik[1]+rik[2]*rik[2];
          if (rik2<=LRcut2){
            ++count;
            double rij1=sqrt(rij2);
            double rik1=sqrt(rik2);
            double cosjik=(rij[0]*rik[0]+rij[1]*rik[1]+rij[2]*rik[2])/(rij1*rik1);
            int Pij=static_cast<int>(rij1/dr);
            int Pik=static_cast<int>(rik1/dr);
            int Pcos=static_cast<int>((cosjik+1.)/da);
            double drij=nrGrid*rij1-int(nrGrid*rij1);
            double drik=nrGrid*rik1-int(nrGrid*rik1);
            double dcos=naGrid*(cosjik+1.)-int(naGrid*(cosjik+1.));

            double weight[2][2][2];
            weight[0][0][0]=(1.-drij)*(1.-drik)*(1.-dcos);
            weight[1][0][0]=drij*(1.-drik)*(1.-dcos);
            weight[0][1][0]=(1.-drij)*drik*(1.-dcos);
            weight[0][0][1]=(1.-drij)*(1.-drik)*dcos;
            weight[1][1][0]=drij*drik*(1.-dcos);
            weight[1][0][1]=drij*(1.-drik)*dcos;
            weight[0][1][1]=(1.-drij)*drik*dcos;
            weight[1][1][1]=drij*drik*dcos;

            // fill in ijk
            int baseIndex = Pij*nRnA+Pik*naGrid+Pcos;
            locIndex = baseIndex;
            Pr[locIndex]+=weight[0][0][0];
            locIndex = baseIndex + nRnA;
            Pr[locIndex]+=weight[1][0][0];
            locIndex = baseIndex + naGrid;
            Pr[locIndex]+=weight[0][1][0];
            locIndex = baseIndex + 1;
            Pr[locIndex]+=weight[0][0][1];
            locIndex = baseIndex + nRnA + naGrid;
            Pr[locIndex]+=weight[1][1][0];
            locIndex = baseIndex + nRnA + 1;
            Pr[locIndex]+=weight[1][0][1];
            locIndex = baseIndex + naGrid + 1;
            Pr[locIndex]+=weight[0][1][1];
            locIndex = baseIndex + nRnA + naGrid + 1;
            Pr[locIndex]+=weight[1][1][1];
            // fill in kji
            baseIndex = Pik*nRnA+Pij*naGrid+Pcos;
            locIndex = baseIndex;
            Pr[locIndex]+=weight[0][0][0];
            locIndex = baseIndex + nRnA;
            Pr[locIndex]+=weight[1][0][0];
            locIndex = baseIndex + naGrid;
            Pr[locIndex]+=weight[0][1][0];
            locIndex = baseIndex + 1;
            Pr[locIndex]+=weight[0][0][1];
            locIndex = baseIndex + nRnA + naGrid;
            Pr[locIndex]+=weight[1][1][0];
            locIndex = baseIndex + nRnA + 1;
            Pr[locIndex]+=weight[1][0][1];
            locIndex = baseIndex + naGrid + 1;
            Pr[locIndex]+=weight[0][1][1];
            locIndex = baseIndex + nRnA + naGrid + 1;
            Pr[locIndex]+=weight[1][1][1];
          }
        }
      }
    }
  }
  cout<<"Found "<<count<<" different triangles within the cutoff\n";
    
  double Vol=box[0]*box[1]*box[2];
  aveVolC += Vol/nTypeC;
  aveVolE += Vol/nTypeE;
  double prefac=dr2da1/nConf/nTypeC/2;    // nType2
  for (int ia=0; ia<naGrid; ++ia){
    for (int ir=1; ir<nrGrid; ++ir){
      double ri=ir*dr;
      double denomi=pi2i/(ri*ri);
      for (int jr=rBinMin; jr<rBinMax; ++jr){
        double rj=jr*dr;
        double denomj=pi2i/(rj*rj);
        int locIndex = ir*nRnA+jr*naGrid+ia;
        g3[ia][ir]+=Pr[locIndex]*prefac*denomi*denomj;
      }
    }
  }
}

//******************************************************************************//

void OutputData(vector< vector< double > > g3){
  aveVolC/=nConf;
  aveVolE/=nConf;
  double g3Coeff = aveVolC*aveVolE/(rBinMax-rBinMin);

  ofstream Output;
  Output.open(outputFile);
  for (int ir=1; ir<nrGrid; ++ir){
    double ri=ir*dr;
    for (int ia=0; ia<naGrid; ++ia){
      double acos=ia*da;
      // due to the implementation of the method (see the README file)
      // the boundary surfaces lack a factor of two (because their "bin size" is only one half)
      if (ir==nrGrid-1) g3[ia][ir]*=2;
      if ((ia==0) || (ia==naGrid-1)) g3[ia][ir]*=2;
      Output<<setw(10)<<ri<<setw(10)<<acos-1.<<setw(15)<<g3[ia][ir]*g3Coeff<<endl;
    }
    Output<<"\n";
  }
  Output.clear();
}

//******************************************************************************//

void pbc(double r[3], double box[3]){
  for (int iDim=0; iDim<3; ++iDim){
    if (r[iDim] > box[iDim]/2) r[iDim]-=box[iDim];
    else if (r[iDim] < -box[iDim]/2) r[iDim]+=box[iDim];
  }
}
