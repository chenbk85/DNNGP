#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <math.h>
using namespace std;

#include "kvpar.h"

#include "mkl_vsl.h"
#include <mkl.h>

// #define MATHLIB_STANDALONE
// #include <Rmath.h>

void show(int *a, int n);
void show(double *a, int n);
void show(int *a, int r, int c);
void show(double *a, int r, int c);
void zeros(double *a, int n);
void writeRMatrix(string outfile, double * a, int nrow, int ncol);
void writeRMatrix(string outfile, int * a, int nrow, int ncol);

double logit(double theta, double a, double b){
  return log((theta-a)/(b-theta));
}

double logitInv(double z, double a, double b){
  return b-(b-a)/(1+exp(z));
}

// void mvrnorm(VSLStreamStatePtr rng, double *des, double *mu, double *cholCov, int dim){
  
//   int i;
//   int inc = 1;
//   double one = 1.0;
//   double zero = 0.0;
  
//   //make some std norm draws
//   vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, rng, dim, des, 0, 1);
//   // for(i = 0; i < dim; i++){
//   //   des[i] = rnorm(0.0, 1.0);
//   // }

//   //mult this vector by the lower triangle of the cholCov
//   dtrmv("L", "N", "N", &dim, cholCov, &dim, des, &inc);
  
//   //add the mean to the result
//   daxpy(&dim, &one, mu, &inc, des, &inc);
  
// }


void printList(int **m, int length){
  int i,j;
  
  for(i = 0; i < length; i++){
    
    cout << "[" << m[i][0] << "] ";
    
    for(j = 0; j < m[i][0]; j++){
      cout << m[i][1+j] << " ";
    }
    cout << endl;
  }
  
}

int which(int a, int *b, int length){
  int i;
  for(i = 0; i < length; i++){
    if(a == b[i]){
      return(i);
    }
  }

  cout << "c++ error: which failed" << endl;
  return -9999;
}

double C(double h, double u, double sigmaSq, double *phi){
  //a phi[0] time
  //c phi[1] space
  
  //return sigmaSq/(phi[0]*pow(u, 2) + 1)*(1 + phi[1]*h/pow(phi[0]*pow(u,2) + 1, 0.5))*exp(-phi[1]*h/pow(phi[0]*pow(u,2) + 1, 0.5));
  return sigmaSq/(phi[0]*pow(u, 2) + 1)*exp(-phi[1]*h/pow(phi[0]*pow(u,2) + 1, 0.5));
}

double dist1(double &a, double &b){
  return sqrt(pow(a-b,2));
}

double dist2(double &a1, double &a2, double &b1, double &b2){
  return(sqrt(pow(a1-b1,2)+pow(a2-b2,2)));
}

void updateNTheta(int n, int **E, int **NTheta, double *coords,  double sigmaSq, double *theta, int nEMax, double *E_tmp){
  
  int i, j, k, l;
  double v, u, h;
  
  #pragma omp parallel for private(j, h, u, k, v, l)
  for(i = 0; i < n; i++){
    
    for(j = 0; j < E[i][0]; j++){
      
      h = dist2(coords[i], coords[n+i], coords[E[i][1+j]], coords[n+E[i][1+j]]);
      u = dist1(coords[2*n+i], coords[2*n+E[i][1+j]]);
      E_tmp[(i*nEMax)+j] = C(h, u, sigmaSq, theta);
      
      //copy eligible list to NTheta for subsequent sort
      NTheta[i][1+j] = E[i][1+j];
    }
    
    //sort the eligible list in terms of the covariance and store the result in NTheta (do something faster than a bubble sort eventually)
    // for(j = 0; j < E[i][0]; j++){
    
    //   for(k = 0; k < E[i][0]-1; k++){
    
    // 	if(E_tmp[(i*nEMax)+k] < E_tmp[(i*nEMax)+k+1]){
    
    // 	  v = E_tmp[(i*nEMax)+k]; l = NTheta[i][1+k];
    // 	  E_tmp[(i*nEMax)+k] = E_tmp[(i*nEMax)+k+1]; NTheta[i][1+k] = NTheta[i][2+k];
    // 	  E_tmp[(i*nEMax)+k+1] = v; NTheta[i][2+k] = l;
    
    // 	}
    //   }
    // }
    
    for(j = 1; j <= E[i][0]-1; j++){
      k = j;  
      while(k > 0 && E_tmp[(i*nEMax)+k] > E_tmp[(i*nEMax)+k-1]) {
    	v = E_tmp[(i*nEMax)+k]; l = NTheta[i][1+k];
    	E_tmp[(i*nEMax)+k] = E_tmp[(i*nEMax)+k-1]; NTheta[i][1+k] = NTheta[i][1+k-1];
    	E_tmp[(i*nEMax)+k-1] = v; NTheta[i][1+k-1] = l;
    	k--;
      }
    }
  }
  
}

void updateU(int n, int **NTheta, int **U, int **UE){
  
  int k, i, l, j;
  
  // #pragma omp parallel for private(l, j)
  // for(k = 0; k < n; k++){
  //   U[k][0] = 0;
  //   l = 0;
  //   for(i = 0; i < n; i++){
  //     for(j = 0; j < NTheta[i][0]; j++){
  // 	if(NTheta[i][1+j] == k){
  // 	  U[k][1+l] = i;
  // 	  l++; 
  // 	}
  //     }
  //   }
  //   U[k][0] = l;
  // }

  #pragma omp parallel for private(l, j)
  for(k = 0; k < n; k++){
    U[k][0] = 0;
    l = 0;

    for(i = 0; i < UE[k][0]; i++){

      for(j = 0; j < NTheta[UE[k][1+i]][0]; j++){

  	if(NTheta[UE[k][1+i]][1+j] == k){

  	  U[k][1+l] = UE[k][1+i];
  	  l++; 
 	  
  	}
      }
    }
    U[k][0] = l;
  }

  // #pragma omp parallel for private(i, j, l)
  // for(k = 0; k < n; k++){
    
  //   //allocate
  //   for(i = 0, l = 0; i < n; i++){
  //     for(j = 0; j < NTheta[i][0]; j++){
  // 	if(NTheta[i][1+j] == k){
  // 	  l++;
  // 	}
  //     }
  //   }
 
  //   //if the new neighbor list is larger than the old one then allocate more space
  //   if(U[k][0] < l){
  //     delete[] U[k];
  //     U[k] = new int[l+1];
  //   }

  //   U[k][0] = l;
    
  //   //load
  //   for(i = 0, l = 0; i < n; i++){
  //     for(j = 0; j < NTheta[i][0]; j++){
  // 	if(NTheta[i][1+j] == k){
  // 	  U[k][1+l] = i;
  // 	  l++;
  // 	}
  //     }
  //   }
    
  // }
  
}

void updateBF(int n, int **NTheta, int **U, double *coords, double sigmaSq, double *theta, double **CNTheta, double **ClNTheta, double **B, double *F){
   
  int i, j, k, info;
  double h, u;
  int incOne = 1;
  double one = 1;
  double zero = 0;
  char const *lower = "L";
  
#pragma omp for private(j, k, h, u, info)
  for(i = 0; i < n; i++){
    
    for(j = 0; j < NTheta[i][0]; j++){
      h = dist2(coords[i], coords[n+i], coords[NTheta[i][1+j]], coords[n+NTheta[i][1+j]]);
      u = dist1(coords[2*n+i], coords[2*n+NTheta[i][1+j]]);
      ClNTheta[i][j] = C(h, u, sigmaSq, theta);
    }
    
    for(j = 0; j < NTheta[i][0]; j++){
      for(k = 0; k < NTheta[i][0]; k++){
	h = dist2(coords[NTheta[i][1+k]], coords[n+NTheta[i][1+k]], coords[NTheta[i][1+j]], coords[n+NTheta[i][1+j]]);
	u = dist1(coords[2*n+NTheta[i][1+k]], coords[2*n+NTheta[i][1+j]]);
	CNTheta[i][j*NTheta[i][0]+k] = C(h, u, sigmaSq, theta);
      }
    }
    
    if(NTheta[i][0] > 0){
      dpotrf(lower, &NTheta[i][0], CNTheta[i], &NTheta[i][0], &info); if(info != 0){cout << "c++ 1 error: dpotrf failed" << endl;}
      dpotri(lower, &NTheta[i][0], CNTheta[i], &NTheta[i][0], &info); if(info != 0){cout << "c++ 2 error: dpotri failed" << endl;}
      dsymv(lower, &NTheta[i][0], &one, CNTheta[i], &NTheta[i][0], ClNTheta[i], &incOne, &zero, B[i], &incOne);	  
      F[i] = sigmaSq - ddot(&NTheta[i][0], B[i], &incOne, ClNTheta[i], &incOne);
    }else{
      F[i] = sigmaSq;
    }
  }
}


int main(int argc, char **argv){
  
  /*****************************************
                common variables
  *****************************************/
  int i, j, k, l, s, info;
  char const *lower = "L";
  char const *upper = "U";
  char const *nUnit = "N";
  char const *yUnit = "U";
  char const *ntran = "N";
  char const *ytran = "T";
  char const *rside = "R";
  char const *lside = "L";
  const double one = 1.0;
  const double negOne = -1.0;
  const double zero = 0.0;
  const int incOne = 1;
  
  /*****************************************
                     Set-up
  *****************************************/
  string parfile;
  if(argc > 1)
    parfile = argv[1];
  else
    parfile = "pfile";
  
  kvpar par(parfile);
  
  int nThreads; par.getVal("n.threads", nThreads);
  omp_set_num_threads(nThreads);

  int p; par.getVal("p", p);
  int pp = p*p;
  int n; par.getVal("n", n);
  int m; par.getVal("m", m);
  string outFile; par.getVal("out.file", outFile);

  int seed; par.getVal("seed", seed);
  
  ifstream file;

  //coord file
  string coordFile; par.getVal("coord.file", coordFile);
  file.open(coordFile.c_str());
  if(!file.is_open()){cout << "no coords file" << endl; return 0;}

  double *coords = new double[n*3];
  for(i = 0; i < 3*n; i++){
    file >> coords[i];
  }
  file.close();
  
  //x file
  string xFile; par.getVal("x.file", xFile);
  file.open(xFile.c_str());
  if(!file.is_open()){cout << "no x file" << endl; return 0;}

  double *X = new double[n*p];
  for(i = 0; i < p*n; i++){
    file >> X[i];  
  }
  file.close();

  //y file
  string yFile; par.getVal("y.file", yFile);
  file.open(yFile.c_str());
  if(!file.is_open()){cout << "no y file" << endl; return 0;}
  
  double *Y = new double[n];
  for(i = 0; i < n; i++){
    file >> Y[i];
  }
  file.close();

  //y obs indx
  string yIndxFile; par.getVal("y.indx.file", yIndxFile);
  file.open(yIndxFile.c_str());
  if(!file.is_open()){cout << "no y indx file" << endl; return 0;}
  
  int *yIndx = new int[n];
  int n0 = 0;
  for(i = 0; i < n; i++){
    file >> yIndx[i];
    n0 += yIndx[i];
  }
  file.close();

  //nE
  string nEFile; par.getVal("nE.file", nEFile);
  file.open(nEFile.c_str());
  if(!file.is_open()){cout << "no nE file" << endl; return 0;}

  int *nE = new int[n-1];
  for(i = 0; i < n-1; i++){
    file >> nE[i];
  }
  file.close();

  //E
  string EFile; par.getVal("E.file", EFile);
  file.open(EFile.c_str());
  if(!file.is_open()){cout << "no nE file" << endl; return 0;}
  
  int **E = new int*[n];

  for(i = 0; i < n; i++){
    
    if(i == 0){
      E[i] = new int[1];
      E[i][0] = 0;
    }else{
      E[i] = new int[1+nE[i-1]];
      E[i][0] = nE[i-1];
    }

  }
  
  double aa;
  for(i = 0; i < n; i++){
    for(j = 0; j < E[i][0]; j++){
      //file >> E[i][1+j];
      file >> aa;
      E[i][1+j] = aa;
    }
  }
  file.close();
	   
  // printList(E, n);
  // exit(1);

   
  //priors and starting
  vector<double> betaStarting; par.getVal("beta.starting", betaStarting);

  double tauSq_b; par.getVal("tauSq.b", tauSq_b);
  double tauSqStarting; par.getVal("tauSq.starting", tauSqStarting);

  double sigmaSq_b; par.getVal("sigmaSq.b", sigmaSq_b);
  double sigmaSqStarting; par.getVal("sigmaSq.starting", sigmaSqStarting);

  double phi1_a; par.getVal("phi.1.a", phi1_a);
  double phi1_b; par.getVal("phi.1.b", phi1_b);
  double phi1Starting; par.getVal("phi.1.starting", phi1Starting);
  double phi1Tuning; par.getVal("phi.1.tuning", phi1Tuning);

  double phi2_a; par.getVal("phi.2.a", phi2_a);
  double phi2_b; par.getVal("phi.2.b", phi2_b);
  double phi2Starting; par.getVal("phi.2.starting", phi2Starting);
  double phi2Tuning; par.getVal("phi.2.tuning", phi2Tuning);

  int nSamples; par.getVal("n.samples", nSamples);
  int start; par.getVal("start", start);
  int nReport; par.getVal("n.report", nReport);

  int nSamplesSaved = nSamples-start;
  
  /*****************************************
         Set-up MCMC sample matrices etc.
  *****************************************/ 
  //parameters
  int sigmaSqIndx = 0; 
  int tauSqIndx = 1; 
  
  int nTheta = 2;
  int aIndx = 0;
  int cIndx = 1;
  
  //starting
  double *theta = new double[nTheta];
  theta[aIndx] = phi1Starting;
  theta[cIndx] = phi2Starting;
  
  //tuning
  double *tuning = new double[nTheta];
  tuning[aIndx] = phi1Tuning;
  tuning[cIndx] = phi2Tuning;
  
  //return stuff  
  double *betaSamples = new double[p*nSamplesSaved];
  double *varSamples = new double[2*nSamplesSaved];
  double *thetaSamples = new double[nTheta*nSamplesSaved];
  double *wSamples = new double[n*nSamplesSaved];
  
  /*****************************************
       Set-up MCMC alg. vars. matrices etc.
  *****************************************/
  int status=1, batchAccept=0;
  double logMHRatio =0, logPostCurrent = 0, logPostCand = 0, det = 0;
  double h, u, v, accept=0;
  double *thetaCand = new double[nTheta];
  
  double *tmp_m = new double[m*n];
  double *tmp_n = new double[n];
  double *tmp_pp = new double[pp];
  double *tmp_p = new double[p];
  double *tmp_p2 = new double[p];
  
  double *beta = new double[p]; 
  for(i = 0; i < p; i++){
    beta[i] = betaStarting[i];
  }
  
  double *XtX = new double[pp]; zeros(XtX, pp);
  for(i = 0; i < n; i++){
    if(yIndx[i] == 1){
      dgemm(ytran, ntran, &p, &p, &incOne, &one, &X[i], &n, &X[i], &n, &one, XtX, &p);
    }
  }
  
  double *VBeta = new double[p];
  for(i = 0; i < p; i++){
    VBeta[i] = 10000;
  }

  //The first element in the ith NTheta will be m but I will allocate the corresponding vector 
  //of length specified in the eligible list. This will simplify things later. If storage becomes
  //a problem then I can shorten the this vector to the appropriate m (but it should be okay). 
  int **NTheta = new int*[n];
  
  for(i = 0; i < n; i++){
    
    NTheta[i] = new int[1+E[i][0]];
    
    if(i < m){
      NTheta[i][0] = i;
    }else{
      NTheta[i][0] = m;
    }
    
    //just zero it out for fun
    for(j = 0; j < E[i][0]; j++, k++){
      NTheta[i][1+j] = 0;
    }
    
  }
  //printList(NTheta, n);
  
  //maximum length vector in the eligible list. I'll need to make more workspace for parallel.
  int nEMax = 0;
  for(i = 0; i < n; i++){
    if(nEMax < nE[i-1]){nEMax = nE[i-1];}
  }
  
  double *E_tmp = new double[n*nEMax];
  
  double **B = new double*[n];
  double *F = new double[n];
  double **ClNTheta = new double*[n];
  double **CNTheta = new double*[n];
  
  for(i = 0; i < n; i++){//note, if NTheta[i][0] == 0 then no space is allocated which should be okay since it will never be referenced. I'm doing this just in case we try different ordering approaches.
    ClNTheta[i] = new double[NTheta[i][0]];
    CNTheta[i] = new double[NTheta[i][0]*NTheta[i][0]];
    B[i] = new double [NTheta[i][0]]; 
  }
  
  double sigmaSq = sigmaSqStarting;
  double tauSq = tauSqStarting;
  double mu, var, aij, a, b, aCand;
  
  double *w = new double[n]; zeros(w, n);
  for(i = 0; i < n; i++){
    if(yIndx[i] == 1){
      w[i] = 0.0;//Y[i] - ddot(&p, &X[i], &n, beta, &incOne);
    }
  }
  
  int **U = new int*[n]; 
  int **UE = new int*[n];
  
  #pragma omp parallel for private(l, j)
  for(k = 0; k < n; k++){
    
    //allocate
    for(i = 0, l = 0; i < n; i++){
      for(j = 0; j < E[i][0]; j++){
  	if(E[i][1+j] == k){
  	  l++;
  	}
      }
    }
    
    UE[k] = new int[l+1];
    UE[k][0] = l;
    
    U[k] = new int[l+1];
    
    //load
    for(i = 0, l = 0; i < n; i++){
      for(j = 0; j < E[i][0]; j++){
  	if(E[i][1+j] == k){
  	  UE[k][1+l] = i;
  	  l++;
  	}
      }
    }
    
  }
  
  //rng
  VSLStreamStatePtr rng;
  vslNewStream(&rng, VSL_BRNG_MT19937, seed);
  
  double t1, t2;
  int ss = 0;

  cout << "Sampling" << endl;
  t1 = omp_get_wtime();

  for(s = 0; s < nSamples; s++){    
    
    //update NTheta given sigmaSq and theta
    //t1 = omp_get_wtime();
    updateNTheta(n, E, NTheta, coords, sigmaSq, theta, nEMax, E_tmp);
    //t2 = omp_get_wtime();
    //cout << "updateNTheta " << t2-t1 << endl;
    
    //update U given NTheta
    //t1 = omp_get_wtime();
    updateU(n, NTheta, U, UE);
    //t2 = omp_get_wtime();
    //cout << "updateU " << t2-t1 << endl;
    
    //update B and F
    //t1 = omp_get_wtime();
    updateBF(n, NTheta, U, coords, sigmaSq, theta, CNTheta, ClNTheta, B, F);
    //t2 = omp_get_wtime();
    //cout << "updateBF " << t2-t1 << endl;
    
    /////////////////////
    //update beta
    /////////////////////
    //t1 = omp_get_wtime();
    
    zeros(tmp_p, p);
    for(i = 0; i < n; i++){
      if(yIndx[i] == 1){
	for(j = 0; j < p; j++){
	  tmp_p[j] += X[i+j*n]*(Y[i] - w[i])/tauSq;
	}
      }
    }
	      
    for(i = 0; i < pp; i++){
      tmp_pp[i] = XtX[i]/tauSq;
    }

    for(i = 0; i < p; i++){
      tmp_pp[i*p+i] += 1/VBeta[i];
    }
    
    dpotrf(lower, &p, tmp_pp, &p, &info); if(info != 0){cout << "c++ error: 3 dpotrf failed" << endl;}
    dpotri(lower, &p, tmp_pp, &p, &info); if(info != 0){cout << "c++ error: 4 dpotri failed" << endl;}
    
    dsymv(lower, &p, &one, tmp_pp, &p, tmp_p, &incOne, &zero, tmp_p2, &incOne);
    
    dpotrf(lower, &p, tmp_pp, &p, &info); if(info != 0){cout << "c++ error: 5 dpotrf failed" << endl;}
    
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, rng, p, beta, 0, 1);
    dtrmv(lower, ntran, ntran, &p, tmp_pp, &p, beta, &incOne);
    daxpy(&p, &one, tmp_p2, &incOne, beta, &incOne);
    
    //t2 = omp_get_wtime();
    //cout << "update beta " << t2-t1 << endl;
    if(s >= start){
      dcopy(&p, beta, &incOne, &betaSamples[ss*p], &incOne);
    }
    /////////////////////
    //update w
    /////////////////////
    //t1 = omp_get_wtime();
    #pragma omp for private(j, a, v, aij, b, mu, var)
    for(i = 0; i < n; i++){
      a = 0;
      v = 0;
      
      if(U[i][0] > 0){	  
	for(j = 0; j < U[i][0]; j++){
	  b = 0;
	  for(k = 0; k < NTheta[U[i][1+j]][0]; k++){	      
	    if(NTheta[U[i][1+j]][1+k] != i){
	      b += B[U[i][1+j]][which(NTheta[U[i][1+j]][1+k], &(NTheta[U[i][1+j]][1]), NTheta[U[i][1+j]][0])] * w[NTheta[U[i][1+j]][1+k]];
	    }	 
	  }	 
	  aij = w[U[i][1+j]] - b;	    
	  a += B[U[i][1+j]][which(i, &(NTheta[U[i][1+j]][1]), NTheta[U[i][1+j]][0])]*aij/F[U[i][1+j]];
	  v += pow(B[U[i][1+j]][which(i, &(NTheta[U[i][1+j]][1]), NTheta[U[i][1+j]][0])],2)/F[U[i][1+j]];
	}
      }
      
      if(NTheta[i][0] > 0){  
	for(j = 0; j < NTheta[i][0]; j++){
	  tmp_m[(i*m)+j] = w[NTheta[i][1+j]];
	}	  
	mu = (Y[i] - ddot(&p, &X[i], &n, beta, &incOne))*yIndx[i]/tauSq + ddot(&NTheta[i][0], B[i], &incOne, &tmp_m[(i*m)], &incOne)/F[i] + a;
      }else{
	mu = (Y[i] - ddot(&p, &X[i], &n, beta, &incOne))*yIndx[i]/tauSq + a;
      }
      
      var = 1.0/(yIndx[i]/tauSq + 1.0/F[i] + v);
      
      //w[i] = rnorm(var*mu, sqrt(var));
      vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, rng, 1, &w[i], var*mu, sqrt(var));
    }
    
    //t2 = omp_get_wtime();
    //cout << "update w " << t2-t1 << endl;
    if(s >= start){
      dcopy(&n, w, &incOne, &wSamples[ss*n], &incOne);
    }
    /////////////////////
    //update tau^2
    /////////////////////
    //t1 = omp_get_wtime();
    a = 0;
    #pragma omp parallel for reduction(+:a)
    for(i = 0; i < n; i++){
      if(yIndx[i] == 1){
	a += pow(Y[i] - w[i] - ddot(&p, &X[i], &n, beta, &incOne), 2);
      }
    }
    // tauSq = varSamples[2*s+tauSqIndx] = 1.0/rgamma(2+n/2.0, 
    // 					     1.0/(tauSq_b+0.5*ddot(&n, tmp_n, &incOne, tmp_n, &incOne)));
    
    vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM, rng, 1, &tauSq, 2+n0/2.0, 0, 1.0/(tauSq_b+0.5*a));
    
    tauSq = 1/tauSq;

    //t2 = omp_get_wtime();
    //cout << "update tau^2 " << t2-t1 << endl;
    
    if(s >= start){
      varSamples[2*ss+tauSqIndx] = tauSq;
    }
    
    /////////////////////
    //update theta
    /////////////////////
    //t1 = omp_get_wtime();
    a = 0;
    det = 0;
   #pragma omp parallel for private(j, b) reduction(+:a, det)
    for(i = 0; i < n; i++){
      if(NTheta[i][0] > 0){
	
	for(j = 0; j < NTheta[i][0]; j++){
	  tmp_m[(i*m)+j] = w[NTheta[i][1+j]];
	}
	
	b = w[i] - ddot(&NTheta[i][0], B[i], &incOne, &tmp_m[(i*m)], &incOne);
      }else{
	b = w[i];
      }	
      a += b*b/F[i];
      det += 2*log(sqrt(F[i]));
    }
    
    logPostCurrent = -0.5*det - 0.5*a; 
    logPostCurrent += log(theta[0] - phi1_a) + log(phi1_b - theta[0]); 
    logPostCurrent += log(theta[1] - phi2_a) + log(phi2_b - theta[1]); 
    
    //t2 = omp_get_wtime();
    //cout << "update theta 1 " << t2-t1 << endl;
    
    //
    //propose extract and transform
    //
    //t1 = omp_get_wtime();
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, rng, 1, &v, logit(theta[0], phi1_a, phi1_b), sqrt(tuning[0]));
    thetaCand[0] = logitInv(v, phi1_a, phi1_b);
    
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, rng, 1, &v, logit(theta[1], phi2_a, phi2_b), sqrt(tuning[1]));
    thetaCand[1] = logitInv(v, phi2_a, phi2_b);
    
    //t2 = omp_get_wtime();
    //cout << "update theta 2 " << t2-t1 << endl;
    
    //update NTheta given sigmaSq and theta
    updateNTheta(n, E, NTheta, coords, sigmaSq, thetaCand, nEMax, E_tmp);
    
    //update U given NTheta
    updateU(n, NTheta, U, UE);
    
    //update B and F
    updateBF(n, NTheta, U, coords, sigmaSq, thetaCand, CNTheta, ClNTheta, B, F);
    
    //t1 = omp_get_wtime();
    aCand = 0;
    det = 0;
    #pragma omp parallel for private(j, b) reduction(+:aCand, det)
    for(i = 0; i < n; i++){
      if(NTheta[i][0] > 0){
	
	for(j = 0; j < NTheta[i][0]; j++){
	  tmp_m[(i*m)+j] = w[NTheta[i][1+j]];
	}
	
	b = w[i] - ddot(&NTheta[i][0], B[i], &incOne,& tmp_m[(i*m)], &incOne);
      }else{
	b = w[i];
      }	
      aCand += b*b/F[i];
      det += 2*log(sqrt(F[i]));
    }
    
    logPostCand = -0.5*det - 0.5*aCand;  
    logPostCand += log(thetaCand[0] - phi1_a) + log(phi1_b - thetaCand[0]); 
    logPostCand += log(thetaCand[1] - phi2_a) + log(phi2_b - thetaCand[1]); 
    
    //t2 = omp_get_wtime();
    //cout << "update theta 3 " << t2-t1 << endl;
    
    //
    //MH accept/reject	
    //      
    //t1 = omp_get_wtime();
    logMHRatio = logPostCand - logPostCurrent;
    
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rng, 1, &v, 0, 1);
    
    if(v <= exp(logMHRatio)){
      accept++;
      batchAccept++;
      dcopy(&nTheta, thetaCand, &incOne, theta, &incOne);
      a = aCand;//for sigma^2
    }
    
    //t2 = omp_get_wtime();
    //cout << "update theta 4 " << t2-t1 << endl;
    
    if(s >= start){
      dcopy(&nTheta, theta, &incOne, &thetaSamples[ss*nTheta], &incOne);
    }

    /////////////////////
    //update sigma^2
    /////////////////////
    //   sigmaSq = REAL(varSamples_r)[2*s+sigmaSqIndx] = 1.0/rgamma(sigmaSqIGa+n/2.0, 
    //   							    1.0/(sigmaSqIGb+0.5*a*sigmaSq)); 
    //t1 = omp_get_wtime();
    vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM, rng, 1, &sigmaSq, 2+n/2.0, 0, 1.0/(sigmaSq_b+0.5*a*sigmaSq));

    sigmaSq = 1/sigmaSq;

    if(s >= start){
      varSamples[2*ss+sigmaSqIndx] = sigmaSq;
      ss++;
    }


    //t2 = omp_get_wtime();
    //cout << "update sigma^2 " << t2-t1 << endl;
    //cout << "--------------------" << endl;
    
    //report
    if(status == nReport){
      
      cout << "Sampled: " << 100.0*s/nSamples << "%, " << s << " of " << nSamples << endl;
      cout << "Report interval Metrop. Acceptance rate: " << 100.0*batchAccept/nReport << endl;
      cout << "Overall Metrop. Acceptance rate: " << 100.0*accept/s << endl;
      cout << "-----------------" << endl;
      status = 0;
      batchAccept = 0;
    }
    
    status++;
    
  }//end sample loop
  
  t2 = omp_get_wtime();
  cout << "time " << t2-t1 << endl;

  string out = "beta.samples-"+outFile;
  writeRMatrix(out, betaSamples, p, nSamplesSaved);
  
  out = "w.samples-"+outFile;
  writeRMatrix(out, wSamples, n, nSamplesSaved);
  
  out = "var.samples-"+outFile;
  writeRMatrix(out, varSamples, 2, nSamplesSaved);
  
  out = "theta.samples-"+outFile;
  writeRMatrix(out, thetaSamples, nTheta, nSamplesSaved);
  
  return(1);
  
}


void writeRMatrix(string outfile, double * a, int nrow, int ncol){
    ofstream file(outfile.c_str());
    if ( !file ) {
      cerr << "Data file could not be opened." << endl;
      exit(1);
    }
  

  for(int i = 0; i < nrow; i++){
    for(int j = 0; j < ncol-1; j++){
      file << setprecision(10) << fixed << a[j*nrow+i] << "\t";
    }
    file << setprecision(10) << fixed << a[(ncol-1)*nrow+i] << endl;    

  }
  file.close();
}


void writeRMatrix(string outfile, int* a, int nrow, int ncol){
    ofstream file(outfile.c_str());
    if ( !file ) {
      cerr << "Data file could not be opened." << endl;
      exit(1);
    }
  

  for(int i = 0; i < nrow; i++){
    for(int j = 0; j < ncol-1; j++){
      file << fixed << a[j*nrow+i] << "\t";
    }
    file << fixed << a[(ncol-1)*nrow+i] << endl;    

  }
  file.close();
}

void show(double *a, int n){
  for(int i = 0; i < n; i++)
    cout << setprecision(20) << fixed << a[i] << endl;
}


void show(int *a, int n){
  for(int i = 0; i < n; i++)
    cout << fixed << a[i] << endl;
}


void zeros(double *a, int n){
  for(int i = 0; i < n; i++)
    a[i] = 0.0;
}


void show(double *a, int r, int c){

  for(int i = 0; i < r; i++){
    for(int j = 0; j < c; j++){

      cout << fixed << a[j*r+i] << "\t";
    }
    cout << endl;
  }
}


void show(int *a, int r, int c){

  for(int i = 0; i < r; i++){
    for(int j = 0; j < c; j++){

      cout << fixed << a[j*r+i] << "\t";
    }
    cout << endl;
  }
}
