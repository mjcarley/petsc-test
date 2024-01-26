#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include <petscksp.h>
#include <petscoptions.h>

PetscErrorCode petsc_MatMult_real(Mat mat, Vec x, Vec y) ;

static void matrix_vector_mul(double *A, int m, int n, double *x, double *y)

{
  int i, j ;
  
  for ( i = 0 ; i < m ; i ++ ) {
    y[i] = 0.0 ;
    for ( j = 0 ; j < n ; j ++ ) {
      y[i] += A[i*n+j]*x[j] ;
    }
  }
  
  return ;
}

PetscErrorCode petsc_MatMult_real(Mat mat, Vec x, Vec y)

{
  double *A, *xx, *yy ;
  void *petsc_ctx ;
  int i, j, m, n ;
  
  PetscCall(MatShellGetContext(mat, &petsc_ctx)) ;  

  A = petsc_ctx ;

  PetscCall(VecGetArrayRead(x, (const double **)(&xx))) ;
  PetscCall(VecGetArrayRead(y, (const double **)(&yy))) ;

  PetscCall(MatGetSize(mat, &m, &n)) ;

  matrix_vector_mul(A, m, n, xx, yy) ;
  
  return 0 ;
}

static double *random_matrix(int n)

{
  double *A ;
  int i ;
  
  A = (double *)malloc(n*n*sizeof(double)) ;

  for ( i = 0 ; i < n*n ; i ++ ) {
    A[i] = 2.0*(drand48() - 0.5) ;
  }
  for ( i = 0 ; i < n ; i ++ ) {
    A[i*n + i] += 4.0 ;
  }
  
  return A ;
}

int main(int argc, char **argv)

{
  PetscInt n ;
  KSP         ksp ;      /*linear solver context*/
  PC          pc ;       /*preconditioner context*/
  int i, j, size, gmres_max_iter, direct ;
  char ch, *help = "", *kspfile = NULL ;  
  Vec b, s, ref ; /*RHS, solution*/
  Mat A ;    /*linear system matrix*/
  double *matrix, *pb, *ps, *pr, tol ;
  
  n = 64 ; gmres_max_iter = 10 ; tol = 1e-6 ;
  direct = 0 ;
  while ( (ch = getopt(argc, argv, "dk:n:"))
	  != EOF ) {
    switch ( ch ) {
    case 'd': direct = 1 ; break ;
    case 'k': kspfile = strdup(optarg) ; break ;
    case 'n': n = atoi(optarg) ; break ;
    }
  }

  i = 0 ;
  if ( kspfile == NULL ) {
    PetscCall(PetscInitialize(&i, &argv, (char *)0, help)) ;
  } else {
    PetscCall(PetscInitialize(&i, &argv, kspfile, help)) ;
  }
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  PetscCall(VecCreate(PETSC_COMM_SELF, &b));
  PetscCall(PetscObjectSetName((PetscObject)b, "rhs"));
  PetscCall(VecSetType(b, VECSEQ)) ; 
  PetscCall(VecSetSizes(b, PETSC_DECIDE, n));
  PetscCall(VecDuplicate(b, &s));
  PetscCall(PetscObjectSetName((PetscObject)s, "solution"));
  PetscCall(VecDuplicate(b, &ref));

  /*random matrix for system*/
  matrix = random_matrix(n) ;
  /*random reference solution*/
  PetscCall(VecGetArrayRead(ref, (const double **)(&pr))) ;
  for ( i = 0 ; i < n ; i ++ ) {
    pr[i] = 2.0*(drand48() - 0.5) ;
  }
  
  PetscCall(MatCreateShell(PETSC_COMM_SELF, n, n, n, n, matrix, &A)) ;
  PetscCall(MatShellSetOperation(A, MATOP_MULT,
				 (void *)petsc_MatMult_real)) ;
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
    
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /*create the right hand side from the reference solution*/
  if ( !direct ) {
    PetscCall(MatMult(A, ref, b)) ;
  } else {
    PetscCall(VecGetArrayRead(ref, (const double **)(&pr))) ;
    PetscCall(VecGetArrayRead(b  , (const double **)(&pb))) ;
    matrix_vector_mul(matrix, n, n, pr, pb) ;
  }
  
  PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED)) ;
  PetscCall(KSPSetType(ksp, KSPGMRES));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE)) ;
  PetscCall(KSPSetTolerances(ksp, tol, PETSC_DEFAULT, PETSC_DEFAULT,
			     gmres_max_iter));
  
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, b, s));
  PetscCall(KSPGetIterationNumber(ksp, &i)) ;
  fprintf(stderr, "%s: %d iterations\n", argv[0], i) ;

  return 0 ;
}
