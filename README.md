# Testing PETSc

This is a minimum code to investigate the behaviour of PETSc when
user-defined matrices are used. It can be compiled using

`make`

and run using

`./petsc-test ...`

Options are `-d` to use "direct" access to the user-defined matrix,
rather than through the PETSc interface;`-k` to specify a KSP options
file; `-n` to specify the size of the system of equations to be
generated and solved. 

Running

`./petsc-test -k ksp-options -n 32`

generates and solves a random system of 32 equations, using the PETSc
GMRES solver. The right hand side is generated by multiplying a
reference solution by the system matrix, through the PETSc interface.

Calling with the `-d` option

`./petsc-test -k ksp-options -n 32 -d`

performs the same calculation, but with the right hand side generated
operating on the user data, rather than using the PETSc interface. 
