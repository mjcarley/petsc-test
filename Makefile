CC = gcc -g -fopenmp -Wall
CFLAGS = `pkg-config --cflags petsc`
LIBS = `pkg-config --libs petsc`

petsc-test: petsc-test.o
	$(CC) $(CFLAGS) petsc-test.o $(LIBS) -o petsc-test

.c.o:
	$(CC) -c $(CFLAGS) $<

clean:
	rm -f *.c~ *.o *.h~
