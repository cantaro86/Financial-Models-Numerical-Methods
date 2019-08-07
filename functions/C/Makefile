PROGRAM_NAME=BS_sor
OBJECTS=BS_SOR_main.o SOR.o PDE_solver.o       

CXX=gcc #clang


CXXFLAGS +=-Wall -Werror
LIBS=-lm                  




$(PROGRAM_NAME):$(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(PROGRAM_NAME) $(OBJECTS) $(LIBS)

	@echo " "
	@echo "Compilation completed!"
	@echo " "



BS_SOR_main.o:SOR.h PDE_solver.h BS_SOR_main.c  
	$(CXX) $(CXXFLAGS) -c -o BS_SOR_main.o BS_SOR_main.c


SOR.o:SOR.h SOR.c 
	$(CXX) $(CXXFLAGS) -c -o SOR.o SOR.c


PDE_solver.o:SOR.h SOR.c PDE_solver.h PDE_solver.c 
	$(CXX) $(CXXFLAGS) -c -o PDE_solver.o PDE_solver.c



clean:
	rm -f *.o
	rm -f *~
	rm -f core


cat: 
	cat Makefile


all:	$(PROGRAM_NAME) clean
