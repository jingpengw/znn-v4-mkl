ODIR		=	./bin
SFL			=	test_sparse.cpp
CPP		 	=	/opt/intel/bin/icpc
CPP_FLAGS	= 	-g
INC_FLAGS	=	-I../../../.. -I.. -I/opt/intel/composer_xe_2015.3.187/mkl/include
LIB_FLAGS	=	-L/opt/intel/mkl/lib/intel64
OPT_FLAGS	=	-DNDEBUG -O3 -std=c++1y
OTH_FLAGS	=	-Wall -Wextra -Wno-unused-result -Wno-unused-local-typedefs
LIBS		=	-lfftw3 -lfftw3f -lpthread -pthread -mkl=sequential

test: $(SFL)
	$(CPP) -o $(ODIR)/test $(SFL) $(CPP_FLAGS) $(INC_FLAGS) $(LIB_FLAGS) $(OPT_FLAGS) $(OTH_FLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*
