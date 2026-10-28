# Executable
EXE    = tiSegmenter

# Compiler, Linker Defines
CC      = g++ 
CFLAGS  = -w -O3 -Wno-deprecated
#LDFLAGS = $(LIBS)

# Compile and Assemble C++ Source Files into Object Files
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@
# Source and Object files
SRC    = $(wildcard *.cpp)
OBJ    = $(patsubst %.cpp, %.o, $(SRC))
	
# Link all Object Files with external Libraries into Binaries
$(EXE): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $(EXE) -lz

.PHONY: clean
clean:
	 -rm -f core *.o

