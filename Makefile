CXX=hipcc
SOURCE=acopar3otim.cpp
BINARY=acopar3otim
FLAGS=-O2 -g -lrocrand

all:
	$(CXX) $(FLAGS) $(SOURCE) -o $(BINARY)

clean:
	$(RM) $(BINARY)
