CXX=hipcc
SOURCE=acopar3otim.cpp
BINARY=acopar3otim
FLAGS=-O2 -lrocrand -Wall -pedantic

all:
	$(CXX) $(FLAGS) $(SOURCE) -o $(BINARY)

debug:
	$(CXX) -g $(SOURCE) -o $(BINARY)

clean:
	$(RM) $(BINARY)
