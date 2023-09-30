CC=hipcc

LIBS=-I./include
CFLAGS=-Wall -O2
EXECUTABLE=acopar_gpu.out
SOURCES=$(wildcard src/*.cpp)
OBJECTS=$(patsubst %.cpp, %.o, $(SOURCES))

all: $(OBJECTS)
	$(CC) $(OBJECTS) -o $(EXECUTABLE)

$(OBJECTS): src/%.o : src/%.cpp
	$(CC) $(LIBS) $(CFLAGS) -c $< $(LIBS) -o $@

clean:
	$(RM) *.out src/*.o

debug:
	$(CC) $(LIBS) $(CFLAGS) -g $(SOURCES) -o $(EXECUTABLE)
