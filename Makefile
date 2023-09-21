CC=hipcc

libs=-I ./include
flags=-Wall -O2
executable=acopar_gpu.out
source=src/main.cpp \
	   src/acopar.cpp \
	   src/matrix.cpp \
	   src/parse_csv.cpp

all:
	$(CC) $(libs) $(flags) $(source) -o $(executable)

clean:
	$(RM) $(executable)
