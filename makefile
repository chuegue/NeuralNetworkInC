PROJECT_NAME = runner

CC = gcc
CFLAGS = -Wall -pedantic -std=c99 -O3 
OBJECTS = Matrix.o NeuralNetwork.o main.o

all: $(PROJECT_NAME)

$(PROJECT_NAME): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $(PROJECT_NAME)
	rm -f *.o

Main.o: main.c