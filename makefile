PROJECT_NAME = runner

CC = gcc
CFLAGS = -g#-Wall -pedantic -std=c99 -O3 
MATH_FLAGS = -lm
OBJECTS = Matrix.o NeuralNetwork.o main.o

all: $(PROJECT_NAME)

$(PROJECT_NAME): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $(PROJECT_NAME) $(MATH_FLAGS)
	rm -f *.o

Main.o: main.c