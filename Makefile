CXX      = g++
LDFLAGS  = 
CXXFLAGS = -w -O3 -std=c++11 -funroll-loops

OBJECTS  = KNRM.o KNRM_main.o

TARGET   = KNRM_main

all: $(TARGET)

KNRM_main: $(OBJECTS)
	$(CXX) -o $@ $(OBJECTS) $(LDFLAGS)

KNRM.o: KNRM.cpp KNRM.h
	$(CXX) $(CXXFLAGS) -c KNRM.cpp

KNRM_main.o: KNRM_main.cpp
	$(CXX) $(CXXFLAGS) -c KNRM_main.cpp

clean:
	rm  *.o $(TARGET)
