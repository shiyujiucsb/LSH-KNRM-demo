CXX      = g++
LDFLAGS  = 
CXXFLAGS = -w -O3 -std=c++11 -funroll-loops

OBJECTS  = LSH_KNRM.o LSH_KNRM_main.o

TARGET   = LSH_KNRM_main

all: $(TARGET)

LSH_KNRM_main: $(OBJECTS)
	$(CXX) -o $@ $(OBJECTS) $(LDFLAGS)

LSH_KNRM.o: LSH_KNRM.cpp LSH_KNRM.h
	$(CXX) $(CXXFLAGS) -c LSH_KNRM.cpp

LSH_KNRM_main.o: LSH_KNRM_main.cpp
	$(CXX) $(CXXFLAGS) -c LSH_KNRM_main.cpp

clean:
	rm  *.o $(TARGET)
