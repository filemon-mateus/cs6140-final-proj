CXX:=g++
NVCC:=/usr/local/cuda/bin/nvcc
INCLUDE:=-I/usr/local/cuda/include
CXXFLAGS:=-std=c++17 -O3

kmeans1: kmeans-eig1.cc
	$(CXX) $(CXXFLAGS) $< -o $@ 

kmeans2: kmeans-gpu1.cu
	$(NVCC) $(CXXFLAGS) $(INCLUDE) $< -o $@ 

kmeans3: kmeans-gpu2.cu
	$(NVCC) $(CXXFLAGS) $(INCLUDE) $< -o $@ 

clean:
	$(RM) kmeans1 kmeans2 kmeans3

.PHONY: kmeans1 kmeans2 kmeans3 clean
