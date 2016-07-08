all: nn nn-dbg

nn: main.cpp
	$(CXX) -std=c++11 -O3 $< -o $@

nn-dbg: main.cpp
	$(CXX) -std=c++11 -g -O0 $< -o $@

clean:
	rm -f nn
