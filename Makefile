CFLAGS=-Wall -Werror

all: nn nndbg nn2 nn2dbg

nn: main.cpp
	$(CXX) $(CFLAGS) -std=c++11 -O3 $< -o $@

nndbg: main.cpp
	$(CXX) $(CFLAGS) -std=c++11 -g -O0 $< -o $@

nn2: main2.cpp
	$(CXX) $(CFLAGS) -std=c++11 -O3 $< -o $@

nn2dbg: main2.cpp
	$(CXX) $(CFLAGS) -std=c++11 -g -O0 $< -o $@

clean:
	rm -f nn nndbg nn2 nn2dbg
