CFLAGS=-Wall -Werror

all: nn nndbg nn2 nndbg2 nn3 nndbg3

nn: main.cpp
	$(CXX) $(CFLAGS) -std=c++1y -O3 $< -o $@

nndbg: main.cpp
	$(CXX) $(CFLAGS) -std=c++1y -g -O0 $< -o $@

nn2: main2.cpp
	$(CXX) $(CFLAGS) -std=c++1y -O3 $< -o $@

nndbg2: main2.cpp
	$(CXX) $(CFLAGS) -std=c++1y -g -O0 $< -o $@

nn3: main3.cpp
	$(CXX) $(CFLAGS) -std=c++1y -O3 $< -o $@

nndbg3: main3.cpp
	$(CXX) $(CFLAGS) -std=c++1y -g -O0 $< -o $@

clean:
	rm -f nn nndbg nn2 nndbg2 nn3 nndbg3
