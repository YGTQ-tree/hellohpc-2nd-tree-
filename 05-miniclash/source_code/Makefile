CXX := g++
CPPFLAGS := -O3 -march=native -funroll-loops
OBJS := block0.o block1.o block1stevens00.o block1stevens01.o block1stevens10.o block1stevens11.o block1wang.o main.o runner.o md5.o timer.o
LIBS :=

.PHONY := clean all legacy

all: run

# `run` is the graded entry point: the grader executes `./run tasks.txt`.
run: $(OBJS)
	$(CXX) $(CPPFLAGS) $(LIBS) $^ -o $@

# `md5fastcoll` is the original single-file tool, kept for manual experiments.
legacy: $(OBJS:main.o=main_legacy.o)
	$(CXX) $(CPPFLAGS) $(LIBS) $^ -o $@

main_legacy.o: main.cpp legacy_cli.inc
	$(CXX) $(CPPFLAGS) -DHASHCLASH_LEGACY_CLI -c $< -o $@

clean:
	rm -f $(OBJS) main_legacy.o
	rm -f md5fastcoll
	rm -f run

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@
