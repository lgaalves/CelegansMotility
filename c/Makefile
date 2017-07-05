# file: Makefile		G. Moody		20 July 2005

# 'make' description file for building and testing 'dfa'

BINDIR=/usr/bin

# 'make' or 'make dfa': compile 'dfa' executable from dfa.c
dfa:		dfa.c
	$(CC) -o dfa -O dfa.c -lm

# 'make install': install 'dfa' executable into BINDIR (see above)
install:	dfa
	cp dfa $(BINDIR)

# 'make check': test the executable by analyzing a known input and comparing
# the output with the expected output
check:		dfa
	./dfa <rr-intervals >dfa.out
	@-diff -q dfa.out rr-dfa && echo test passed

# 'make clean': remove generated and temporary files in this directory
clean:
	rm -f dfa dfa.out *~