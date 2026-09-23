#include <iostream>
#include <fstream>
#include <time.h>

#include "main.hpp"
#include "timer.hpp"
#ifndef HASHCLASH_LEGACY_CLI
#include "runner.hpp"
#endif

using namespace std;

const uint32 MD5IV[] = { 0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476 };

unsigned load_block(istream& i, uint32 block[]);
void save_block(ostream& o, const uint32 block[]);
void find_collision(const uint32 IV[], uint32 msg1block0[], uint32 msg1block1[], uint32 msg2block0[], uint32 msg2block1[], bool verbose = false);

// ---------------------------------------------------------------------------
// The historical single-task command line interface of md5_fastcoll lives in
// legacy_cli.inc.  It is not compiled by default (that whole file is wrapped in
// #if defined(HASHCLASH_LEGACY_CLI)); `make legacy` builds it for manual
// experiments.  The graded entry point is the batch runner below, which is what
// `./run tasks.txt` executes.
// ---------------------------------------------------------------------------
#include "legacy_cli.inc"

// ---------------------------------------------------------------------------
// Batch runner: the grader runs `./run tasks.txt` and times this call.
// ---------------------------------------------------------------------------
#ifndef HASHCLASH_LEGACY_CLI
int main(int argc, char** argv)
{
	return hashclash::run_task_list(argc, argv);
}
#endif

// ---------------------------------------------------------------------------
// Produce one pair of colliding messages for the given initial value.
//
// Both returned messages are the concatenation of two 64 byte blocks; the
// caller (runner.cpp) prepends the user supplied prefix.  This routine is the
// original hashclash search, unchanged.
// ---------------------------------------------------------------------------
void find_collision(const uint32 IV[], uint32 msg1block0[], uint32 msg1block1[], uint32 msg2block0[], uint32 msg2block1[], bool verbose)
{
	if (verbose)
		cout << "Generating first block: " << flush;
	find_block0(msg1block0, IV);

	uint32 IHV[4] = { IV[0], IV[1], IV[2], IV[3] };
	md5_compress(IHV, msg1block0);

	if (verbose)
		cout << endl << "Generating second block: " << flush;
	find_block1(msg1block1, IHV);

	// The standard MD5 collision trick: flip the most significant bit of m4 and
	// m14, and add/subtract 2^15 in m11.  Those three differences cancel out
	// inside the compression function, so message 2 hashes to the same value as
	// message 1 while the two messages differ.
	for (int t = 0; t < 16; ++t)
	{
		msg2block0[t] = msg1block0[t];
		msg2block1[t] = msg1block1[t];
	}
	msg2block0[4] += 1 << 31; msg2block0[11] += 1 << 15; msg2block0[14] += 1 << 31;
	msg2block1[4] += 1 << 31; msg2block1[11] -= 1 << 15; msg2block1[14] += 1 << 31;
	if (verbose)
		cout << endl;
}
