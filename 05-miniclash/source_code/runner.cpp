// Batch driver for the Miniclash problem.
//
// The grader runs `./run tasks.txt`, where every line of tasks.txt is
//     <input_file> <output_file1> <output_file2>
// and the whole run is timed once, from process start to process exit.
//
// Everything here is plain C++11 with no dependencies beyond the standard
// library, so it builds with any compiler the grading container happens to
// provide.
//
// Design notes
// ------------
// * One colliding pair per task is found by the original hashclash search
//   (find_collision() in main.cpp).  The search is randomised, so every task
//   is an independent random experiment and tasks can be run in any order or
//   in parallel without changing what is produced.
// * The 256 task case is eight times larger than the 32 core machine, so the
//   total run time is dominated by how evenly the work is spread.  A single
//   shared task counter is therefore used: whenever a thread finishes a task
//   it grabs the next free index.  Fast tasks free their core for the next
//   task instead of leaving it idle until a slow neighbour finishes (which is
//   what the original one-process-per-task shell script did).
// * The number of threads is taken from the CPU affinity mask, because the
//   grader starts us under `taskset -c ...`.  nproc/hardware_concurrency()
//   would report every core of the machine instead of the cores we own.

#include "runner.hpp"

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <sched.h>
#include <time.h>
#include <unistd.h>

#include "main.hpp"

// The original program exposes these two globals as the state of its xorshift
// generator (see main.hpp).  Each worker thread owns them for the duration of
// one task, so two threads never search from the same state.
extern uint32 seed32_1;
extern uint32 seed32_2;

// Defined at global scope in main.cpp.
void find_collision(const uint32 IV[], uint32 msg1block0[], uint32 msg1block1[],
	uint32 msg2block0[], uint32 msg2block1[], bool verbose);

namespace hashclash {

namespace {


struct Task {
	std::string input;
	std::string output1;
	std::string output2;
};

// --------------------------------------------------------------------------
// Initial value of the search
//
// A collision found by hashclash is a pair of *whole* 64 byte blocks whose
// difference cancels out inside the compression function, so it can be
// appended to any message as long as both files reach those two blocks with
// the same chaining value.
//
// Both output files share the prefix, and the two generated block pairs have
// the same length, so MD5 appends exactly the same padding to both of them and
// the collision survives.  What matters is where that padding lands: the last
// generated block is only 64 bytes long, so MD5 always has to append the 0x80
// marker plus the 64 bit length, and those bytes are simply part of the last
// block when the verifier hashes the file.
//
// The chaining value the two generated blocks start from is therefore the MD5
// state after the prefix bytes on their own, with no padding of its own: a
// trailing partial block is zero filled and compressed as-is, because the
// padding that MD5 will eventually append belongs to the end of the whole file,
// not to the end of the prefix.
// --------------------------------------------------------------------------

// Reads up to 16 little endian words (the zero padded prefix).  Returns the
// number of bytes actually read.
unsigned load_block(std::istream& in, uint32 block[16])
{
	unsigned len = 0;
	char uc;
	for (unsigned k = 0; k < 16; ++k) {
		block[k] = 0;
		for (unsigned c = 0; c < 4; ++c) {
			in.get(uc);
			if (in)
				++len;
			else
				uc = 0;
			block[k] += uint32((unsigned char)uc) << (c * 8);
		}
	}
	return len;
}

// Chaining value that precedes the two generated blocks: MD5 state after the
// prefix bytes, with the final partial block zero filled and the padding that
// MD5 appends to the complete file left to the verifier.
void prefix_state(const std::vector<char>& data, uint32 state[4])
{
	state[0] = 0x67452301;
	state[1] = 0xefcdab89;
	state[2] = 0x98badcfe;
	state[3] = 0x10325476;

	for (size_t offset = 0; offset < data.size(); offset += 64) {
		uint32 block[16];
		for (unsigned k = 0; k < 16; ++k) {
			uint32 word = 0;
			for (unsigned c = 0; c < 4; ++c) {
				size_t index = offset + k * 4 + c;
				word += uint32(index < data.size() ? (unsigned char)data[index] : 0) << (c * 8);
			}
			block[k] = word;
		}
		md5_compress(state, block);
	}
}

void store_block(std::vector<char>& out, const uint32 block[16])
{
	for (unsigned k = 0; k < 16; ++k)
		for (unsigned c = 0; c < 4; ++c)
			out.push_back((char)((block[k] >> (c * 8)) & 0xFF));
}

bool write_file(const std::string& path, const std::vector<char>& data, std::string& error);

// Writes both colliding files: prefix followed by the two generated blocks.
bool write_pair(const Task& task, const std::vector<char>& prefix,
	const uint32 msg1block0[16], const uint32 msg1block1[16],
	const uint32 msg2block0[16], const uint32 msg2block1[16], std::string& error)
{
	std::vector<char> buffer1(prefix);
	std::vector<char> buffer2(prefix);
	store_block(buffer1, msg1block0);
	store_block(buffer1, msg1block1);
	store_block(buffer2, msg2block0);
	store_block(buffer2, msg2block1);

	if (!write_file(task.output1, buffer1, error))
		return false;
	return write_file(task.output2, buffer2, error);
}

bool write_file(const std::string& path, const std::vector<char>& data, std::string& error)
{
	std::ofstream out(path.c_str(), std::ios::binary | std::ios::trunc);
	if (!out) {
		error = "cannot open output file: " + path;
		return false;
	}
	if (!data.empty())
		out.write(&data[0], (std::streamsize)data.size());
	out.close();
	if (!out) {
		error = "cannot write output file: " + path;
		return false;
	}
	return true;
}

// --------------------------------------------------------------------------
// Task list parsing
// --------------------------------------------------------------------------

bool parse_tasks(const char* path, std::vector<Task>& tasks, std::string& error)
{
	std::ifstream in(path);
	if (!in) {
		error = std::string("cannot read task file: ") + path;
		return false;
	}
	std::string line;
	unsigned lineno = 0;
	while (std::getline(in, line)) {
		++lineno;
		std::istringstream iss(line);
		Task task;
		if (!(iss >> task.input >> task.output1 >> task.output2))
			continue; // blank or whitespace-only line
		std::string extra;
		if (iss >> extra) {
			std::ostringstream oss;
			oss << "task file line " << lineno << " has more than three fields";
			error = oss.str();
			return false;
		}
		tasks.push_back(task);
	}
	return true;
}

// --------------------------------------------------------------------------
// Worker pool
// --------------------------------------------------------------------------

// Deterministic 64 bit mixer (splitmix64).  Used to derive independent xorshift
// states for the searches from one per-thread counter.
uint64_t mix64(uint64_t x)
{
	x += 0x9E3779B97F4A7C15ull;
	x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
	x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
	return x ^ (x >> 31);
}

struct Shared {
	const std::vector<Task>* tasks;
	std::atomic<std::size_t> next;
	std::atomic<std::size_t> failures;
	std::atomic<bool> stop;
	std::mutex error_mutex;
	std::string first_error;

	Shared() : tasks(0), next(0), failures(0), stop(false) {}
};

void worker(Shared* shared, uint64_t stream_id)
{
	// Every thread walks its own stream of xorshift states.  Two threads may
	// start from the same second, so the stream id is folded into both words to
	// guarantee different starting points (a zero state would make the
	// generator degenerate, hence the forced non-zero values).
	uint32 s1 = (uint32)mix64(stream_id * 2 + 1);
	uint32 s2 = (uint32)mix64(stream_id * 2 + 2);
	if (s1 == 0)
		s1 = 0x9E3779B9u;
	if (s2 == 0)
		s2 = 0x85EBCA6Bu;
	s1 ^= (uint32)time(0);

	const std::vector<Task>& tasks = *shared->tasks;

	for (;;) {
		size_t index = shared->next.fetch_add(1, std::memory_order_relaxed);
		if (index >= tasks.size() || shared->stop.load(std::memory_order_relaxed))
			return;

		seed32_1 = s1;
		seed32_2 = s2;

		const Task& task = tasks[index];

		// Chaining value that precedes the two generated blocks in the output
		// files: MD5 state after the prefix, padding included.
		std::vector<char> prefix;
		{
			std::ifstream in(task.input.c_str(), std::ios::binary);
			if (!in) {
				std::lock_guard<std::mutex> lock(shared->error_mutex);
				if (shared->first_error.empty())
					shared->first_error = "cannot open input file: " + task.input;
				shared->failures.fetch_add(1, std::memory_order_relaxed);
				continue;
			}
			in.seekg(0, std::ios::end);
			std::streamoff size = in.tellg();
			in.seekg(0, std::ios::beg);
			if (size > 0) {
				prefix.resize((size_t)size);
				in.read(&prefix[0], size);
				if (in.gcount() != size) {
					std::lock_guard<std::mutex> lock(shared->error_mutex);
					if (shared->first_error.empty())
						shared->first_error = "short read on input file: " + task.input;
					shared->failures.fetch_add(1, std::memory_order_relaxed);
					continue;
				}
			}
		}

		uint32 IV[4];
		prefix_state(prefix, IV);

		uint32 msg1block0[16], msg1block1[16], msg2block0[16], msg2block1[16];
		find_collision(IV, msg1block0, msg1block1, msg2block0, msg2block1, false);

		// Advance this thread's generator so the next task starts somewhere new.
		seed32_1 = xrng64();
		seed32_2 = xrng64();
		s1 = seed32_1;
		s2 = seed32_2;

		std::string error;
		if (!write_pair(task, prefix, msg1block0, msg1block1, msg2block0, msg2block1, error)) {
			std::lock_guard<std::mutex> lock(shared->error_mutex);
			if (shared->first_error.empty())
				shared->first_error = error;
			shared->failures.fetch_add(1, std::memory_order_relaxed);
		}
	}
}

// Number of CPUs this process is actually allowed to run on.  The grader
// restricts us with taskset, so the affinity mask is the authoritative answer;
// hardware_concurrency() would report the whole machine.
unsigned detect_threads()
{
	const char* env = std::getenv("HASHCLASH_THREADS");
	if (env && *env) {
		long value = std::strtol(env, 0, 10);
		if (value > 0)
			return (unsigned)value;
	}

#ifdef CPU_COUNT
	cpu_set_t set;
	if (sched_getaffinity(0, sizeof(set), &set) == 0) {
		int count = CPU_COUNT(&set);
		if (count > 0)
			return (unsigned)count;
	}
#endif

	unsigned hw = std::thread::hardware_concurrency();
	return hw ? hw : 1;
}

} // namespace

int run_task_list(int argc, char** argv)
{
	if (argc != 2) {
		std::cerr << "usage: " << (argc > 0 ? argv[0] : "run") << " <tasks.txt>" << std::endl;
		return 2;
	}

	std::vector<Task> tasks;
	std::string error;
	if (!parse_tasks(argv[1], tasks, error)) {
		std::cerr << "run: " << error << std::endl;
		return 1;
	}
	if (tasks.empty())
		return 0;

	unsigned threads = detect_threads();
	if (threads > tasks.size())
		threads = (unsigned)tasks.size();

	Shared shared;
	shared.tasks = &tasks;

	// Progress note on stderr only: the grader checks the exit status and the
	// produced files, and keeping stdout unused means nothing can interfere
	// with any output the grader may collect.
	std::cerr << "run: " << tasks.size() << " task(s), " << threads << " thread(s)" << std::endl;

	std::vector<std::thread> pool;
	pool.reserve(threads - 1);
	for (unsigned t = 1; t < threads; ++t)
		pool.push_back(std::thread(worker, &shared, (uint64_t)t));
	worker(&shared, 0);
	for (size_t t = 0; t < pool.size(); ++t)
		pool[t].join();

	if (shared.failures.load() != 0) {
		std::cerr << "run: " << shared.failures.load() << " of " << tasks.size()
			<< " task(s) failed";
		if (!shared.first_error.empty())
			std::cerr << ": " << shared.first_error;
		std::cerr << std::endl;
		return 1;
	}
	return 0;
}

} // namespace hashclash
