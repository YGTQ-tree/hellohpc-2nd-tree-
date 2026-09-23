#ifndef HASHCLASH_RUNNER_HPP
#define HASHCLASH_RUNNER_HPP

namespace hashclash {

// Entry point behind `./run tasks.txt`.
//
// Reads the task list, generates one colliding pair of files per line using a
// pool of worker threads, and returns 0 when every line succeeded.
int run_task_list(int argc, char** argv);

}

#endif // HASHCLASH_RUNNER_HPP
