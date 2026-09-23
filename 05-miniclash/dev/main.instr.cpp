// Development-only instrumented driver: measures block0/block1 cost split and
// path frequencies.  NOT part of the submitted source.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sys/time.h>
#include "main.hpp"

extern uint32 seed32_1, seed32_2;
extern void find_block0(uint32 block[], const uint32 IV[]);
extern void find_block1(uint32 block[], const uint32 IV[]);
extern void md5_compress(uint32 ihv[], const uint32 block[]);

static double now() { struct timeval tv; gettimeofday(&tv,0); return tv.tv_sec + 1e-6*tv.tv_usec; }

int main(int argc, char** argv)
{
    int reps = (argc > 1) ? atoi(argv[1]) : 5;
    uint32 seed = (argc > 2) ? (uint32)strtoul(argv[2],0,10) : (uint32)time(0);
    // OJ says the seed may change; for dev we just need a fixed random IV stream.
    int fixed = (argc > 3) ? atoi(argv[3]) : 1;

    double t0all = now();
    double tb0 = 0, tb1 = 0;
    for (int r = 0; r < reps; ++r)
    {
        uint32 IV[4];
        if (fixed) { IV[0]=0x67452301; IV[1]=0xefcdab89; IV[2]=0x98badcfe; IV[3]=0x10325476; }
        else {
            seed32_1 = seed + r*2654435761u; seed32_2 = seed*2246822519u + r*3266489917u;
            if(!seed32_1) seed32_1=1; if(!seed32_2) seed32_2=1;
            IV[0]=xrng64(); IV[1]=xrng64(); IV[2]=xrng64(); IV[3]=xrng64();
        }
        seed32_1 = seed + r*2654435761u; seed32_2 = seed*2246822519u + r*3266489917u;
        if(!seed32_1) seed32_1=1; if(!seed32_2) seed32_2=1;

        uint32 m1b0[16], m1b1[16];
        double a = now();
        find_block0(m1b0, IV);
        double b = now(); tb0 += b-a;
        uint32 IHV[4] = { IV[0], IV[1], IV[2], IV[3] };
        md5_compress(IHV, m1b0);
        find_block1(m1b1, IHV);
        double c = now(); tb1 += c-b;
        fprintf(stderr, "rep %d: block0 %.4f s  block1 %.4f s\n", r, b-a, c-b);
    }
    double tall = now() - t0all;
    fprintf(stderr, "== reps=%d total=%.4f s  block0=%.4f s (%.1f%%)  block1=%.4f s (%.1f%%)  mean/collision=%.4f s\n",
            reps, tall, tb0, 100*tb0/tall, tb1, 100*tb1/tall, tall/reps);
    return 0;
}
