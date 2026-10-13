#!/usr/bin/env python3
"""oracle.py's `make`, but with the expected answer computed fast enough for N=32768.

The expected product comes from Kronecker substitution with GMP: pack each
polynomial into one huge integer with a slot wide enough for any coefficient of
the plain product, multiply once, unpack, fold the top half back negated. It
shares no code with the RNS/NTT path, so agreement is meaningful.

    python3 make_big.py <dir> <N> <W> [seeds]      then: python3 oracle.py verify <dir>
"""
import json
import os
import random
import sys

import gmpy2

import oracle


def negacyclic_kronecker(a, b, q, N):
    # A slot wide enough for any coefficient of the plain product, rounded up
    # to whole bytes so packing and unpacking are byte slicing (linear in N).
    slot = (2 * q.bit_length() + N.bit_length() + 1 + 7) // 8
    A = gmpy2.mpz(int.from_bytes(b"".join(x.to_bytes(slot, "little") for x in a), "little"))
    B = gmpy2.mpz(int.from_bytes(b"".join(x.to_bytes(slot, "little") for x in b), "little"))
    raw = int(A * B).to_bytes(2 * N * slot, "little")
    full = [int.from_bytes(raw[k * slot:(k + 1) * slot], "little") for k in range(2 * N)]
    return [(full[k] - full[k + N]) % q for k in range(N)]


def make(root, N, W, seeds):
    L = oracle.ceil_div(W, 32)
    q = oracle.negacyclic_prime(N, W)
    os.makedirs(os.path.join(root, "point"), exist_ok=True)
    with open(os.path.join(root, "point", "q.bin"), "wb") as f:
        f.write(oracle.pack([q], L))
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump({"N": N, "W": W, "L": L, "cases": seeds}, f)
    for s in range(seeds):
        rng = random.Random(1000 + s)
        a = [rng.randrange(0, q) for _ in range(N)]
        b = [rng.randrange(0, q) for _ in range(N)]
        if s == 0:
            a = [q - 1] * N   # worst case for the CRT margin: every product maximal
            b = [q - 1] * N
        case = os.path.join(root, "cases", "%06d" % s)
        exp = os.path.join(root, "expected", "%06d" % s)
        os.makedirs(case, exist_ok=True)
        os.makedirs(exp, exist_ok=True)
        with open(os.path.join(case, "a.bin"), "wb") as f:
            f.write(oracle.pack(a, L))
        with open(os.path.join(case, "b.bin"), "wb") as f:
            f.write(oracle.pack(b, L))
        with open(os.path.join(exp, "c.bin"), "wb") as f:
            f.write(oracle.pack(negacyclic_kronecker(a, b, q, N), L))
    print("made N=%d W=%d L=%d cases=%d at %s" % (N, W, L, seeds, root))


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    make(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 1)
