"""Correctness fuzz for fherma-polymul: many (N, W) points, adversarial inputs,
every copy mode and twiddle path, GPU and CPU builds.

    python3 fuzz.py <work dir> [random points] [seed]

Expected answers come from make_big.negacyclic_kronecker (GMP), cross-checked
against the exact O(N^2) product for N <= 256. Prints one line per (point,
config) and a summary; exits 1 on any failure.
"""
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time

REPO = os.path.expanduser("~/fherma-polymul")
sys.path.insert(0, REPO)
import make_big  # noqa: E402
import oracle    # noqa: E402

GPU = os.path.join(REPO, "build/solution")
CPU = os.path.join(REPO, "solution_cpu")

CONFIGS = [
    ("default", {}),
    ("staged", {"FHERMA_IN": "staged", "FHERMA_OUT": "staged"}),
    ("hmm", {"FHERMA_IN": "hmm", "FHERMA_OUT": "hmm"}),
    ("stream", {"FHERMA_IN": "stream", "FHERMA_OUT": "stream"}),
    ("mapped", {"FHERMA_IN": "mapped", "FHERMA_OUT": "mapped"}),
    ("register", {"FHERMA_IN": "register", "FHERMA_OUT": "register"}),
    ("tw-gen", {"FHERMA_TWIDDLES": "gen"}),
    ("tw-table", {"FHERMA_TWIDDLES": "table"}),
]


def schoolbook(a, b, q, N):
    c = [0] * N
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            k = i + j
            if k < N:
                c[k] += x * y
            else:
                c[k - N] -= x * y
    return [v % q for v in c]


def patterns(N, q, rng, count):
    """Adversarial operand pairs, then random ones."""
    out = [
        ([q - 1] * N, [q - 1] * N),                                  # CRT worst case
        ([rng.randrange(q) for _ in range(N)], [rng.randrange(q) for _ in range(N)]),
        ([0] * N, [rng.randrange(q) for _ in range(N)]),             # zero
        ([0] * (N - 1) + [1], [0] * (N - 1) + [q - 1]),              # wrap: x^(N-1) * x^(N-1)
        ([rng.randrange(q) for _ in range(N)], [1] + [0] * (N - 1)),  # identity
        ([(q - 1) * (i % 2) for i in range(N)], [(q - 1) * ((i + 1) % 2) for i in range(N)]),
    ]
    return out[:count]


def make_point(root, N, W, cases, rng):
    q = oracle.negacyclic_prime(N, W)
    L = (W + 31) // 32
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(os.path.join(root, "point"))
    with open(os.path.join(root, "point", "q.bin"), "wb") as f:
        f.write(oracle.pack([q], L))
    pairs = patterns(N, q, rng, cases)
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump({"N": N, "W": W, "L": L, "cases": len(pairs)}, f)
    for s, (a, b) in enumerate(pairs):
        c = make_big.negacyclic_kronecker(a, b, q, N)
        if N <= 256 and c != schoolbook(a, b, q, N):
            raise RuntimeError("reference disagrees with schoolbook at N=%d W=%d case %d" % (N, W, s))
        for sub, name, v in (("cases", "a", a), ("cases", "b", b), ("expected", "c", c)):
            d = os.path.join(root, sub, "%06d" % s)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, name + ".bin"), "wb") as f:
                f.write(oracle.pack(v, L))
    return L, len(pairs)


def check(root, ncases):
    """'ok', or why not: compares out/ against expected/ byte for byte."""
    for s in range(ncases):
        exp = os.path.join(root, "expected", "%06d" % s, "c.bin")
        got = os.path.join(root, "out", "%06d" % s, "c.bin")
        if not os.path.exists(got):
            return "case %d: no output" % s
        if open(got, "rb").read() != open(exp, "rb").read():
            return "case %d: WRONG" % s
    return "ok"


def run(binary, root, env_extra, ncases, timeout):
    shutil.rmtree(os.path.join(root, "out"), ignore_errors=True)
    env = dict(os.environ, **env_extra)
    try:
        p = subprocess.run([binary, root], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    if p.returncode != 0:
        return "exit %d: %s" % (p.returncode, p.stderr.decode(errors="replace").strip().splitlines()[-1:])
    return check(root, ncases)


def main():
    work = sys.argv[1]
    nrand = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 12345)
    os.makedirs(work, exist_ok=True)

    points = [(2, 3), (2, 32), (2, 33), (4, 64), (4, 65), (8, 96), (8, 97), (16, 127), (32, 31),
              (64, 32), (1024, 512), (8192, 256), (32768, 868), (65536, 1747),
              (256, 63), (256, 64), (256, 65), (512, 255), (512, 256), (512, 257),
              (128, 1023), (128, 1024), (128, 1025), (64, 1999), (64, 2047), (64, 2048), (4096, 2048),
              (1 << 17, 64), (1 << 18, 128), (1 << 20, 31), (1 << 22, 32)]
    for _ in range(nrand):
        k = rng.randint(1, 16)
        N = 1 << k
        wmin = math.ceil(math.log2(2 * N + 2)) + 1
        W = rng.randint(wmin, 2048)
        points.append((N, W))

    failures = 0
    t0 = time.time()
    for N, W in points:
        root = os.path.join(work, "p")
        cases = 6 if N <= (1 << 16) else 2
        L, ncases = make_point(root, N, W, cases, rng)
        timeout = 120 if N <= (1 << 16) else 600
        results = []
        for name, env in CONFIGS:
            r = run(GPU, root, env, ncases, timeout)
            results.append((name, r))
        if N <= (1 << 16):
            results.append(("cpu", run(CPU, root, {}, ncases, 1200)))
        bad = [(n, r) for n, r in results if r != "ok"]
        failures += len(bad)
        print("N=%-8d W=%-5d L=%-3d cases=%d  %s" % (N, W, L, ncases,
              "all %d configs ok" % len(results) if not bad else "FAIL " + str(bad)), flush=True)

    # Out of range: must be refused cleanly (non-zero exit, no output), never wrong.
    for N, W in ((1 << 23, 32), (64, 2049)):
        root = os.path.join(work, "p")
        try:
            make_point(root, N, W, 1, rng)
        except Exception as e:  # generator itself may be too slow / unable
            print("N=%d W=%d: generator: %s" % (N, W, e))
            continue
        r = run(GPU, root, {}, 1, 600)
        ok = r.startswith("exit") and "WRONG" not in r
        failures += 0 if ok else 1
        print("N=%-8d W=%-5d out of range -> %s  [%s]" % (N, W, r, "refused cleanly" if ok else "PROBLEM"), flush=True)

    print("SUMMARY: %d points, %d failures, %.0f s" % (len(points), failures, time.time() - t0))
    sys.exit(1 if failures else 0)


main()
