#!/usr/bin/env python3
"""Reference model of the fast algorithm: RNS decomposition + negacyclic NTT + CRT.

This is the blueprint the C++/CUDA code follows step for step. Nothing in it is
fast on its own; it exists to pin down the parameters and prove the math exact.

    c = a*b mod (X^N + 1, q),  q ~ 2^868

1. Pick K word-size primes p_j = 1 (mod 2N), p_j < 2^31, whose product P
   exceeds 2*N*q^2 with a safety margin. The exact integer negacyclic product
   has coefficients in (-N q^2, N q^2), so it is determined by its residues
   mod P when read in the centred range.
2. Decompose: a_i mod p_j for every coefficient and prime (sum of limbs times
   2^(32l) mod p_j).
3. Per prime: forward negacyclic NTT of a and b, pointwise multiply, inverse NTT.
4. CRT per coefficient: y_j = x_j * (P/p_j)^-1 mod p_j,
   v = round(sum_j y_j / p_j)           (a float estimate, exact thanks to the margin)
   c mod q = sum_j y_j * (P/p_j mod q) - v * (P mod q)   (mod q)

    python3 rns_ref.py solve <point dir>     answer every case like ./solution would
    python3 rns_ref.py params <N> <W>        show the prime set
"""
import json
import os
import sys

import numpy as np

import oracle

MARGIN_BITS = 20   # |c/P| < 2^-MARGIN_BITS keeps the rounding of v exact
PRIME_BITS = 31    # p < 2^31: products fit in 62 bits, so u64 on CPU and 32-bit Montgomery on GPU


def is_prime(n):
    return oracle.is_prime(n)


def choose_primes(N, W):
    """Largest primes below 2^PRIME_BITS with p == 1 (mod 2^16), enough for the bound.

    Using 2^16 rather than 2N makes one prime set serve every N <= 2^15.
    """
    step = 1 << 16
    assert 2 * N <= step
    need_bits = 2 * W + (N.bit_length() - 1) + 1 + MARGIN_BITS
    primes, P = [], 1
    p = ((1 << PRIME_BITS) - 1) // step * step + 1
    while P.bit_length() <= need_bits:
        if p < (1 << PRIME_BITS) and is_prime(p):
            primes.append(p)
            P *= p
        p -= step
    return primes


def primitive_root_2n(p, N):
    """psi with psi^N = -1 mod p (a primitive 2N-th root of unity)."""
    for g in range(2, p):
        psi = pow(g, (p - 1) // (2 * N), p)
        if pow(psi, N, p) == p - 1:
            return psi
    raise RuntimeError("no root")


def bitrev(x, bits):
    return int(bin(x)[2:].zfill(bits)[::-1], 2) if bits else 0


class Tables:
    """Per-prime twiddles in bit-reversed order, the layout the kernels use."""

    def __init__(self, N, primes):
        self.N, self.logN = N, N.bit_length() - 1
        K = len(primes)
        self.p = np.array(primes, dtype=np.uint64)[:, None]
        self.psi = np.zeros((K, N), dtype=np.uint64)       # psi^bitrev(i)
        self.psi_inv = np.zeros((K, N), dtype=np.uint64)   # psi^-bitrev(i)
        self.n_inv = np.zeros((K, 1), dtype=np.uint64)
        for j, p in enumerate(primes):
            psi = primitive_root_2n(p, N)
            ipsi = pow(psi, p - 2, p)
            pw = [pow(psi, bitrev(i, self.logN), p) for i in range(N)]
            ipw = [pow(ipsi, bitrev(i, self.logN), p) for i in range(N)]
            self.psi[j] = pw
            self.psi_inv[j] = ipw
            self.n_inv[j] = pow(N, p - 2, p)


def ntt_forward(x, T):
    """Negacyclic NTT, Cooley-Tukey, natural order in -> bit-reversed out (all primes at once)."""
    x = x.copy()
    p, N = T.p, T.N
    t, m = N, 1
    while m < N:
        t //= 2
        v = x.reshape(x.shape[0], m, 2, t)
        w = T.psi[:, m:2 * m][:, :, None]
        U = v[:, :, 0, :]
        V = v[:, :, 1, :] * w % p[:, :, None]
        hi = (U + V) % p[:, :, None]
        lo = (U + p[:, :, None] - V) % p[:, :, None]
        v[:, :, 0, :] = hi
        v[:, :, 1, :] = lo
        m *= 2
    return x


def ntt_inverse(x, T):
    """Inverse negacyclic NTT, Gentleman-Sande, bit-reversed in -> natural order out."""
    x = x.copy()
    p, N = T.p, T.N
    t, m = 1, N
    while m > 1:
        h = m // 2
        v = x.reshape(x.shape[0], h, 2, t)
        w = T.psi_inv[:, h:m][:, :, None]
        U = v[:, :, 0, :].copy()
        V = v[:, :, 1, :].copy()
        pp = p[:, :, None]
        v[:, :, 0, :] = (U + V) % pp
        v[:, :, 1, :] = (U + pp - V) % pp * w % pp
        t *= 2
        m = h
    return x * T.n_inv % p


def decompose(limbs, primes):
    """limbs: (N, L) u32 array -> (K, N) residues."""
    N, L = limbs.shape
    K = len(primes)
    p = np.array(primes, dtype=np.uint64)[:, None]
    pow2 = np.array([[pow(2, 32 * l, pj) for l in range(L)] for pj in primes], dtype=np.uint64)
    acc = np.zeros((K, N), dtype=np.uint64)
    lim = limbs.astype(np.uint64)
    for l in range(L):
        # limb < 2^32, pow2 < 2^31: product < 2^63, then reduce before the next add
        acc = (acc + lim[None, :, l] % p * pow2[:, l:l + 1]) % p
    return acc


class CRT:
    def __init__(self, primes, q):
        self.primes = primes
        self.q = q
        P = 1
        for p in primes:
            P *= p
        self.P = P
        self.inv = [pow(P // p, -1, p) for p in primes]          # (P/p_j)^-1 mod p_j
        self.Mq = [(P // p) % q for p in primes]                # P/p_j mod q
        self.Pq = P % q

    def reconstruct(self, x):
        """x: (K, N) residues -> list of N values in [0, q)."""
        p = np.array(self.primes, dtype=np.uint64)[:, None]
        inv = np.array(self.inv, dtype=np.uint64)[:, None]
        y = x * inv % p
        # v = round(sum y_j / p_j): the only float step, exact given MARGIN_BITS
        v = np.rint((y.astype(np.float64) / p.astype(np.float64)).sum(axis=0)).astype(np.int64)
        out = []
        q, Mq, Pq = self.q, self.Mq, self.Pq
        yl = y.T.tolist()
        for k, row in enumerate(yl):
            s = sum(yj * m for yj, m in zip(row, Mq)) - int(v[k]) * Pq
            out.append(s % q)
        return out

    def check_margin(self, N):
        # worst case |c| < N q^2; need |c|/P below 2^-MARGIN_BITS
        return (N * self.q * self.q * (1 << MARGIN_BITS)) < self.P


def multiply(a_limbs, b_limbs, q, N, primes=None, T=None, crt=None):
    primes = primes or choose_primes(N, q.bit_length())
    T = T or Tables(N, primes)
    crt = crt or CRT(primes, q)
    A = ntt_forward(decompose(a_limbs, primes), T)
    B = ntt_forward(decompose(b_limbs, primes), T)
    C = ntt_inverse(A * B % T.p, T)
    return crt.reconstruct(C)


def solve(root):
    with open(os.path.join(root, "manifest.json")) as f:
        m = json.load(f)
    N, W, L, cases = m["N"], m["W"], m["L"], m["cases"]
    q = oracle.from_limbs(np.fromfile(os.path.join(root, "point", "q.bin"), dtype="<u4").tolist())
    primes = choose_primes(N, W)
    T, crt = Tables(N, primes), CRT(primes, q)
    assert crt.check_margin(N)
    for s in range(cases):
        case = os.path.join(root, "cases", "%06d" % s)
        a = np.fromfile(os.path.join(case, "a.bin"), dtype="<u4").reshape(N, L)
        b = np.fromfile(os.path.join(case, "b.bin"), dtype="<u4").reshape(N, L)
        c = multiply(a, b, q, N, primes, T, crt)
        out = os.path.join(root, "out", "%06d" % s)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "c.bin"), "wb") as f:
            f.write(oracle.pack(c, L))
    print("solved %d case(s) N=%d W=%d with K=%d primes" % (cases, N, W, len(primes)))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "solve":
        solve(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "params":
        N, W = int(sys.argv[2]), int(sys.argv[3])
        ps = choose_primes(N, W)
        print("K=%d primes, P has %d bits, need > %d" % (
            len(ps), sum(p.bit_length() for p in ps), 2 * W + N.bit_length() + MARGIN_BITS))
        print("largest %#x smallest %#x" % (ps[0], ps[-1]))
    else:
        print(__doc__)
        sys.exit(2)
