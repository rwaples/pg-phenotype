"""Random test pedigrees with the structures prep must get right."""

from __future__ import annotations

import numpy as np


def random_pedigree(seed: int, *, n_founders: int = 12, generations: int = 4, size: int = 14) -> dict[str, np.ndarray]:
    """A small inbred pedigree with MZ twins, missing and external parents.

    Rows are shuffled so no order is assumed; ids are sparse and unsorted.
    """
    rng = np.random.default_rng(seed)
    mother, father, twin, sex = [], [], [], []
    for _ in range(n_founders):
        mother.append(-1)
        father.append(-1)
        twin.append(-1)
        sex.append(int(rng.integers(0, 2)))
    previous = list(range(n_founders))
    for _ in range(generations):
        females = [r for r in previous if sex[r] == 0] or previous[:1]
        males = [r for r in previous if sex[r] == 1] or previous[-1:]
        current = []
        while len(current) < size:
            m, f = int(rng.choice(females)), int(rng.choice(males))
            if m == f:
                continue
            kids = int(rng.integers(1, 4))
            for k in range(kids):
                r = len(mother)
                mother.append(m)
                father.append(f)
                twin.append(-1)
                sex.append(int(rng.integers(0, 2)))
                current.append(r)
                if k == 1 and rng.random() < 0.3:
                    twin[r], twin[r - 1] = r - 1, r
                    sex[r] = sex[r - 1]
        previous = current
    n = len(mother)
    ids = rng.permutation(np.arange(1000, 1000 + 7 * n, 7))
    mother_id = np.array([ids[m] if m >= 0 else -1 for m in mother])
    father_id = np.array([ids[f] if f >= 0 else -1 for f in father])
    twin_id = np.array([ids[t] if t >= 0 else -1 for t in twin])
    lost = rng.random(n) < 0.08
    mother_id[lost & (twin_id < 0)] = -1
    external = rng.random(n) < 0.05
    father_id[external & (twin_id < 0)] = 99_000 + np.arange(n)[external & (twin_id < 0)]
    order = rng.permutation(n)
    return {
        "id": ids[order],
        "mother": mother_id[order],
        "father": father_id[order],
        "twin": twin_id[order],
        "sex": np.array(sex)[order],
    }
