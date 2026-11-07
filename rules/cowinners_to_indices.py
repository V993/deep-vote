import numpy as np

def to_indices(cowinners, cands)->np.ndarray:
    return np.array([1 if cand in cowinners else 0 for cand in sorted(cands)])

if __name__ == "__main__":
    print(to_indices([1,2], [1,2,3,4]))
    print(to_indices([1,5], [1,2,3,4]))