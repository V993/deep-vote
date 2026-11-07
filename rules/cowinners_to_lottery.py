import numpy as np

def to_lottery(cowinners, cands)->np.ndarray:
    return np.asarray([1.0/len(cowinners) if c in cowinners else 0.0 for c in cands])

if __name__ == "__main__":
    print(to_lottery([1,2], [1,2,3,4]))
    print(to_lottery({1,2}, (1,2,3,4)))