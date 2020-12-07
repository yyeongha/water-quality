import numpy as np
from miss_data import MissData


def main():
    block = np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [np.nan, np.nan, 4, 4, 4],
        [np.nan, np.nan, 5, 5, 5],
        [6, 6, 6, 6, 6],
        [7, 7, 7, 7, 7],
        [8, 8, 8, 8, 8],
        [9, 9, np.nan, np.nan, np.nan],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5],
        [6, 6, 6, 6, 6],
        [7, 7, 7, 7, 7],
        [8, 8, 8, 8, 8],
        [np.nan, np.nan, np.nan, 9, np.nan],
    ])
    

    miss_data = MissData(load_dir='save')
    miss_data.save(block,100)

    # create miss data
    org_data = np.zeros(shape=(100,5))
    data_x = miss_data.make_missdata(org_data, missrate=0.2)
    print(data_x)

if __name__ == "__main__":
    main()