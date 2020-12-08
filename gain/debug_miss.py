import numpy as np
from miss_data import MissData
import pandas as pd

def main():
    # block = np.array([
    #     [0, 0, 0, 0, 0],
    #     [1, 1, 1, 1, 1],
    #     [2, 2, 2, 2, 2],
    #     [3, 3, 3, 3, 3],
    #     [np.nan, np.nan, 4, 4, 4],
    #     [np.nan, np.nan, 5, 5, 5],
    #     [6, 6, 6, 6, 6],
    #     [7, 7, 7, 7, 7],
    #     [8, 8, 8, 8, 8],
    #     [9, 9, np.nan, np.nan, np.nan],
    #     [0, 0, 0, 0, 0],
    #     [1, 1, 1, 1, 1],
    #     [2, 2, 2, 2, 2],
    #     [3, 3, 3, 3, 3],
    #     [4, 4, 4, 4, 4],
    #     [5, 5, 5, 5, 5],
    #     [6, 6, 6, 6, 6],
    #     [7, 7, 7, 7, 7],
    #     [8, 8, 8, 8, 8],
    #     [np.nan, np.nan, np.nan, 9, np.nan],
    # ])
    folder = './input'
    file_name = '/가평_2019.xlsx'
    df_full = pd.read_excel(folder+file_name)
    # print(df_full)
    df_full_1=df_full.drop([df_full.columns[0],df_full.columns[1]], axis=1)
    # print(df_full_1)
    df_full_1.replace(r'^\s*$', np.nan, regex=True)
    
    # print('df_full',df_full)
    
    block = df_full_1.values
    # block.replace(r'^\s*$', np.nan, regex=True)
    # print(block)

    miss_data = MissData(load_dir='save')
    miss_data.save(block,100)
    # create miss data
    # shape(row,column)
    
    

    org_data = np.zeros(shape=(100,10))
    print('oooooooooolooo',org_data)
    data_x = miss_data.make_missdata(org_data, missrate=0.2)
    # print(data_x)

if __name__ == "__main__":
    main()