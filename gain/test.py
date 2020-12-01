from preProcess import PreProcess


if __name__ == "__main__":
    # input = './data'
    input = './data/서상_2019.xlsx'
    p = PreProcess(
        input=input, # file or directory
        target=[0,2,4,5], # if target_all is True, this option ignore
        time=3,
        target_all=True,
        fill_cnt=0
    )

    # make train data
    output_np = p.getDataSet()
    p.npToExcel(output_np, "./output/merge.xlsx")

    # make raw data (directory)
    '''
    output_np_list = p.getRawDataSet()
    idx = 1
    for output_np in output_np_list:
        p.npToExcel(output_np, "./output/raw_{idx}.xlsx".format(idx=idx))
        output_np = p.reverseReShape(output_np)
        p.npToExcel(output_np, "./output/raw_reverse_{idx}.xlsx".format(idx=idx), timeFormat=True)
        idx += 1
    '''

    # make raw data (file)
    '''
    output_np = p.getRawDataSet()
    p.npToExcel(output_np, "./output/raw.xlsx")
    '''

    # make reverse raw data (file)
    '''
    output_np = p.reverseReShape(output_np)
    p.npToExcel(output_np, "./output/raw_reverse.xlsx", timeFormat=True)
    '''

    # test ground
    output_np = p.getRawDataSet()
    print(output_np)
    p.npToExcel(output_np, "./output/raw.xlsx")
    output_np = p.reverseReShape(output_np)
    print(output_np)
    p.npToExcel(output_np, "./output/raw_reverse.xlsx", timeFormat=True)