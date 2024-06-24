from datetime import datetime
import pandas as pd
import time
import math
import csv
import sys

t = 'trace1'

if __name__ == '__main__':
    data = pd.read_csv(sys.argv[1])
    traces = [f'trace{i}' for i in range(1,21)]
    print(traces)
    # print(data.loc[:, "timestamp"])
    # print(data.loc[:, t])
    trray = data.loc[:, "timestamp"]
    for t in traces:
        drray = data.loc[:, t]
        # array = data.loc[:, t].tolist()
        # print(len(array))
        ts = time.time()
        with open('/tmp/%s.tsv' % t, 'w') as out_file:
            for item in zip(trray, drray):
                # print(item)
                tsv_writer = csv.writer(out_file, delimiter='\t')
                tsv_writer.writerow([datetime.utcfromtimestamp(ts).strftime('%Y.%j.%H.%M.')+'{:2.6f}'.format(float(item[0].replace(',','.')))])
                tsv_writer.writerow(['', 'err', item[1].replace(',','.')])
