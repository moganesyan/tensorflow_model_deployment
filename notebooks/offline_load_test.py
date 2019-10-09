import json
import requests
import pickle
from datetime import datetime

with open('payload.pkl','rb') as fid:
    payload = pickle.load(fid)

time_start = datetime.utcnow()
repeat = 100
for i in range(repeat):
    prediction = requests.post('http://localhost:8501/v1/models/coco_test:predict', data=json.dumps(payload), headers=headers)
    print(i)
    print(prediction.status_code)
time_end = datetime.utcnow()
time_elapsed_sec = (time_end - time_start).total_seconds()

print('Total elapsed time: {} seconds'.format(time_elapsed_sec))
print('Average latency per batch: {} seconds'.format(time_elapsed_sec/repeat))