#!/usr/bin/env python

# Copyright 2015 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import uuid
import pickle
from datetime import datetime
from locust import HttpLocust, TaskSet, task
import json

class MetricsTaskSet(TaskSet):
    _deviceid = None

    def on_start(self):
        self._deviceid = str(uuid.uuid4())
        with open('/locust-tasks/test_image.jpg','rb') as textfile:
            bytestring = textfile.read()
        self.payload = {'image': bytestring}
    def on_stop(self):
        print('Done')

    @task(1)
    def predict_image(self):
        self.client.post('/predict', files = self.payload)

class MetricsLocust(HttpLocust):
    task_set = MetricsTaskSet
