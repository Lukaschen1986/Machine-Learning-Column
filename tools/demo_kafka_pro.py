# -*- coding: utf-8 -*-
'''
https://kafka-python.readthedocs.io/en/master/usage.html
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple kafka-python
'''
import json
from kafka import KafkaProducer


producer = KafkaProducer(
    bootstrap_servers="127.0.0.1:9092",
    value_serializer=lambda x: json.dumps(x).encode("utf-8"),
    api_version=(2, 0, 2)
    )

t = "test_topic"
k = b"test_key"
v = {
     "name": "Lukas",
     "gender": "male",
     "age": 37
     }

producer.send(topic=t, key=k, value=json.dumps(v))
producer.close()