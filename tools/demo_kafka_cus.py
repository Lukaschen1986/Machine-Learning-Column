# -*- coding: utf-8 -*-
'''
https://kafka-python.readthedocs.io/en/master/usage.html
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple kafka-python
'''
import json
from kafka import KafkaConsumer


consumer = KafkaConsumer(bootstrap_servers="127.0.0.1:9092")

t = "test_topic"
consumer.subscribe(topics=t)  # 订阅 topic
conn = consumer.bootstrap_connected()

if conn:
    print(consumer.topics())

for msg in consumer:
    print(msg.topic)
    print(msg.key)
    print(json.loads(msg.value))