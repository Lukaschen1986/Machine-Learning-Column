# -*- coding: utf-8 -*-
"""
https://huggingface.co/tasks
https://github.com/zyds/transformers-code/blob/master/01-Getting%20Started/06-evaluate/evaluate.ipynb
"""
import os
import torch as th
import evaluate
from evaluate.visualization import radar_plot


device = th.device("cuda" if th.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------------------------------------------
# 路径
path_project = "C:/my_project/MyGit/Machine-Learning-Column/hugging_face"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# 评估
lst_eval_modules = evaluate.list_evaluation_modules(include_community=False, with_details=True)

accuracy = evaluate.load("accuracy")
print(accuracy.description)
print(accuracy.inputs_description)
results = accuracy.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0])

# for (ref, pred) in zip([0,1,0,1], [1,0,0,1]):
#     accuracy.add(references=ref, predictions=pred)
# accuracy.compute()

# for (refs, preds) in zip([[0,1],[0,1]], [[1,0],[0,1]]):
#     accuracy.add_batch(references=refs, predictions=preds)
# accuracy.compute()

clf_metrics = evaluate.combine(["accuracy", "f1", "recall", "precision"])
clf_metrics.compute(predictions=[0, 1, 0], references=[0, 1, 1])

# ----------------------------------------------------------------------------------------------------------------
# plot
data = [
   {"accuracy": 0.99, "precision": 0.8, "f1": 0.95, "latency_in_seconds": 33.6},
   {"accuracy": 0.98, "precision": 0.87, "f1": 0.91, "latency_in_seconds": 11.2},
   {"accuracy": 0.98, "precision": 0.78, "f1": 0.88, "latency_in_seconds": 87.6}, 
   {"accuracy": 0.88, "precision": 0.78, "f1": 0.81, "latency_in_seconds": 101.6}
   ]
model_names = ["Model 1", "Model 2", "Model 3", "Model 4"]
plot = radar_plot(data=data, model_names=model_names)








