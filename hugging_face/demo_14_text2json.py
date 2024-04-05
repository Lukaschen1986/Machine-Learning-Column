# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import torch as th
from torch.utils.tensorboard import SummaryWriter
from datasets import (load_dataset, load_from_disk, Dataset)
from transformers import (AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, DataCollatorWithPadding, DataCollatorForLanguageModeling,
                          DataCollatorForSeq2Seq, DataCollatorForTokenClassification)
from transformers.integrations import TensorBoardCallback
from peft import (LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training)
from trl import SFTTrainer


device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(device)

# ----------------------------------------------------------------------------------------------------------------
# path
path_project = "C:/my_project/MyGit/Machine-Learning-Column/hugging_face"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# load dataset official
filename = "text-to-json/json-mode-eval-train.arrow"
filename = "text-to-json/50K_deduplicated_ner_indexes_name_country_alpaca_format_json_response.json"
filename = "text-to-json/100K_deduplicated_ner_indexes_name_country_alpaca_format_json_response_all_cases.json"
filename = "text-to-json/500K_deduplicated_ner_indexes_multiple_organizations_locations_alpaca_format_json_response_all_cases.json"

dataset = load_dataset(
    path="arrow",
    data_files=os.path.join(path_data, filename),
    split="all"
)

# json-mode-eval-train
print(dataset)
print(dataset[0])
'''
{
	"title": "WirelessAccessPoint",
	"type": "object",
	"properties": {
		"ssid": {
			"title": "SSID",
			"type": "string"
		},
		"securityProtocol": {
			"title": "SecurityProtocol",
			"type": "string"
		},
		"bandwidth": {
			"title": "Bandwidth",
			"type": "string"
		}
	},
	"required": ["ssid", "securityProtocol", "bandwidth"]
}
'''

# 50K_deduplicated_ner_indexes_name_country_alpaca_format_json_response
print(dataset)
print(dataset[0])

# 100K_deduplicated_ner_indexes_name_country_alpaca_format_json_response_all_cases
print(dataset)
print(dataset[0])

# 500K_deduplicated_ner_indexes_multiple_organizations_locations_alpaca_format_json_response_all_cases
print(dataset)
print(dataset[0])
'''
{
	"organizations": [{
		"name": "max planck institute for biological cybernetics",
		"location": ""
	}, {
		"name": "forschungszentrum julich",
		"location": "germany"
	}]
}
'''



