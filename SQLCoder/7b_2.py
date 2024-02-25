# -*- coding: utf-8 -*-
"""
https://huggingface.co/defog/sqlcoder-7b-2
https://github.com/defog-ai/sqlcoder/blob/main/inference.py
!pip install torch transformers bitsandbytes accelerate langchain
"""
import sys
import os
import pandas as pd
import torch as th
from transformers import (AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline)
from langchain import PromptTemplate

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(device)

# ----------------------------------------------------------------------------------------------------------------
# path
path_project = "C:/my_project/MyGit/Machine-Learning-Column/SQLCoder"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# 1 - load LLM
checkpoint = "defog/sqlcoder-7b-2"

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
    cache_dir=path_model,
    force_download=False,
    local_files_only=True,
    trust_remote_code=True
    )

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
    cache_dir=path_model,
    force_download=False,
    local_files_only=True,
    trust_remote_code=True,
    device_map=device,
    torch_dtype=th.bfloat16,
    # load_in_8bit=True,
    # load_in_4bit=True,
    use_cache=True
    )

# ----------------------------------------------------------------------------------------------------------------
# 2 - set table metadata
table_metadata = """CREATE TABLE t_order (
   datasource_id STRING, -- Data source ID
   hotel_city_id STRING, -- Hotel's City ID
   hotel_city_name STRING, -- Hotel's city name
   hotel_id BIGINT, -- ID of the hotel
   hotel_name STRING, -- Name of the hotel
   hotel_star_grade STRING, -- Star rating of the hotel
   order_create_time STRING, -- Order creation time
   order_checkin_date STRING, -- Check-in date of the order
   order_id STRING -- ID of the order
 );
"""

# ----------------------------------------------------------------------------------------------------------------
# 3 - set template
template = '''### Task
Generate a SQL query to answer [QUESTION]{user_question}[/QUESTION]

### Database Schema
The query will run on a database with the following schema:
{table_metadata_string_DDL_statements}

### Answer
Given the database schema, here is the SQL query that [QUESTION]{user_question}[/QUESTION]
[SQL]
'''

# ----------------------------------------------------------------------------------------------------------------
# 4 - inference
text = "Which is the hotel with the largest order book in 2023 and what is its order book?"

prompt = PromptTemplate.from_template(template)
prompt_format = prompt.format(user_question=text, table_metadata_string_DDL_statements=table_metadata)
print(prompt_format)

# make sure the model stops generating at triple ticks
# eos_token_id = tokenizer.convert_tokens_to_ids(["```"])[0]
eos_token_id = tokenizer.eos_token_id

pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    do_sample=False,
    return_full_text=False,  # added return_full_text parameter to prevent splitting issues with prompt
    num_beams=5,  # do beam search with 5 beams for high quality results
)

t0 = pd.Timestamp.now()
outputs = pipe(
    prompt_format,
    num_return_sequences=1,
    eos_token_id=eos_token_id,
    pad_token_id=eos_token_id,
)
t1 = pd.Timestamp.now()
print(outputs)
print(t1 - t0)

result = outputs[0]["generated_text"].split(";")[0].split("```")[0].strip() + ";"
print(result)

# ----------------------------------------------------------------------------------------------------------------
# 5 - inference - 2
inputs = tokenizer(prompt_format, return_tensors="pt").to("cuda")

t0 = pd.Timestamp.now()
generated_ids = model.generate(
    **inputs,
    num_return_sequences=1,
    eos_token_id=eos_token_id,
    pad_token_id=eos_token_id,
    max_new_tokens=300,
    do_sample=False,
    num_beams=5
)
t1 = pd.Timestamp.now()
print(generated_ids)
print(t1 - t0)

outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
th.cuda.empty_cache()

# result = outputs[0].split("```sql")[-1].split("```")[0].split(";")[0].strip() + ";"  # 7b
result = outputs[0].split("[SQL]\n")[-1].split(";")[0].strip() + ";"
print(result)
