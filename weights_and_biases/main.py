import os
import wandb
import random

from dotenv import load_dotenv


path_project = os.path.dirname(__file__)
load_dotenv(dotenv_path=os.path.join(path_project, "wandb.env"))
api_key = os.getenv("WANDB_API_KEY")

wandb.login(key=api_key)

epochs = 10
lr = 0.01
run = wandb.init(
    project="weights-and-biases", 
    config={
        "epochs": epochs, 
        "learning_rate": lr
        }
    )

offset = random.random() / 5
print(f"lr: {lr}")

# Simulate a training run
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    wandb.log({"accuracy": acc, "loss": loss})

# run.log_code()

'''
Visit https://wandb.ai/home to view recorded metrics such as accuracy and loss and how they changed during each training step. 
The following image shows the loss and accuracy tracked from each run. 
Each run object appears in the Runs column with generated names.
'''



if __name__ == "__main__":
    print(path_project)