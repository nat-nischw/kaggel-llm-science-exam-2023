################## data environment ##################
mkdir -p data/train-val/
pip install datasets
python fetch_data.py
### *** you can use this code to load data from huggingface datasets *** ###
### from datasets import load_dataset
### dataset = load_dataset("natnitaract/kaggel-llm-science-exam-2023-RAG")
### train_data, val_data = dataset['train'], dataset['validation'] ###

################## genprompt environment ##################
pip install openai itables plotly python-dotenv Wikipedia-API 

################## tranning environment ##################
pip install git+https://github.com/huggingface/transformers
pip install --upgrade wandb
pip install sentencepiece
pip install accelerate -U
pip install transformers[torch]
pip install colorama 

################## evaluate environment ##################
pip install -U bitsandbytes
pip install -U git+https://github.com/huggingface/transformers.git
pip install -U git+https://github.com/huggingface/peft.git
pip install -U git+https://github.com/huggingface/accelerate.git