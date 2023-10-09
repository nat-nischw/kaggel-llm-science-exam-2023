################## tranning environment ##################
pip install git+https://github.com/huggingface/transformers
pip install --upgrade wandb
pip install sentencepiece
pip install accelerate -U
pip install transformers[torch]
pip install datasets colorama 

################## evaluate environment ##################
pip install -U bitsandbytes
pip install -U git+https://github.com/huggingface/transformers.git
pip install -U git+https://github.com/huggingface/peft.git
pip install -U git+https://github.com/huggingface/accelerate.git