################## EVALUTION CONFIGs ##################
MODEL_EVA = 'NousResearch/Nous-Hermes-Llama2-13b'
EVA_PATH_CSV = './data/qa.csv'
# report to wandb
PROJECT_NAME = 'kaggle-llm-science-exam-2023' 
PROMPT_TEMP = """Answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D, E]
Question: {prompt}\n
A) {a}\n
B) {b}\n
C) {c}\n
D) {d}\n
E) {e}\n

Answer:"""

################## General CONFIGs ##################
VER=11
# PARAMETER EFFICIENT FINE TUNING
# PEFT REQUIRES 1XP100 GPU NOT 2XT4
USE_PEFT = False
# NUMBER OF LAYERS TO FREEZE
# DEBERTA LARGE HAS TOTAL OF 24 LAYERS
FREEZE_LAYERS = 18
# BOOLEAN TO FREEZE EMBEDDINGS
FREEZE_EMBEDDINGS = True
# LENGTH OF CONTEXT PLUS QUESTION ANSWER
MAX_INPUT = 512
# HUGGING FACE MODEL
MODEL = 'microsoft/deberta-v3-large'

################## TRAINING PARAMETERS ##################
# EARLY STOPPING PATIENCE
EARLY_STOPPING = 1
# NUMBER OF EPOCHS
EPOCHS = 2
# LEARNING RATE
LR = 2e-6
# WARMUP RATIO
WARMUP_RATIO = 0.8
# BATCH SIZE
BATCH_SIZE = 4
# EVALUATION STRATEGY
EVAL_STRATEGY = 'epoch'
# SAVE STRATEGY
SAVE_STRATEGY = 'epoch'
# METRIC FOR BEST MODEL
METRIC_FOR_BEST_MODEL = 'map@3'
# LR SCHEDULER TYPE
LR_SCHEDULER_TYPE = 'cosine'
# SAVE TOTAL LIMIT
SAVE_TOTAL_LIMIT = 2
# SEED
SEED = 42
# FP16
FP16 = True
# REPORT TO
REPORT_TO = 'none'
# OVERWRITE OUTPUT DIRECTORY
OVERWRITE_OUTPUT_DIR = True

################## INPUT/OUTPUT DIRECTORY ##################
INPUT_TRAIN = './data/train.csv'
INPUT_VAL = './data/val.csv'
INPUT_EVA = './data/eva.csv'
OUTPUT_DIR = f'./output/{MODEL}-ver{VER}'

