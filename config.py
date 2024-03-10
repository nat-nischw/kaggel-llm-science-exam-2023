################## GENERATE PROMPT CONFIGs ##################
MODEL_GEN = 'gpt-3.5-turbo-16k'
MAX_TOKEN = 8192
TEMPERATURE = 0 
STEM_WEIGHTS = [1, 1.25, 1, 1]
STEM = {
        "S": ["Category:Applied_sciences", "Category:Biotechnology", "Category:Biology", "Category:Natural_history"],
        "T": [
            "Category:Technology_strategy", "Category:Technical_specifications", "Category:Technology_assessment", 
            "Category:Technology_hazards", "Category:Technology_systems", "Category:Hypothetical_technology", 
            "Category:Mobile_technology", "Category:Obsolete_technologies", "Category:Philosophy_of_technology", 
            "Category:Real-time_technology", "Category:Software", "Category:Technology_development", 
            "Category:Computing", "Category:Artificial_objects", "Category:Technological_change", 
            "Category:Technical_communication", "Category:Technological_comparisons"
        ],
        "E": ["Category:Engineering_disciplines", "Category:Engineering_concepts", "Category:Industrial_equipment", "Category:Manufacturing"],
        "M": ["Category:Fields_of_mathematics", "Category:Physical_sciences"]
    }
EXCLUDE_CATEGORIES = set([
        "Category:Technology", "Category:Mathematics", "Category:Works about technology", 
        "Category:Technology evangelism", "Category:Artificial objects", "Category:Fictional physical scientists"
])

OPTIONS_SET = set(("option_1", "option_2", "option_3", "option_4", "option_5"))
RESPONSE_KEYS_SET = set(("question", "option_1", "option_2", "option_3", "option_4", "option_5", "answer"))

DELIMITER = "####"
SYSTEM_MESSAGE = f"""
You will be provided with TEXT from wikipedia. \
The TEXT will be delimited with {DELIMITER} characters.
Output a python list of 5 dict objects, where each object is \
a multiple choice question whom answers should be in \
the given TEXT and that has 5 choices each and has the following format:
    'question': <question on the TEXT>
    'option_1': <question answer option>
    'option_2': <question answer option>
    'option_3': <question answer option>
    'option_4': <question answer option>
    'option_5': <question answer option>
    'answer': <answer option key label>

You should tell me which one of your proposed options is right \
by assigning the corresponding option's key label in the 'answer' field.

The question, the answer and question answer options should be broad, \
challenging, long, detailed and based on the TEXT provided.

Only output the list of objects, with nothing else.
"""

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
VER=1
# PARAMETER EFFICIENT FINE TUNING
# PEFT REQUIRES 1XP100 GPU NOT 2XT4
USE_PEFT = False
# NUMBER OF LAYERS TO FREEZE
# DEBERTA LARGE HAS TOTAL OF 24 LAYERS
FREEZE_LAYERS = 6
# BOOLEAN TO FREEZE EMBEDDINGS
FREEZE_EMBEDDINGS = False
# LENGTH OF CONTEXT PLUS QUESTION ANSWER
MAX_INPUT = 1280
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
BATCH_SIZE = 2
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
REPORT_TO = "wandb"
# OVERWRITE OUTPUT DIRECTORY
OVERWRITE_OUTPUT_DIR = True
# LOAD BEST MODEL AT END
LOAD_BEST_MODEL_AT_END = True

################## INPUT/OUTPUT DIRECTORY ##################
OUTPUT_DIR = f'./output/{MODEL}-ver{VER}'

