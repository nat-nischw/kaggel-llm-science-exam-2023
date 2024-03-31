# Scaling Up LLMs Performance for Multi-Choice Science Exams Using RAG (Retrieval Augmented Generation ), TF-IDF (Term Frequency-Inverse Document Frequency), and FAISS (Facebook AI Similarity Search)
## Kaggle - LLM Science Exam - 2023 
## Overall 
<img src="asset/simple_pipeline_unlocked.png" width="1200" height="400">

In the Kaggle competition, we are not able to unlock certain features because Kaggle's environment does not support them. Therefore, we choose to use some cases for testing and submit them to see the momentum on the Leaderboard. This is one of the methods supported on the Kaggle environment. We might refer to the technique as RAG, which is available in the advanced system in tier-3.

## Installation
```
bash script/run.sh
```
```
bash script/data_prep.sh
```
## Retrieval Augmented Generation (RAG)

In the face of limited GPU resources, specifically the P100 available on Kaggle, we have opted for the [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) algorithm as our primary means of embedding vectors derived from our preparatory data. To prioritize and order the relevance of this information, we have employed the TF-IDF (Term Frequency-Inverse Document Frequency) method. Furthermore, for efficient data retrieval and storage, we have utilized FAISS (Facebook AI Similarity Search) to create and manage an Index format.

### Thought of Query

<img src="asset/ToQ.png" alt="Illustration of the Thought of Query (ToQ) System Implementing RAG for Enhanced LLM Performance" style="max-width:1200px; max-height:400px; width: auto; height: auto;">



From the figure above: The system leverages Retrieval Augmented Generation (RAG) to enhance the efficacy of Large Language Models (LLMs) in response generation. This implementation, dubbed Thought of Query (ToQ), incorporates the following elements:

- **RAG**: Consists of the system detailed above.
- **Context Integration**: Context is woven into the System Prompt of LLMs to guide prompt engineering.
- **Prompt Engineering** is bifurcated into two segments:
  1. **Prompt for Context Retrieval**: Aimed at fetching relevant context, the structure is as follows:
     ```
     "Represent this sentence for searching relevant passages:
     + PROMPT (the text of the question being asked)
     + OPTION"
     ```
  2. **System Prompt in LLM**: Utilizes the structure:
     ```
     PROMPT (the text of the question being asked)
     + OPTION
     + CONTEXT
     ```

This strategic integration of contextual data markedly bolsters the LLM's precision and relevance in generating responses.


### Infer
```
python rag/infer_rag.py
```

## Fine-Tuning
```
python train.py --train
```
## Eval-Model
```
python train.py --eval
```

## Team Members
Teetouch Jaknamon - [@TeetouchQQ](https://github.com/TeetouchQQ), Natapong Nitarach- [@nat-nischw](https://github.com/nat-nischw), Kunat Pipatanakul (Guest) - [@kunato](https://github.com/kunato), Sittipong Sripaisarnmongkol (Guest) - [@pongib](https://github.com/pongib), Phatrasek Jirabovonvisut (Guest) - [@yoyoismee](https://github.com/yoyoismee)

## Mentor Spotlight
Chris Deotte - [@cdeotte](https://github.com/cdeotte), Mohammadreza Banaei - [@MohammadrezaBanaei](https://github.com/MohammadrezaBanaei), Kaggle Community - [Kaggle - LLM Science Exam](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion?sort=votes)

## Limitations and Discussion
### Fine-tune: 

In this training iteration, we have not yet employed [QLoRA (Efficient Finetuning of Quantized LLMs)](https://arxiv.org/abs/2305.14314) as a fine-tuning technique. QLoRA is known for its efficiency in reducing the computational cost of training. This technique facilitates the adjustment of input context length, thereby optimizing the model's capacity to handle varying lengths of input data. It is worth mentioning that QLoRA can be conceptually linked to the [LongLoRA)](https://arxiv.org/abs/2309.12307) and [LongAlpaca)](https://github.com/dvlab-research/LongLoRA) for Long-context LLMs technique, as both share a commonality in their approach to handling input context length adjustments. However for the intent of training lora models, we created other lab's methodologies available here [lingjoor-research/finetuning-model-qlora](https://github.com/lingjoor-research/finetuning-model-qlora) for use in other labs.
## Fully Paper (Available in soon): 
