# Scaling Up LLMs Performance for Multi-Choice Science Exams Using RAG (Retrieval Augmented Generation ), TF-IDF (Term Frequency-Inverse Document Frequency), and FAISS (Facebook AI Similarity Search)
## Kaggle - LLM Science Exam - 2023 
## Overall 
<img src="asset/simple_pipeline_unlocked.png" width="1200" height="400">

In the Kaggle competition, we are not able to unlock certain features because Kaggle's environment does not support them. Therefore, we choose to use some cases for testing and submit them to see the momentum on the Leaderboard. This is one of the methods supported on the Kaggle environment. We might refer to the technique as RAG, which is available in the advanced system tier-3 methodologies by [(Gao et al.,2024)](https://arxiv.org/abs/2312.10997) in their survey on Retrieval-Augmented Generation for Large Language Models.

## Installation
```
bash script/run.sh
```
```
bash script/data_prep.sh
```
## Retrieval Augmented Generation (RAG)

Our study is focused on the advancement of Large Language
Models (LLMs) in the context of multi-choice science examinations. Central to our methodology is the implementation [(Luo et al.,2023)](https://arxiv.org/abs/2305.15225) for effective data augmentation. In the face of limited GPU resources, specifically the P100 available on Kaggle, we have opted for the [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) algorithm as our primary means of embedding vectors derived from our preparatory data. To prioritize and order the relevance of this information, we have employed the TF-IDF (Term Frequency-Inverse Document Frequency) method. Furthermore, for efficient data retrieval and storage, we have utilized FAISS (Facebook AI Similarity Search) to create and manage an Index format.

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

## Data Preparation

### Data sources and sizes used for data preparation

| **Data Sources** | **Size** |
| ---------------- | -------- |
| STEM Wiki Scraping | 3.7K |
| [Cohere-wikipedia](https://huggingface.co/datasets/Cohere/wikipedia-22-12-en-embeddings) | 35.2M |
| [Wikipedia Plaintext (2023-07-01)](https://www.kaggle.com/datasets/jjinho/wikipedia-20230701) | 21M |
| [270K Wikipedia STEM articles](https://www.kaggle.com/datasets/mbanaei/all-paraphs-parsed-expanded) | 270K |

We have meticulously selected data from four distinct sources as delineated in the table above. These sources include:  
1. STEM topics from STEM Wiki Scraping, processed with GPT-3.5-16k for the generation of this self dataset.  
2. [Cohere/wikipedia-22-12-en-embeddings](https://huggingface.co/datasets/Cohere/wikipedia-22-12-en-embeddings), from which we have chosen the TOP-100 articles and processed with Multilingual-22-12.  
3. Wikipedia Plaintext (as of 2023-07-01), a public dataset available through the Kaggle community in disunion sessions, focusing on the TOP-100 articles.  
4. A collection of 270K Wikipedia STEM articles, also publicly available through the Kaggle community in disunion sessions, and included in its entirety in our study.

### Train-Validation Set Methods

In our evaluation of the LLM-as-a-judge prompt, we utilized the [NousResearch/Nous-Hermes-Llama2-13b](https://huggingface.co/NousResearch/Nous-Hermes-Llama2-13b) model, incorporating the Model Evaluation Loss of INSTRUCTION MINING [(Cao et al., 2023)](https://arxiv.org/abs/2307.06290) Indicator Method. This method distinguishes the difficulty levels of science exam questions as either hard or simple, based on the quality of instructions. We employed the MAP@3 score for this differentiation, with scores ≤ 0.6 i indicating hard questions, while higher scores denoted simpler questions. Our Validation (Val) set commenced with a randomly sampled dataset of 200. We then calculated the initial size of our training set based on the leverage ratios between the training and validation sets as follows:

$$
T_{\text{initial}} = V_{\text{initial}} \times \left( \frac{L_{\text{train}}}{L_{\text{val}}} \right) = 200 \times \left( \frac{74}{11.25} \right)
$$


This equation allowed us to deduce the initial size of the training set, utilizing the established leverage ratios and the initial size of the validation set. In the final train-validation set of our methods, we chose to employ a specifically curated dataset, [natnitaract/kaggel-llm-science-exam-2023-RAG](https://huggingface.co/datasets/natnitaract/kaggel-llm-science-exam-2023-RAG), which was an instrumental part of our strategy in the recent competition. This dataset, tailored to the unique demands of our research, provided an invaluable resource in fine-tuning our model and refining our approach to effectively address the challenges posed in the competition.

## Result
We compared our test set evaluation and training context RAG with the H2O.ai team. For more details, check out the report on [Fine-tuning a model with RAG vs. No-RAG](https://wandb.ai/nat-nitarach/kaggel-llm-science-exam-2023/reports/Fine-tuning-model-with-RAG-vs-No-RAG--Vmlldzo3MDk3NjMz).


## Team Members
Teetouch Jaknamon - [@TeetouchQQ](https://github.com/TeetouchQQ), Kunat Pipatanakul (Guest) - [@kunato](https://github.com/kunato), Sittipong Sripaisarnmongkol (Guest) - [@pongib](https://github.com/pongib), Phatrasek Jirabovonvisut (Guest) - [@yoyoismee](https://github.com/yoyoismee),Natapong Nitarach- [@nat-nischw](https://github.com/nat-nischw)

## Mentor Spotlight
Chris Deotte - [@cdeotte](https://github.com/cdeotte), Mohammadreza Banaei - [@MohammadrezaBanaei](https://github.com/MohammadrezaBanaei), Kaggle Community - [Kaggle - LLM Science Exam](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion?sort=votes)

## Limitations and Discussion
### Fine-tune: 

In this training iteration, we have not yet employed [QLoRA (Efficient Finetuning of Quantized LLMs)](https://arxiv.org/abs/2305.14314) as a fine-tuning technique. QLoRA is known for its efficiency in reducing the computational cost of training. This technique facilitates the adjustment of input context length, thereby optimizing the model's capacity to handle varying lengths of input data. It is worth mentioning that QLoRA can be conceptually linked to the [LongLoRA)](https://arxiv.org/abs/2309.12307) and [LongAlpaca)](https://github.com/dvlab-research/LongLoRA) for Long-context LLMs technique, as both share a commonality in their approach to handling input context length adjustments. However for the intent of training lora models, we created other lab's methodologies available here [lingjoor-research/finetuning-model-qlora](https://github.com/lingjoor-research/finetuning-model-qlora) for use in other labs.