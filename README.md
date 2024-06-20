# 说明文档
## 工作总结
主要构建如下文件：

-`ChatModel.py`: 实现多轮对话流式输出。

-`QuestionProcessor.py`: 从txt和CSV文件中读取问题数据，并将其转换为标准格式, 并调取服务获取相关context。将处理后的数据写入CSV文件。

-`Filter_Agent.py`: 对问题集进行筛选构建测试集的Agent，输入为query和context，评估标准为具体性（Groundedness）：问题是否可以从给定的上下文中得到回答？

-`Generate_Agent.py`: 回答用户提问的Agent。对于有groundtruth的问题集，还可计算response与groundtruth的menteor得分。

-`Eval_Agent.py`: 评估response与context的忠实度（Faithfulness）的Agent。衡量生成的答案是否基于给定的事实（检索到的上下文内容）。

### 数据集描述
-`questions.txt`:包括pojk_questions, pojk_questions2和uu_questions 共87条问题。对每个问题检索context后得到`txt_eval.csv`

-`Task2-80 Q&A(New).csv`: 由人工标注过后的word，
表头分别为Question、Answer、Checking Result、Source，其中Checking Result含有人工批注，这里使用`QuestionProcessor.py`对其进行处理得到的答案作为问题的groundtruth。

# ChatModel.py 

## 概述

`ChatModel` 类是一个用于实现对话流式输出的类，基于 Hugging Face 的 `transformers` 库。可载入`sailor`、`llama3`等模型，并支持多种配置选项，包括`model_path`、`device_id`、`temperature`、`max_tokens`等。

## 类定义

### `ChatModel`

#### 初始化

- **参数**:
  - `config`: 一个包含配置选项的字典。默认值会在未指定时使用。
    - `model_path`: 预训练模型的路径。默认值为 `"/mnt/zfs01/snowdar/pretrained/Sailor-7B-Chat"`。
    - `temperature`: 控制生成文本的随机性。默认值为 `0`。
    - `max_tokens`: 生成文本的最大令牌数。默认值为 `512`。
    - `device_id`: 模型运行的设备。默认值为 `"cuda:6"`。
    - `top_p`: 核采样概率。默认值为 `0.95`。
    - `top_k`: 用于 top-k 过滤的令牌数。默认值为 `1`。
    - `prompt`: 初始对话提示。默认值为空列表。

- **方法**:
  - `__init__(self, **config)`: 初始化模型和分词器，并将模型移动到指定设备。
  - `format(self, prompt)`: 使用分词器的聊天模板格式化提示。
  - `run_generation(self)`: 基于当前提示生成回复。
  - `chat(self)`: 启动一个交互式聊天会话。

### 示例用法

```python
if __name__ == "__main__":
    prompt = [
        {"role": "system", "content": "You are an AI research assistant."},
        {"role": "assistant", "content": "Greeting! I am an AI research assistant. How can I help you today?"}
    ]
    config = {
        'prompt': prompt,
        'device_id': 'cuda:6',
        'model_path': "/mnt/ceph01/snowdar/DS/pretrained/Meta-Llama-3-8B-Instruct",
        'max_tokens': 1024,
        'temperature': 0.3,
        'top_p': 0.95,
        'top_k': 1,
    }
    chat_model = ChatModel(**config)
    chat_model.chat()
```

# QuestionProcessor.py

## 概述

`QuestionProcessor` 是一个用于处理问题数据的类。它可以从文本文件和CSV文件中读取问题数据，并将其转换为标准格式的字典列表。此外，它还可以将处理后的数据写入CSV文件。

## 类方法

### `read_txt`

从文本文件中读取问题数据。

- **参数**:
  - `txt_path` (str): 文本文件的路径。
- **返回值**:
  - `questions_list` (list): 包含问题数据的字典列表。

### `remove_note_and_after`

由于`Task2-80 Q&A(New).csv`含有Correct，Note等人工批注，可使用该函数去除答案中的"Note:"及其后面的内容。

- **参数**:
  - `answer` (str): 原始答案字符串。
- **返回值**:
  - `answer` (str): 去除"Note:"及其后面的内容的答案字符串。

### `read_csv`

从CSV文件中读取问题数据。

- **参数**:
  - `csv_path` (str): CSV文件的路径。
- **返回值**:
  - `questions_list` (list): 包含问题数据的字典列表。

### `get_context`

通过API获取问题的上下文信息。

- **参数**:
  - `userquery` (str): 用户查询字符串。
- **返回值**:
  - `context` (str): 包含上下文信息的字符串。

### `write_to_csv`

将处理后的数据写入CSV文件。

- **参数**:
  - `data_list` (list): 包含问题数据的字典列表。
  - `output_file` (str): 输出CSV文件的路径。


## 使用示例

```python
if __name__ == "__main__":
    processor = QuestionProcessor()
    csv_path = '/home/weiming.huang/sailor/ragdata/Final_code/Task2-80 Q&A(New).csv'
    txt_path = '/home/weiming.huang/sailor/ragdata/Final_code/questions.txt'
    # data_list = processor.read_csv(csv_path)
    data_list = processor.read_txt(txt_path)
    # data_list.extend(processor.read_txt(txt_path))
    processor.write_to_csv(data_list, '/home/weiming.huang/sailor/ragdata/Final_code/txt_eval.csv')
```


# Filter_Agent.py 

## 概述

`QuestionFilterAgent` 是一个用于过滤问题的代理类。它使用 `OpenAIAPILikeLLM` 类与语言模型进行交互，根据给定的上下文评估问题的可回答性，并根据评分标准过滤出高质量的问题。评估结果可以保存为 CSV 文件。
### 评估标准
检索到的context不一定能完整回答用户问题，所以需要对问题与context对进行筛选，进行质量检查。首先让模型给出判断理由，再总结出分数。

筛选掉所有小于5分的问题。

评判标准为：具体性（Groundedness）：问题是否可以从给定的上下文中得到回答？

参考：https://arxiv.org/pdf/2312.10003
## Prompt
```
"""
You will be given a context and a question, which are written by Bahasa Indonesia.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and context.

Question: {question}\n
Context: {context}\n
Answer::: """
```

## 类定义

### `QuestionFilterAgent`

#### 初始化方法 (`__init__`)

- **参数**:
  - `config`: 包含语言模型配置的字典。
  - `data_list`: 包含待评估问题的数据列表。
  - `k`: 过滤阈值，默认为1。

- **功能**:
  - 初始化语言模型实例。
  - 初始化过滤问题列表。
  - 定义问题可回答性评估的提示模板。

#### 生成评估方法 (`generate_evaluation`)

- **参数**:
  - `userquery`: 待评估的问题。
  - `context`: 问题的上下文。

- **功能**:
  - 使用语言模型生成评估反馈和评分。
  - 返回总评分和评估内容。

#### 过滤问题方法 (`filter_questions`)

- **功能**:
  - 遍历数据列表，对每个问题进行评估。
  - 根据评分过滤出高质量的问题。
  - 将过滤后的问题添加到过滤问题列表中。



## 使用示例

```python
def main():
    config = {
        'base_url': "http://192.168.101.15:8087/v1",
        'model': "qwen",
        'max_tokens': 512,
        'temperature': 0,
    }
    path = "/home/weiming.huang/sailor/ragdata/Final_code/txt_eval.csv"
    data_list = csv_reader(path)  # 请在此处填充实际的数据列表
    agent = QuestionFilterAgent(config, data_list)
    agent.filter_questions()
    agent.save_to_csv('/home/weiming.huang/sailor/ragdata/Final_code/filter_question.csv')

if __name__ == "__main__":
    main()
```

如下问题就会被筛选剔除
```
question: Bagaimana definisi "Prosesor Data Pribadi"?
content: The context provided is about the Indonesian Personal Data Protection Law, specifically detailing the responsibilities of a Data Processor (Prosesor Data Pribadi). However, it does not directly define what a "Prosesor Data Pribadi" is. The term is mentioned throughout the text, but there is no explicit definition given.
Total rating: 2
```


# Generate_Agent.py 

## 概述

`GenerateAgent` 是一个用于生成和评估模型响应的代理类。它使用 `OpenAIAPILikeLLM` 类与语言模型进行交互，根据给定的上下文生成响应，并计算响应的 METEOR 评分。评估结果可以保存为 CSV 文件。

## Prompt
```
"""
You are a Question and Answer assistant. To respond to user questions and retrieve relevant documents, you must think carefully and answer the user's questions as accurately as possible. If you quote relevant documents, please also indicate the page number and location of the quoted document. If the cited document does not contain relevant content, you can directly answer that there is currently no relevant information, and do not make up the answer.
1.Try to respond using sentences from the retrieval document.
2.If the user asks in Indonesian, please answer in Indonesian.
3.If the user asks in English, please answer in English.
/```
Information context taken from the following document.
/```
{document_template}
"""
```


## 类定义

### `GenerateAgent`

#### 初始化方法 (`__init__`)

- **参数**:
  - `config`: 包含语言模型配置的字典。
  - `path`: 包含过滤问题的 CSV 文件路径。
  - `k`: 过滤阈值，默认为1。

- **功能**:
  - 初始化语言模型实例。
  - 从 CSV 文件中读取过滤问题列表。
  - 定义生成响应的提示模板和消息模板。

#### 生成响应方法 (`generate_response`)

- **参数**:
  - `userquery`: 用户查询。
  - `context`: 查询的上下文。

- **功能**:
  - 使用语言模型生成响应。
  - 返回生成的响应。

#### 计算 METEOR 评分方法 (`calculate_meteor_score`)

- **参数**:
  - `groundtruth`: 真实答案。
  - `model_answer`: 模型生成的答案。

- **功能**:
  - 计算模型生成答案与真实答案之间的 METEOR 评分。
  - 返回 METEOR 评分。

#### 评估方法 (`evaluate`)

- **功能**:
  - 遍历过滤问题列表，对每个问题生成响应并计算 METEOR 评分。
  - 将评估结果添加到评估列表中。
  - 计算并打印平均 METEOR 评分。
  - 返回评估列表。

#### 保存到 CSV 方法 (`save_to_csv`)

- **参数**:
  - `evaluation_list`: 包含评估结果的列表。
  - `filename`: 保存评估结果的 CSV 文件名。

- **功能**:
  - 将评估结果列表保存为 CSV 文件。

## 使用示例

```python
def main():
    config = {
        'base_url': "http://192.168.101.15:8087/v1",
        'model': "qwen",
        'max_tokens': 1024,
        'temperature': 0.3,
    }
    path = "/home/weiming.huang/sailor/ragdata/Final_code/filter_question.csv"
    agent = GenerateAgent(config, path)
    evaluation_list = agent.evaluate()
    agent.save_to_csv(evaluation_list, '/home/weiming.huang/sailor/ragdata/Final_code/Model_Answer.csv')

if __name__ == "__main__":
    main()
```



# Eval_Agent.py 

## 概述

`EvalAgent` 是一个用于评估模型生成答案质量的代理类。它使用 `OpenAIAPILikeLLM` 类与语言模型进行交互，并根据给定的评分标准对模型的回答进行评估。评估结果可以保存为 CSV 文件。

### 评估标准
选择只关注忠实度，即问题是否仅由检索到的上下文得来，不是来自于额外的知识。先让模型给出理由，再输出得分，有助于判断模型回答是否产生幻觉。并强调了输出格式，保证正确提取出打分。

在评估提示中，给出了每个指标的详细描述，采用 1-5 分的评分刻度，这有助于模型精确地确定其指标。

## Prompt

```
"""
###Task Description:
An instruction (might include an Input inside it), a context,a query, a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.
5. Provide your answer as follows:
Answer:::
Feedback: (your rationale for the rating, as a text)
[RESULT]: (your rating, as an integer number between 1 and 5)

###The instruction to evaluate:
Based on the retrieved Indonesian context from "SALINANPERATURAN OTORITAS JASA KEUANGAN UNDANG-UNDANG REPUBLIK INDONESIA" and the question, 
evaluate whether the response content is derived ONLY from the given context WITHOUT introducing additional knowledge.

###The context:
{context}

###The query:
{query}

###Response to evaluate:
{model_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the context?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.
###Answer:"""
```

## 类定义

### `EvalAgent`

#### 初始化方法 (`__init__`)

- **参数**:
  - `config`: 包含语言模型配置的字典。
  - `eval_path`: 包含评估数据的 CSV 文件路径。

- **功能**:
  - 初始化语言模型实例。
  - 从 CSV 文件中读取评估数据。
  - 初始化评估结果的累加和及计数器。

#### 评估方法 (`evaluation`)

- **功能**:
  - 遍历评估数据列表，对每个数据项进行评估。
  - 使用语言模型生成评估反馈和评分。
  - 根据评分将结果分为正确和错误两类。
  - 计算并打印平均评分。
  - 返回正确和错误的结果列表。



## 使用示例

```python
if __name__ == "__main__":
    config = {
        'base_url': "http://192.168.101.15:8087/v1",
        'model': "qwen",
        'max_tokens': 512,
        'temperature': 0,
    }

    path = "/home/weiming.huang/sailor/ragdata/Final_code/Model_Answer.csv"
    agent = EvalAgent(config, path)
    correct_list, error_list = agent.evaluation()
    agent.save_to_csv(correct_list, '/home/weiming.huang/sailor/ragdata/Final_code/Eval_correct.csv')
    agent.save_to_csv(error_list, '/home/weiming.huang/sailor/ragdata/Final_code/Eval_error.csv')
```

低分回答示例：
```
question: Siapa yang dimaksud dengan Pelapor dalam peraturan ini?
content: The response is not derived solely from the given context. It introduces additional knowledge by specifying that "Pelapor" refers to Bank Perkreditan Rakyat (BPR), which is not mentioned in the context provided. The context only defines the responsibilities and obligations of the "Pelapor" without specifying the entity. [RESULT] 2
```
