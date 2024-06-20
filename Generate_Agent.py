import csv
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize
from OpenAIAPILikeLLM import OpenAIAPILikeLLM
from Filter_Agent import csv_reader

class GenerateAgent:
    def __init__(self, config, path, k=1):
        self.config = config
        self.path = path
        self.k = k
        self.llm = OpenAIAPILikeLLM(**config)
        self.filter_question = csv_reader(path)
        self.en_prompt = """You are a Question and Answer assistant. To respond to user questions and retrieve relevant documents, you must think carefully and answer the user's questions as accurately as possible. If you quote relevant documents, please also indicate the page number and location of the quoted document. If the cited document does not contain relevant content, you can directly answer that there is currently no relevant information, and do not make up the answer.
        1.Try to respond using sentences from the retrieval document.
        2.If the user asks in Indonesian, please answer in Indonesian.
        3.If the user asks in English, please answer in English.
        """
        self.en_message = """{prompt_template}
        ```
        Information context taken from the following document.
        ```
        {document_template}
        ```
        """

    def generate_response(self, userquery, context):
        chat_history = [{"role": "system", "content": self.en_message.format(prompt_template=self.en_prompt, document_template=context)}]
        messages = chat_history + [{"role": "user", "content": userquery}]
        full_response = ""
        for response in self.llm.generate(messages):
            full_response += response.choices[0].delta.content
        return full_response

    def calculate_meteor_score(self, groundtruth, model_answer):
        return meteor_score([word_tokenize(groundtruth)], word_tokenize(model_answer))

    def evaluate(self):
        evaluation_list = []
        sum_meteor = 0
        n = 0
        for item in tqdm(self.filter_question, desc="Processing items"):
            userquery = item["query"]
            groundtruth = item["groundtruth"]
            context = item["context"]
            full_response = self.generate_response(userquery, context)
            menteor = self.calculate_meteor_score(groundtruth, full_response)
            sum_meteor += menteor
            n += 1
            evaluation_dict = {
                "meteor_score": menteor,
                "query": userquery,
                "model_answer": full_response,
                "groundtruth": groundtruth,
                "context": context,
                "source": item['source'],
                "page": item['page'],
            }
            evaluation_list.append(evaluation_dict)
        print(sum_meteor / n)
        return evaluation_list

    def save_to_csv(self, evaluation_list, filename):
        headers = evaluation_list[0].keys() if evaluation_list else []
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for evaluation_dict in evaluation_list:
                writer.writerow(evaluation_dict)
        print(f"CSV file '{filename}' has been created.")

def main():
    config ={
        # 'base_url': "http://192.168.101.15:8081/v1",
        # 'model': "llama3",
        # 'base_url': "http://8.219.154.252:8082/v1",#llama3 4bit
        # 'model': "llama3",
        'base_url': "http://192.168.101.15:8087/v1",
        'model': "qwen",
        'max_tokens': 1024,
        'temperature': 0.3,
    }
    path = "/home/weiming.huang/sailor/ragdata/Final_code/filter_question.csv"
    agent = GenerateAgent(config, path)
    evaluation_list = agent.evaluate()
    agent.save_to_csv(evaluation_list, '/home/weiming.huang/sailor/ragdata/Final_code/ Model_Answer.csv')

if __name__ == "__main__":
    main()