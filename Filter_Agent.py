import csv
from tqdm import tqdm
from openai import OpenAI
from OpenAIAPILikeLLM import OpenAIAPILikeLLM

class QuestionFilterAgent:
    def __init__(self, config, data_list, k=1):
        self.config = config
        self.data_list = data_list
        self.k = k
        self.llm = OpenAIAPILikeLLM(**config)
        self.filter_question = []
        self.question_groundedness_critique_prompt = """
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

    def generate_evaluation(self, userquery, context):
        chat_history = [{"role": "system", "content": self.question_groundedness_critique_prompt.format(question=userquery, context=context)}]
        full_response = ""
        for response in self.llm.generate(chat_history):
            full_response += response.choices[0].delta.content
        evaluation_str, total_rating_str = full_response.split("Total rating: ")
        total_rating = float(total_rating_str.split("\n")[0])
        evaluation_content = evaluation_str.split("Evaluation: ")[1]
        return total_rating, evaluation_content

    def filter_questions(self):
        n = 0
        for item in tqdm(self.data_list, desc="Processing items"):
            userquery = item["query"]
            context = item["context"]
            total_rating, evaluation_content = self.generate_evaluation(userquery, context)
            if total_rating > 4:
                evaluation_dict = {
                    "total_rating": total_rating,
                    "evaluation_content": evaluation_content,
                    'query': userquery,
                    'groundtruth': item['groundtruth'],
                    'context': context,
                    'source': item['Source'],
                    'page': item['page'],
                }
                self.filter_question.append(evaluation_dict)
                n += 1
        # print('total_questions', n)

    def save_to_csv(self, filename):
        headers = self.filter_question[0].keys() if self.filter_question else []
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for evaluation_dict in self.filter_question:
                writer.writerow(evaluation_dict)
        print(f"CSV file '{filename}' has been created.")
        
def csv_reader(path):
    data_list = []
    # Open the CSV file
    with open(path, mode='r', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        # Loop through the CSV rows
        for row in csv_reader:
            # Create a dictionary for the current row with desired keys
            data_dict = {
                        'query': row['query'],
                        'groundtruth': row['groundtruth'],
                        'context': row['context'],
                        'source': row['source'],
                        'page': row['page'],
                    }
            data_list.append(data_dict)
    return data_list

def main():
    config = {
        # 'base_url': "http://192.168.101.15:8081/v1",
        # 'model': "llama3",
        # 'base_url': "http://8.219.154.252:8082/v1",#llama3 4bit
        # 'model': "llama3",
        # 'base_url': "http://192.168.101.15:8086/v1",
        # 'model': "glm4",
        'base_url': "http://192.168.101.15:8087/v1",
        'model': "qwen",
        'max_tokens': 512,
        'temperature': 0,
    }
    path="/home/weiming.huang/sailor/ragdata/Final_code/txt_eval.csv"
    data_list = csv_reader(path)  # 请在此处填充实际的数据列表
    agent = QuestionFilterAgent(config, data_list)
    agent.filter_questions()
    agent.save_to_csv('/home/weiming.huang/sailor/ragdata/Final_code/filter_question.csv')

if __name__ == "__main__":
    main()