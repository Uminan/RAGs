from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from tqdm import tqdm
from OpenAIAPILikeLLM import OpenAIAPILikeLLM
import csv




class EvalAgent:
    def __init__(self, config, eval_path):
        self.llm = OpenAIAPILikeLLM(**config)
        self.eval_list = csv_reader(eval_path)
        self.sum = 0
        self.n = 0
        self.EVALUATION_PROMPT = """###Task Description:
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
    
    

    def evaluation(self):
        correct_list = []
        error_list=[]
        for item in tqdm(self.eval_list, desc="Processing items"):
            full_response = ""

            query = item["query"]
            model_answer = item["model_answer"]
            context = item["context"]

            chat_history = [{"role": "system", "content": self.EVALUATION_PROMPT.format(context=context, query=query, model_answer=model_answer)}]

            for response in self.llm.generate(chat_history):
                full_response += response.choices[0].delta.content
                # print(response.choices[0].delta.content, end='', flush=True)
                

            _, total_rating_str = full_response.split("[RESULT]:")
            total_rating = int(total_rating_str.split("\n")[0])
            self.sum += total_rating
            print(total_rating)
            if total_rating < 5:
                error_list.append({
                    "Eval_score": total_rating,
                    "comment": full_response,
                    "query": query,
                    "response": response,
                    "context": context
                })
            else:
                correct_list.append({
                    "Eval_score": total_rating,
                    "comment": full_response,
                    "query": query,
                    "response": response,
                    "context": context
                })
            self.n += 1
        print(self.sum / self.n)
        return correct_list,error_list
    
    def save_to_csv(self, evaluation_list, filename):
        headers = evaluation_list[0].keys() if evaluation_list else []
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for evaluation_dict in evaluation_list:
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
                            'model_answer': row['model_answer'],
                            'context': row['context'],
                        }
                data_list.append(data_dict)
        return data_list
    
    

if __name__ == "__main__":
    config = {
        'base_url': "http://192.168.101.15:8087/v1",
        'model': "qwen",
        'max_tokens': 512,
        'temperature': 0,
    }
    
    path = "/home/weiming.huang/sailor/ragdata/Final_code/Model_Answer.csv"
    agent = EvalAgent(config, path)
    correct_list,error_list = agent.evaluation()
    agent.save_to_csv(correct_list, '/home/weiming.huang/sailor/ragdata/Final_code/Eval_correct.csv')
    agent.save_to_csv(error_list, '/home/weiming.huang/sailor/ragdata/Final_code/Eval_error.csv')