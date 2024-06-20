import re
import requests
import csv

class QuestionProcessor:
    def __init__(self):
        pass

    def read_txt(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as file:
            str_content = file.read()
        # 使用正则表达式分割字符串
        pattern = re.compile(r'(\d+(?:-\d+)?):(.+)')

        # 分割字符串并提取页码和查询内容
        questions_list = []
        for row in str_content.strip().split('\n'):
            match = pattern.match(row)
            if match:
                page = match.group(1)
                query = match.group(2)
                # 创建一个字典来存储每个问题的数据
                question_data = {
                    'query': query,
                    'page': page,
                    'groundtruth': '',
                    'context': self.get_context(query),
                    'Source': '',
                }
                # 将问题数据添加到数据字典中
                questions_list.append(question_data)
        return questions_list

    def remove_note_and_after(self, answer):
        # 查找"Note:"的索引
        note_index = answer.lower().find("Note:")
        if note_index != -1:
            # 去除"Note:"及其后面的内容
            answer = answer[:note_index].strip()
        return answer

    def read_csv(self, csv_path):
        questions_list = []

        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # 去除正确答案中的"Correct"
                if 'Checking Result' in row:
                    correct_answer = row['Checking Result'].replace("Correct", "").strip()

                    correct_answer = row['Checking Result'].replace("Incorrect", "").strip()
                    # 去除"Note:"及其后面的内容
                    correct_answer = self.remove_note_and_after(correct_answer)
                    # 创建一个字典来存储每个问题的数据
                    question_data = {
                        'query': row['Question'],
                        'groundtruth': correct_answer,
                        'context': self.get_context(row['Question']),
                        'Source': row['Source'],
                        'page': '',
                    }
                    # 将问题数据添加到数据字典中
                    questions_list.append(question_data)
                else:
                    print(f"Warning: 'Checking Result' not found in row {row}")
        return questions_list

    def get_context(self, userquery):
        result = requests.post("http://192.168.101.15:7374/api/v1/retrieve", json=dict(query=userquery))
        context = ""
        for item in result.json():
            # 提取每个元素中的page_content和metadata
            page_content = item[0]['page_content']
            metadata = item[0]['metadata']
            # 从metadata中提取文件名、页码和排名
            file_name = metadata['file_name']
            page = metadata['page']
            rank = item[1]
            # 创建标准格式的文档字符串
            doc = f"""
            ---- Source File: {file_name} - Page {page} - Rank {rank} ----
            {page_content}"""
            context = context + doc
        return context

    def write_to_csv(self, data_list, output_file):
        # 获取列标题（字典的键）
        headers = data_list[0].keys() if data_list else []

        # 打开一个新的文件用于写入，如果文件存在则覆盖
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            # 写入列标题
            writer.writeheader()
            # 写入数据行
            for refer_dict in data_list:
                writer.writerow(refer_dict)

        print(f"CSV file '{output_file}' has been created.")

# 使用示例
if __name__ == "__main__":
    processor = QuestionProcessor()
    csv_path = '/home/weiming.huang/sailor/ragdata/Final_code/Task2-80 Q&A(New).csv'
    txt_path = '/home/weiming.huang/sailor/ragdata/Final_code/questions.txt'
    # data_list = processor.read_csv(csv_path)
    data_list = processor.read_txt(txt_path)
    # data_list.extend(processor.read_txt(txt_path))
    processor.write_to_csv(data_list, '/home/weiming.huang/sailor/ragdata/Final_code/txt_eval.csv')