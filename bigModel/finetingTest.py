'''
https://www.bilibili.com/video/BV1kJWUeoEuG/?p=4&spm_id_from=pageDriver&vd_source=50305204d8a1be81f31d861b12d4d5cf
'''
import jsonlines
import itertools
import pandas as pd
from pprint import pprint
import datasets
from datasets import load_dataset, load_from_disk, Dataset

# pretrained_dataset = load_dataset('EleutherAI/pile', split='train', streaming=True)
pretrained_dataset = load_dataset('monology/pile', split='train', streaming=True)

n = 5
print('Pretrianed dataset:')
top_n = list(itertools.islice(pretrained_dataset, n))
for i in top_n:
    print(i)

prompt_template_qa = """### Question:
  {question}

  ### Answer:
  {answer}"""

question = 'What are the different types of documents available in the repository '
answer = 'Lamini has documentation on Getting Started, Authentication, Question Answer Model'
text_with_prompt_template = prompt_template_qa.format(question=question, answer=answer)
text_with_prompt_template
