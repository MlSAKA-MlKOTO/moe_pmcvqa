import json
import csv
with open('./validate.json','r') as file:
    questions=json.loads(file.read())

with open('./validate.csv','w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['img_id,img_name,question,answer'])
    for line in questions:
        if line['q_lang']=='en':
            writer.writerow([line['qid'],line['img_name'],line['question'],line['answer']])