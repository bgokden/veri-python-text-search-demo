import os
import json
import veriservice

service = "localhost:5678"
veriservice.init_service(service, "./tmp")

client = veriservice.VeriClient(service, "news")
client.create_data_if_not_exists({'no_target': True})

news_json_path = os.path.join('data', 'news.json')
count = 0
with open(news_json_path) as f:
    for line in f:
        row = json.loads(line)
        label = row['label'].strip()
        print(count, label)
        group_label = json.dumps(row['group_label'])
        client.insert(row['feature'], label.encode(), group_label=group_label.encode())
        count = count + 1