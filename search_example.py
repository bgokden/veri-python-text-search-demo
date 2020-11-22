import veriservice
from text_data import TextData

service = "localhost:5678"
client = veriservice.VeriClient(service, "news")

data = TextData(client)

res = data.search("Best movies")
print(res)