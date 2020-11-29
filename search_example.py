import veriservice
from text_data import TextData

service = "localhost:5678"
veriservice.init_service(service, "./tmp")
client = veriservice.VeriClient(service, "news")

data = TextData(client)

res = data.search("Best movies")
print(res)