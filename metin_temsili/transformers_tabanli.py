from transformers import AutoTokenizer, AutoModel
import torch
#model ve tokenizer yükleme
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

#input text 
text = "Transformers can be used for natural language processing"

#metni tokenlara çevirmek
inputs = tokenizer(text, return_tensors="pt") #çıktı pytorch tensoru olacak

#model kullanarak metin temsili oluştur
with torch.no_grad(): #gradyan hesaplamasını kapat ki bellek tasarrufu sağlasın
    outputs = model(**inputs)

#modelin çıkışından son gizli durumu alalim
last_hidden_state = outputs.last_hidden_state #tum token çıktılarını almak için 

#ilk tokenın çıktısını alalım
first_token_embeding = last_hidden_state[0,0,:].numpy()

print(f"metin temsili: {first_token_embeding}")