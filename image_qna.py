from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image

# prepare image + question
url = "https://scontent.fjlr1-1.fna.fbcdn.net/v/t39.30808-6/289010320_7430303097043711_3063447666392591890_n.jpg?_nc_cat=111&ccb=1-7&_nc_sid=52f669&_nc_ohc=-JWNpOHn0NMAX8tvV_u&_nc_ht=scontent.fjlr1-1.fna&oh=00_AfCzQu1Ci7yRu3reCP5_bx3SqPL9s7UkYZw1xt7K4lTamQ&oe=6513F416"
image = Image.open(requests.get(url, stream=True).raw)
text = "are the prople dancing?"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# prepare inputs
encoding = processor(image, text, return_tensors="pt")

# forward pass
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])
