from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
 
tokenizer = AutoTokenizer.from_pretrained("bhavan2410/bias-lens-detection-model")
model = AutoModelForSequenceClassification.from_pretrained("bhavan2410/bias-lens-detection-model")

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
result = classifier("Women are often bad drivers.", return_all_scores=True)
print(result)

# Output: [[{'label': 'Non-Stereotype', 'score': 0.04293655604124069}, {'label': 'Stereotype', 'score': 0.9570633769035339}]]