from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Charger le modèle et le tokenizer depuis Hugging Face
model_name = 'openai-community/gpt2'  # adapt it
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Générer du texte avec des paramètres de génération appropriés
input_text = "Hi there!\n"  # adapt it
inputs = tokenizer.encode(input_text, return_tensors='pt')

# Paramètres de génération pour améliorer la cohérence
outputs = model.generate(
    inputs,
    max_length=100,  # adapt it
    pad_token_id=tokenizer.eos_token_id,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    temperature=0.7,
    min_p= 1,
    top_p=0.9,
    top_k=1,
    do_sample=True
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
