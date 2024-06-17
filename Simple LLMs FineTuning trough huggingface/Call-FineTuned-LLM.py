from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Chemin vers le répertoire où le modèle et le tokenizer sont sauvegardés
output_dir = './fine_tuned_model'

# Charger le modèle fine-tuné et le tokenizer
model = GPT2LMHeadModel.from_pretrained(output_dir)
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

# Assurer que le `pad_token_id` est défini
tokenizer.pad_token = tokenizer.eos_token

# Préparer le texte d'entrée avec plus de contexte
input_text = """
User: Hi there!\n
A.I:\n
"""
inputs = tokenizer.encode(input_text, return_tensors='pt')
attention_mask = inputs.ne(tokenizer.pad_token_id).long()

# Générer du texte avec le modèle
outputs = model.generate(
    inputs, 
    attention_mask=attention_mask,
    max_length=250, 
    num_return_sequences=1, 
    no_repeat_ngram_size=2, 
    temperature=0.7, 
    top_p=0.9, 
    top_k=50, 
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id  # Définir explicitement le pad_token_id
)

# Décoder le texte généré
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
