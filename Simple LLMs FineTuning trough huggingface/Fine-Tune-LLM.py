from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import json

# Charger les données du fichier JSON
with open('conversations.json', 'r', encoding='utf-8') as f: # adapt it
    data = json.load(f)

# Préparer les données pour le fine-tuning
conversations = []
for entry in data:
    if entry['role'] and entry['text']:  # Vérifier que les champs ne sont pas vides
        conversations.append(f"{entry['role']}: {entry['text']}\n")

print(f"Total conversations: {len(conversations)}")  # Vérifier le nombre total de conversations

# Créer un dataset compatible avec Hugging Face
dataset = Dataset.from_dict({"text": conversations})

# Charger le tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")  # adapt it
tokenizer.pad_token = tokenizer.eos_token

# Ajouter le `pad_token` au modèle
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")  # adapt it
model.resize_token_embeddings(len(tokenizer))

# Tokeniser le dataset
def tokenize_function(examples):
    tokens = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    tokens['labels'] = tokens['input_ids'].copy()
    return tokens

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Configurer les arguments d'entraînement
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=10, # adapt it
    per_device_eval_batch_size=10, # adapt it
    num_train_epochs=10, # adapt it
    weight_decay=0.01,
    fp16=True,
)

# Créer un Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets
)

# Fine-tuner le modèle
trainer.train()

# Sauvegarder le modèle fine-tuné
output_dir = './fine_tuned_model' # Sera crée dans le meme dossier
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")
