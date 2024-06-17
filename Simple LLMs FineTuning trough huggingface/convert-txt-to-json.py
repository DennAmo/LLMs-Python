import json
from deep_translator import GoogleTranslator
import time

def batch_translate(texts, batch_size=10):
    translator = GoogleTranslator(source='fr', target='en')
    translated_texts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"Translating batch {i//batch_size + 1} of {len(texts)//batch_size + 1}...")
        try:
            translated_batch = [translator.translate(text) for text in batch]
            translated_texts.extend(translated_batch)
        except Exception as e:
            print(f"Error translating batch {i}-{i+batch_size}: {e}")
            translated_texts.extend(batch)  # Keep the original texts in case of error
        time.sleep(1)  # Sleep to avoid hitting API rate limits
    return translated_texts

print("Reading the file...")

# Lire le fichier texte
file_path = 'Filtered_Important_Conversations_Herika_Amo.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

print(f"Read {len(lines)} lines from the file.")

# Initialiser les variables pour le JSON
conversations = []
current_conversation = {"Amo": "", "Herika": ""}
current_speaker = None

# Parcourir les lignes du fichier
print("Processing lines...")
for line in lines:
    if line.strip():  # Si la ligne n'est pas vide
        if line.startswith("Amo:"):
            if current_speaker == "Herika":
                conversations.append(current_conversation)
                current_conversation = {"Amo": "", "Herika": ""}
            current_speaker = "Amo"
            current_conversation["Amo"] += line.replace("Amo:", "").strip() + " "
        elif line.startswith("Herika:"):
            if current_speaker == "Amo":
                conversations.append(current_conversation)
                current_conversation = {"Amo": "", "Herika": ""}
            current_speaker = "Herika"
            current_conversation["Herika"] += line.replace("Herika:", "").strip() + " "

# Ajouter la dernière conversation
if current_conversation not in conversations:
    conversations.append(current_conversation)

print(f"Processed {len(conversations)} conversations.")

# Préparer les textes à traduire
amo_texts = [convo["Amo"] for convo in conversations]
herika_texts = [convo["Herika"] for convo in conversations]

# Traduction des parties pertinentes en anglais par lot
print("Translating Amo's conversations...")
translated_amo_texts = batch_translate(amo_texts)
print("Translating Herika's conversations...")
translated_herika_texts = batch_translate(herika_texts)

# Mettre à jour les conversations avec les textes traduits
for i, convo in enumerate(conversations):
    convo["Amo"] = translated_amo_texts[i]
    convo["Herika"] = translated_herika_texts[i]

print("Translation completed.")

# Enregistrer les conversations en JSON
output_path = 'conversations.json'
with open(output_path, 'w', encoding='utf-8') as output_file:
    json.dump(conversations, output_file, ensure_ascii=False, indent=4)

print(f"Les conversations ont été converties et enregistrées dans {output_path}.")
