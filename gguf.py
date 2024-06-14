import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from datetime import datetime

# Chemin de cache C:\Users\Your_Username\.cache
os.environ["HF_HOME"] = "/cache"

# Télécharger le fichier modèle
model_name = "TheBloke/Mistral-7B-v0.1-GGUF" # Lien du repo du model avec le nom d'utilisateur
model_file = "mistral-7b-v0.1.Q5_K_M.gguf"
model_path = hf_hub_download(repo_id=model_name, filename=model_file)

# Instancier le modèle à partir du fichier téléchargé
llm = Llama(
    model_path=model_path,
    n_ctx=8000,  # Longueur du contexte
    n_threads=11, 
    n_gpu_layers=4  # Nombre de couches de modèle à décharger sur le GPU
)

# Contexte par défaut contenant le nom de l'utilisateur
default_context = (
    "[CONTEXT] Hi you are an Artificial Intelligence, and try to chat with me if you want to.\n"
    "My name is Dennis, we are in a chatbox through VisualStudioCode terminal, i am here for chat and ask question. "
    "The text below is our last conversation history, including timestamps to indicate when each message was sent. Use it as a memory[CONTEXT]\n"
    "Dennis=[User]\n"
    "Artificial Intelligence=[You]\n"
)

# Paramètres de génération
generation_kwargs = {
    "max_tokens": 5000,  # Limiter le nombre de tokens générés
    "stop": ["</s>"],
    "echo": False, 
    "top_k": 1  # Décodage greedy
}

# Fonction pour lire l'historique des conversations
def read_conversation_history():
    if os.path.exists("conversation_history.txt"):
        with open("conversation_history.txt", "r") as file:
            return file.read()
    return ""

# Fonction pour envoyer un message avec le contexte par défaut et l'historique des conversations du fichier txt
def with_context(input):
    conversation_history = read_conversation_history()
    prompt = f"{default_context}\n{conversation_history}\n[User]: {input}\n\n[You]:"
    print(prompt)  # Afficher le contenu complet du prompt
    res = llm(prompt, **generation_kwargs)
    return res['choices'][0]['text'].strip()

# Fonction pour obtenir l'horodatage actuel
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Fonction pour sauvegarder l'historique des conversations avec horodatages
def save_conversation(user_input, model_response):
    timestamp = get_timestamp()
    with open("conversation_history.txt", "a") as file:
        file.write(f"[{timestamp}] [User]: {user_input}\n")
        file.write(f"[{timestamp}] [You]: {model_response}\n")

# Boucle de chat interactive
print("Discutez avec le modèle. Tapez 'exit' pour terminer la discussion.")
while True:
    user_input = input("Vous: ")
    if user_input.lower() == 'exit':
        print("Fin de la discussion.")
        break

    # Obtenir la réponse du modèle avec le contexte
    answer = with_context(user_input)
    print(f"Modèle: {answer}")

    # Sauvegarder la conversation dans un fichier avec horodatages
    save_conversation(user_input, answer)
