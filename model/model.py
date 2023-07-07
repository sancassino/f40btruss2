import spacy
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Initialize tokenizer and model for chatbot
tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/falcon-40b-sft-mix-1226", padding_side='right', return_attention_mask=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("OpenAssistant/falcon-40b-sft-mix-1226")

# Define function to generate a question based on user input
def generate_question(user_input):
    # Parse the user input with spaCy
    doc = nlp(user_input)

    # Find the most relevant noun phrase
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    most_relevant_noun_phrase = max(noun_phrases, key=len)

    # Define a list of question templates
    if nlp.lang == "es":
        question_templates = [
            "¿Puedes contarme más sobre {}?",
            "Cómo te sientes acerca de {}?",
            "¿Qué hizo que te interesaras por {}?",
            "¿Cuál crees que es el aspecto más importante de {}?",
            "¿Alguna vez has tenido alguna experiencia con {}?"
        ]
    else:
        question_templates = [
            "Can you tell me more about {}?",
            "How do you feel about {}?",
            "What made you interested in {}?",
            "What do you think is the most important aspect of {}?",
            "Have you ever had any experience with {}?"
        ]

    # Choose a random question template from the list
    question_template = random.choice(question_templates)

    # Format the question template with the most relevant noun phrase
    question = question_template.format(most_relevant_noun_phrase)

    # Return the question
    return question


# Initialize variables to keep track of name and location
name = None
location = None

# Ask user if they want to learn Spanish or English
print("Do you want to learn Spanish or English? Quieres aprender español o inglés?")
choice = input("Enter 1 for Spanish, 2 for English: ")

# Load the spaCy Spanish model
if choice == "1":
    nlp = spacy.load("es_core_news_sm")
    print("Soy aispeakbot y te enseñaré a hablar español. Comenzaré preguntando algo en español. ¿Cómo te llamas?")
elif choice == "2":
    nlp = spacy.load("en_core_web_sm")
    print("Soy aispeakbot y te enseñaré a hablar inglés. Comenzaré preguntando algo en inglés. What is your name?")
else:
    print("Invalid choice. Please enter 1 or 2.")


# Initialize flag variable to indicate if pre-made follow-up questions have been printed
follow_up_printed = False

# Initialize flag variable to indicate if pre-made follow-up questions have been printed
follow_up_printed = False

# Start conversation
while True:
    # Wait for user input
    user_input = input("You: ")

    # If pre-made follow-up questions have not been printed yet, print them and set flag to True
    if not follow_up_printed:
        if "My name is" in user_input:
            name = user_input.split("My name is")[-1].strip()
            print(f"aispeakbot: Nice to meet you, {name}. Where do you live?")
            follow_up_printed = True
        elif "Me llamo" in user_input:
            name = user_input.split("Me llamo")[-1].strip()
            print(f"aispeakbot: Encantado de conocerte, {name}. Dónde vives?")
            follow_up_printed = True
        elif "vivo en" in user_input:
            location = user_input.split("vivo en")[-1].strip()
            print(f"aispeakbot: Ah, estás en {location}. ¿A qué te dedicas? ¿Tienes hermanos o hermanas?")
            follow_up_printed = True
    # If pre-made follow-up questions have been printed, generate a new question
    else:
        # Generate chatbot response
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
        chatbot_output = model.generate(input_ids, max_length=300, pad_token_id=tokenizer.eos_token_id)
        chatbot_response = tokenizer.decode(chatbot_output[0], skip_special_tokens=True)
       
        # Print the chatbot's response without the user's input
        print("aispeakbot:", chatbot_response.replace(user_input, "").strip())

        # Generate a question based on the chatbot response
        question = generate_question(chatbot_response)

        # Print the question
        print("aispeakbot:", question)
