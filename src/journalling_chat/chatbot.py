from huggingface_hub import InferenceClient

# Replace with your actual Hugging Face API key
api_key = "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"

# Initialize the Inference Client
client = InferenceClient(api_key=api_key)

# Initialize chat history
messages = [{"role": "user", "content": "I will journal my thoughts. You will ask me some insightful questions that will help me journal my thoughts."}]
messages.append({"role": "assistant", "content": "Sure! Tell me about your day."})

while True:
    # Get user input
    user_input = input("You: ")
    
    # Exit the chat if the user types 'exit'
    if user_input.lower() == 'exit':
        print("Chat ended.")
        break
    
    # Append user message to messages
    messages.append({"role": "user", "content": user_input})

    # Call the chat completion API
    response = client.chat_completion(
        model="mistralai/Mistral-Nemo-Instruct-2407",
        messages=messages,
        max_tokens=150,
        stream=False  # Set to True for streaming responses if desired
    )

    # Get assistant's response and print it
    assistant_response = response.choices[0].message.content
    print(f"Assistant: {assistant_response}")

    # Append assistant's message to messages for context in future interactions
    messages.append({"role": "assistant", "content": assistant_response})
