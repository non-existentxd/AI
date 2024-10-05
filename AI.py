


import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
def load_model(model_name=":)"):
    print("Loading model...")
    try:
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# Generate text
def generate_text(model, tokenizer, prompt, max_length=100, top_k=50, top_p=0.95, num_return_sequences=1):
    try:
        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt")

        # Check if GPU is available and use it
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Generate text with the model
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,  # Prevent repetition
            do_sample=True,  # Enable randomness in generation
            top_k=top_k,  # Top-k sampling for diversity
            top_p=top_p  # Nucleus sampling for more randomness
        )

        # Decode the generated text
        generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return generated_texts

    except Exception as e:
        print(f"Error generating text: {e}")
        return []

# Main function to run the generator
def run_generator():
    #load the :) model
    model, tokenizer = load_model()
    if model is None or tokenizer is None:
        return

    # Input prompt from the user
    prompt = input("Enter a prompt for text generation: ")
    if not prompt:
        print("No prompt entered. Exiting...")
        return

    # Custom parameters for text generation
    try:
        max_length = int(input("Enter max length of generated text (default 100): ") or 100)
        top_k = int(input("Enter top-k sampling value (default 50): ") or 50)
        top_p = float(input("Enter top-p sampling value (default 0.95): ") or 0.95)
        num_return_sequences = int(input("Enter number of sequences to generate (default 1): ") or 1)
    except ValueError:
        print("Invalid input for parameters. Using default values.")
        max_length = 100
        top_k = 50
        top_p = 0.95
        num_return_sequences = 1

    # Generate and print the results
    generated_texts = generate_text(model, tokenizer, prompt, max_length, top_k, top_p, num_return_sequences)
    if generated_texts:
        print("\nGenerated Text:")
        for i, text in enumerate(generated_texts):
            print(f"\n--- Output {i+1} ---\n{text}")
    else:
        #print out the try again ai model cant't respond to  quesrtion
        print("No text generated. Please try again.")

# Run the generator
if __name__ == "__main__":
    run_generator()
