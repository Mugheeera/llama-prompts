from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_response(prompt):
    model_name = /user/mugheera/llama/llama-2-7b/

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Encode the prompt and generate response
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=50, num_beams=5, no_repeat_ngram_size=2)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

if __name__ == "__main__":
    # Input your prompt
    user_prompt = input("Enter your prompt: ")
    print("Generating response...")
    print(generate_response(user_prompt))
