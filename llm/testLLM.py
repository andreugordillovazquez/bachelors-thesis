from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

def sum_numbers(a, b):
    return a + b

def multiply_numbers(a, b):
    return a * b

def extract_numbers(text):
    # Extract numbers from text using regex
    numbers = re.findall(r'-?\d+', text)
    return [int(num) for num in numbers]

def generate_response(prompt, model_name="gpt2-large"):
    try:
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Determine device
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        ).to(device)

        # Add system prompt to guide the model
        system_prompt = """You are a calculator that can perform addition and multiplication.
Rules:
1. For addition questions, respond with CALCULATE: sum(a, b)
2. For multiplication questions, respond with CALCULATE: multiply(a, b)
3. Use the exact numbers from the question
4. For any other type of question, respond with: "I can only help with addition and multiplication calculations. Please ask me to add or multiply two numbers."
5. No other text allowed

Examples:
Q: What is 2+2?
A: CALCULATE: sum(2, 2)

Q: Add 5 and 3
A: CALCULATE: sum(5, 3)

Q: What is 10 plus 20?
A: CALCULATE: sum(10, 20)

Q: What is 2 times 3?
A: CALCULATE: multiply(2, 3)

Q: Multiply 5 by 3
A: CALCULATE: multiply(5, 3)

Q: What is 10 multiplied by 20?
A: CALCULATE: multiply(10, 20)

Q: What's the capital of France?
A: I can only help with addition and multiplication calculations. Please ask me to add or multiply two numbers.

Q: Tell me a joke
A: I can only help with addition and multiplication calculations. Please ask me to add or multiply two numbers."""

        full_prompt = f"{system_prompt}\n\nQ: {prompt}\nA:"
        
        # Generate response
        inputs = tokenizer(full_prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get the length of the input
        input_length = inputs["input_ids"].shape[1]
        
        outputs = model.generate(
            **inputs,
            max_length=input_length + 20,
            num_return_sequences=1,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=1,
            early_stopping=False
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the actual response part (after "A:")
        response = response.split("A:")[-1].strip()
        
        # Check if the response contains a calculation request
        if "CALCULATE:" in response:
            # Extract the function call
            sum_match = re.search(r'CALCULATE:\s*sum\((\d+),\s*(\d+)\)', response)
            multiply_match = re.search(r'CALCULATE:\s*multiply\((\d+),\s*(\d+)\)', response)
            
            if sum_match:
                a, b = map(int, sum_match.groups())
                result = sum_numbers(a, b)
                return f"The sum of {a} and {b} is {result}"
            elif multiply_match:
                a, b = map(int, multiply_match.groups())
                result = multiply_numbers(a, b)
                return f"The product of {a} and {b} is {result}"
        
        return response
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    print("Welcome to the Calculator! Type 'exit' to quit.")
    print("You can ask me to add or multiply numbers.")
    print("Examples:")
    print("- What is 2+2?")
    print("- Multiply 5 by 3")
    print("- What is 10 plus 20?")
    print("- What is 2 times 3?")
    print("\n")
    
    while True:
        prompt = input("Enter your question: ").strip()
        
        if prompt.lower() == "exit":
            print("Goodbye!")
            break
            
        if not prompt:
            print("Please enter a question.")
            continue
            
        response = generate_response(prompt)
        if response:
            print(f"Response: {response}")
        print("\n")

### Examples:
### What is 2+2?
### Multiply 5 by 3
### What is 10 plus 20?
### What is 2 times 3?