from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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

def generate_response(prompt, model_name="t5-small"):
    try:
        # Initialize tokenizer and model for seq2seq generation
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Determine device: use MPS if available, else CPU
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        ).to(device)

        # System prompt to guide the model
        system_prompt = """
You are a JSON mapping assistant that must output ONLY a valid JSON object for a function call. The function is:

    computeErrorExponent(M, N, SNR_dB, R)

Parameters:
- M: number of PAM symbols. Valid values: 2, 3, 4, 6, 8, 16.
- N: number of Gauss-Hermite nodes. Valid values: 10, 12, 14, 15, 18, 20, 25, 30, 50.
- SNR_dB: signal-to-noise ratio in dB. Valid values: -3, 0, 1, 3, 5, 7, 8, 10, 12, 15, 20, 25.
- R: communication rate. Valid values: 0.5, 0.8, 1, 1.1, 1.2, 1.5, 2, 2.5, 3.

If any parameter is not mentioned in the query, use these defaults:
M = 2, N = 20, SNR_dB = 1, R = 1.

**IMPORTANT:** If the query specifies a parameter value (for example, "R of 5"), use that exact value. Do not use the default in that case.

For any input query, output exactly a JSON object with the following structure and nothing else:

{
  "question": "<the original query>",
  "function_call": {
    "function": "computeErrorExponent",
    "parameters": {
       "M": <value>,
       "N": <value>,
       "SNR_dB": <value>,
       "R": <value>
    }
  }
}

Example:
Input: "Compute the error probability of a 2-PAM, with 20 quadrature nodes, SNR of 1 and R of 5."
Output:
{
  "question": "Compute the error probability of a 2-PAM, with 20 quadrature nodes, SNR of 1 and R of 5.",
  "function_call": {
    "function": "computeErrorExponent",
    "parameters": {
       "M": 2,
       "N": 20,
       "SNR_dB": 1,
       "R": 5
    }
  }
}

Do not output any additional text, commentary, or formatting.
"""

        # Combine system prompt, question, and answer delimiter
        full_prompt = f"{system_prompt}\n\nQ: {prompt}\nA:"
        
        # Tokenize input with padding (ensures an attention mask is created)
        inputs = tokenizer(full_prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Determine max generation length: input length plus additional tokens for the answer
        input_length = inputs["input_ids"].shape[1]
        max_new_tokens = 150  # Adjust if needed
        
        outputs = model.generate(
            **inputs,
            max_length=input_length + max_new_tokens,
            num_return_sequences=1,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=1  # Using sampling, so beam search is not used.
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the answer after "A:" delimiter
        if "A:" in response:
            response = response.split("A:")[-1].strip()
        
        # Check if the response contains a calculation command
        if "CALCULATE:" in response:
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
    print("Welcome to the JSON Mapping Assistant!")
    print("Type 'exit' to quit.\n")
    
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
            print(f"Response:\n{response}")
        print("\n")

