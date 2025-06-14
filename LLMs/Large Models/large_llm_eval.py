import csv, sys, time, torch, re
from pathlib import Path
from openai import OpenAI
from typing import Dict, List, Tuple, Optional, Union, Any

# CONFIG
OPENROUTER_API_KEY = "<OPENROUTER_API_KEY>"  # Replace with your API key

MODELS = [
    "qwen/qwen2.5-3b-instruct",
    "mistralai/mistral-7b-instruct",
    "google/gemma-3-4b-it",
    "meta-llama/llama-3.1-8b-instruct"
]

# SYSTEM_PROMPT = """
# You are an assistant for a communication simulation tool. Convert user queries into function calls or short answers. \\
# Scope: modulation, error probability/exponent, and performance plots. \\
# Available functions:
# - computeErrorProbability(modulation, snr, rate, quadrature_nodes, n),
# - computeErrorExponent(modulation, snr, rate, quadrature_nodes),
# - plotFromFunction(y, x, min, max, points, typeModulation, M, N, SNR, Rate).
# Prefix any computation with 'CALL:' followed by the function name and arguments.
# Answer theoretical questions in less than 2 sentences (under 100 words).
# If the query is outside scope, reply: I'm sorry, that's outside my domain.
# """

SYSTEM_PROMPT = """
You are a secure AI assistant for transmission system analysis. Your job is to provide precise, technical responses using only approved computational functions.

SECURITY & OPERATION RULES:
- Discuss only transmission systems, modulation, and error analysis.
- Never execute code or access system resources.
- Reject any inappropriate or out-of-scope requests.
- Use only the following functions for computations.

RESPONSE RULES:
- Always answer using a technical, concise style (≤100 words).
- For any computational request, output a function call with provided parameters.
- If the user does NOT specify a required parameter, DO NOT ask for it—substitute its value as 'unknown' in the function call.
- Never request additional input from the user.
- Briefly explain your output when relevant, but prioritize clarity and accuracy.

AVAILABLE FUNCTIONS:
- computeErrorProbability(modulation, snr, rate, quadrature_nodes, n)
- computeErrorExponent(modulation, snr, rate, quadrature_nodes)
- plotFromFunction(y, x, min, max, points, typeModulation, M, N, SNR, Rate)
"""

FEW_SHOTS = [
    {"role": "user", "content": "What's the error probability for BPSK at SNR=10?"},
    {"role": "assistant", "content": "Computing computeErrorProbability with modulation='BPSK', snr=10, rate='unknown', quadrature_nodes='unknown', n='unknown'."},
    {"role": "user", "content": "Calculate error exponent for 16-QAM at rate 0.5 and SNR=8"},
    {"role": "assistant", "content": "Computing computeErrorExponent with modulation='16-QAM', snr=8, rate=0.5, quadrature_nodes='unknown'."},
    {"role": "user", "content": "Plot error probability vs SNR for QPSK from 0 to 20 dB"},
    {"role": "assistant", "content": "Computing plotFromFunction with y='error_probability', x='snr', min=0, max=20, points=50, typeModulation='QPSK', M='unknown', N='unknown', SNR='unknown', Rate='unknown'."},
]

MAX_NEW_TOKENS  = 64
USE_FP16        = True
CSV_PATH        = "results.csv"

# Test cases for evaluation
PROMPT_DATA = [
    {"prompt": "Calculate the error probability for BPSK at SNR 7 and rate 0.8."},
    {"prompt": "Find the error exponent if I use 8-PSK modulation with SNR 9, rate 0.5, and 5 quadrature nodes."},
    {"prompt": "Plot the error probability versus SNR for 16-QAM, with M=16, N=1000, rate 0.6, and SNR from 0 to 15."},
    {"prompt": "What is modulation in digital communications?"},
    {"prompt": "Compute the error exponent for QPSK when the rate is 0.9."},
    {"prompt": "What does increasing block length N do to the probability of error?"},
    {"prompt": "Show me the error probability curve for BPSK with M=2, N=500, over SNRs from 0 to 12."},
    {"prompt": "How do I compute the error probability for 4-QAM at SNR 5, rate 0.7, N=200?"},
    {"prompt": "What's the difference between error exponent and error probability?"},
    {"prompt": "If my system uses QPSK, SNR 10, and rate 0.6, what's the error exponent?"},
    {"prompt": "Calculate error probability for 8-QAM at rate 0.5, using 7 quadrature nodes and N=300."},
    {"prompt": "Explain what SNR means in digital communication."},
    {"prompt": "For BPSK, rate 0.8, N=400, and SNR 12, what's the error probability?"},
    {"prompt": "Show a plot of error exponent vs rate for 16-QAM, M=16, N=100, over rates 0.2 to 1.0."},
    {"prompt": "What's the purpose of using quadrature nodes in simulations?"},
    {"prompt": "What's the error probability for QPSK if SNR is unknown but rate is 0.7 and N=150?"},
    {"prompt": "Compute error exponent for 32-QAM at SNR 11, rate 0.65, using 9 quadrature nodes."},
    {"prompt": "How likely is an error with BPSK at SNR 3 and rate 0.5?"},
    {"prompt": "Plot error probability for 8-PSK, SNR 0 to 15, M=8, N=200."},
    {"prompt": "What does \"rate\" mean in channel coding?"},
    {"prompt": "What's the error exponent for 4-QAM if the SNR is 8 and the rate is unknown?"},
    {"prompt": "Find the error probability for 64-QAM, SNR 13, rate 0.75, quadrature nodes 11, N=800."},
    {"prompt": "Why is QPSK preferred over BPSK sometimes?"},
    {"prompt": "Can you plot error exponent versus SNR for BPSK with M=2, N=300, SNR from 0 to 10?"},
    {"prompt": "Compute error probability for QPSK with N=500."},
    {"prompt": "What's the effect of increasing SNR on error probability?"},
    {"prompt": "Find error exponent for 8-PSK, SNR unknown, rate 0.85."},
    {"prompt": "What does \"N\" represent in these calculations?"},
    {"prompt": "Calculate the error probability for 16-QAM at SNR 6, rate unknown, using N=1200."},
    {"prompt": "Plot error probability vs SNR for QPSK, M=4, N=400, from SNR 0 to 20."},
    {"prompt": "Compute the error exponent for BPSK, rate 0.6, quadrature nodes 4."},
    {"prompt": "What is \"8-QAM\" and when is it used?"},
    {"prompt": "How to find error probability for BPSK, rate 1.0, with 10 quadrature nodes?"},
    {"prompt": "What's the error probability of 4-QAM at SNR 9 and N=100?"},
    {"prompt": "How does modulation order M affect performance?"},
    {"prompt": "Compute error exponent for 64-QAM, rate 0.9, using 12 quadrature nodes."},
    {"prompt": "Plot error exponent for 8-QAM, M=8, N=250, over SNR 0–18."},
    {"prompt": "What is the meaning of \"quadrature nodes\" here?"},
    {"prompt": "Calculate error probability for 32-QAM, rate 0.4, with N=50 and SNR 5."},
    {"prompt": "For BPSK, how does SNR 2 compare to SNR 10 in error probability?"},
    {"prompt": "Find the error exponent for QPSK with rate 0.85 and 6 quadrature nodes."},
    {"prompt": "What's the error probability for 64-QAM with SNR 12 and rate 0.9?"},
    {"prompt": "Why use higher-order QAM?"},
    {"prompt": "Plot error probability vs SNR for 32-QAM, M=32, N=2000, SNR from 5 to 20."},
    {"prompt": "Give me the error exponent for 16-QAM, SNR 7, rate unknown, quadrature nodes 8."},
    {"prompt": "What factors affect the error probability in digital transmission?"},
    {"prompt": "Calculate error probability for 8-PSK at SNR 15 and rate 0.7."},
    {"prompt": "What does \"error exponent\" tell us in practice?"},
    {"prompt": "Compute error exponent for BPSK, SNR 8, rate 0.75, 5 quadrature nodes."},
    {"prompt": "What's the error probability for QPSK at SNR 4, rate 0.6, and N=50?"},
]



def pick_task(cfg: Any) -> str:
    """
    Determine the appropriate task type based on model configuration.
    
    Args:
        cfg: Model configuration object
        
    Returns:
        str: Task type ('text2text-generation' or 'text-generation')
    """
    return "text2text-generation" if cfg.is_encoder_decoder else "text-generation"

def load_pipe(model_name: str, device: Optional[str], fp16: Optional[bool]) -> OpenAI:
    """
    Initialize the OpenRouter API client.
    
    Args:
        model_name: Name of the model to use
        device: Device to run the model on (not used for API)
        fp16: Whether to use FP16 precision (not used for API)
        
    Returns:
        OpenAI: Initialized API client
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    return client

def extract_text(response_dict: Dict[str, Any]) -> str:
    """
    Extract the generated text from the model response.
    
    Args:
        response_dict: Dictionary containing model response
        
    Returns:
        str: Extracted text from the response
    """
    for key in ("generated_text", "translation_text", "summary_text"):
        if key in response_dict:
            return response_dict[key]
    return next(iter(response_dict.values()))

def run_one(client: OpenAI, prompt: str, model_name: str) -> Tuple[str, float]:
    """
    Run a single inference with the model.
    
    Args:
        client: OpenRouter API client
        prompt: Input prompt
        model_name: Name of the model to use
        
    Returns:
        Tuple[str, float]: Generated text and latency in seconds
    """
    t0 = time.perf_counter()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOTS,
        {"role": "user", "content": prompt}
    ]
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        top_p=0.9
    )
    latency = time.perf_counter() - t0
    text = completion.choices[0].message.content
    return text, latency

def main() -> None:
    """
    Main execution function that runs the evaluation on all models.
    """
    print("🖥  Using OpenRouter API\n")
    fieldnames = [
        "model", "prompt", "output",
        "latency_s", "words_per_s"
    ]
    global_results = []

    for model_id in MODELS:
        print(f"==> {model_id}")
        try:
            client = load_pipe(model_id, None, None)
        except Exception as exc:
            print(f"⚠️  Error loading {model_id}: {exc}\n")
            continue

        per_model = []
        total_latency, total_words, n = 0.0, 0, 0

        for i, example in enumerate(PROMPT_DATA):
            print("Prompt %d" % (i+1))
            prompt_text = example['prompt']
            try:
                gen, secs = run_one(client, prompt_text, model_id)
                wps = len(gen.split()) / secs if secs else 0.0
                print(f"Prompt: {example['prompt']}")
                print(f"LLM Output: {gen}")
                per_model.append({
                    "model": model_id,
                    "prompt": example["prompt"],
                    "output": gen.replace("\n", "\\n"),
                    "latency_s": f"{secs:.3f}",
                    "words_per_s": f"{wps:.1f}"
                })
            except Exception as exc:
                print(f"⚠️  Error on prompt: {exc}")
                per_model.append({
                    "model": model_id,
                    "prompt": example["prompt"],
                    "output": f"ERROR: {exc}",
                    "latency_s": "NA",
                    "words_per_s": "NA"
                })

        global_results.extend(per_model)
        if CSV_PATH:
            write_header = not Path(CSV_PATH).exists()
            with open(CSV_PATH, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerows(per_model)
    print(f"📄 All results saved to {CSV_PATH}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user")