#!/usr/bin/env python3
import csv
import sys
import time
import torch
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)

# Configuration
# MODEL_PATH = "flan_t5_finetune/checkpoint-220"
# MODEL_PATH = "flan_t5_finetune_v2/checkpoint-680"
MODEL_PATH = "flan_t5_finetune_v/checkpoint-2200"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_NEW_TOKENS = 256
CSV_PATH = "results.csv"

# Global model variables
tokenizer = None
model = None

SYSTEM_PROMPT = (
    "You are an assistant for a communication simulation tool. Convert user queries into function calls or short answers. "
    "Scope: modulation, error probability/exponent, and performance plots. "
    "Prefix any computation with 'CALL:' followed by the function name and arguments. "
    "Available functions: "
    "computeErrorProbability(modulation, snr, rate, quadrature_nodes, n), "
    "computeErrorExponent(modulation, snr, rate, quadrature_nodes), "
    "plotFromFunction(y, x, min, max, points, typeModulation, M, N, SNR, Rate). "
    "Answer theoretical questions in less than 2 sentences (under 100 words). "
    "If the query is outside scope, reply: I'm sorry, that's outside my domain."
)

# Test cases
PROMPT_DATA = [
    "Calculate the error probability for BPSK at SNR 7 and rate 0.8.",
    "Find the error exponent if I use 8-PSK modulation with SNR 9, rate 0.5, and 5 quadrature nodes.",
    "Plot the error probability versus SNR for 16-QAM, with M=16, N=1000, rate 0.6, and SNR from 0 to 15.",
    "What is modulation in digital communications?",
    "Compute the error exponent for QPSK when the rate is 0.9.",
    "What does increasing block length N do to the probability of error?",
    "Show me the error probability curve for BPSK with M=2, N=500, over SNRs from 0 to 12.",
    "How do I compute the error probability for 4-QAM at SNR 5, rate 0.7, N=200?",
    "What's the difference between error exponent and error probability?",
    "If my system uses QPSK, SNR 10, and rate 0.6, what's the error exponent?",
    "Calculate error probability for 8-QAM at rate 0.5, using 7 quadrature nodes and N=300.",
    "Explain what SNR means in digital communication.",
    "For BPSK, rate 0.8, N=400, and SNR 12, what's the error probability?",
    "Show a plot of error exponent vs rate for 16-QAM, M=16, N=100, over rates 0.2 to 1.0.",
    "What's the purpose of using quadrature nodes in simulations?",
    "What's the error probability for QPSK if SNR is unknown but rate is 0.7 and N=150?",
    "Compute error exponent for 32-QAM at SNR 11, rate 0.65, using 9 quadrature nodes.",
    "How likely is an error with BPSK at SNR 3 and rate 0.5?",
    "Plot error probability for 8-PSK, SNR 0 to 15, M=8, N=200.",
    "What does \"rate\" mean in channel coding?",
    "What's the error exponent for 4-QAM if the SNR is 8 and the rate is unknown?",
    "Find the error probability for 64-QAM, SNR 13, rate 0.75, quadrature nodes 11, N=800.",
    "Why is QPSK preferred over BPSK sometimes?",
    "Can you plot error exponent versus SNR for BPSK with M=2, N=300, SNR from 0 to 10?",
    "Compute error probability for QPSK with N=500.",
    "What's the effect of increasing SNR on error probability?",
    "Find error exponent for 8-PSK, SNR unknown, rate 0.85.",
    "What does \"N\" represent in these calculations?",
    "Calculate the error probability for 16-QAM at SNR 6, rate unknown, using N=1200.",
    "Plot error probability vs SNR for QPSK, M=4, N=400, from SNR 0 to 20.",
    "Compute the error exponent for BPSK, rate 0.6, quadrature nodes 4.",
    "What is \"8-QAM\" and when is it used?",
    "How to find error probability for BPSK, rate 1.0, with 10 quadrature nodes?",
    "What's the error probability of 4-QAM at SNR 9 and N=100?",
    "How does modulation order M affect performance?",
    "Compute error exponent for 64-QAM, rate 0.9, using 12 quadrature nodes.",
    "Plot error exponent for 8-QAM, M=8, N=250, over SNR 0–18.",
    "What is the meaning of \"quadrature nodes\" here?",
    "Calculate error probability for 32-QAM, rate 0.4, with N=50 and SNR 5.",
    "For BPSK, how does SNR 2 compare to SNR 10 in error probability?",
    "Find the error exponent for QPSK with rate 0.85 and 6 quadrature nodes.",
    "What's the error probability for 64-QAM with SNR 12 and rate 0.9?",
    "Why use higher-order QAM?",
    "Plot error probability vs SNR for 32-QAM, M=32, N=2000, SNR from 5 to 20.",
    "Give me the error exponent for 16-QAM, SNR 7, rate unknown, quadrature nodes 8.",
    "What factors affect the error probability in digital transmission?",
    "Calculate error probability for 8-PSK at SNR 15 and rate 0.7.",
    "What does \"error exponent\" tell us in practice?",
    "Compute error exponent for BPSK, SNR 8, rate 0.75, 5 quadrature nodes.",
    "What's the error probability for QPSK at SNR 4, rate 0.6, and N=50?",
]

@dataclass
class TestResult:
    """Represents the results of a single test case."""
    prompt: str
    output: str
    latency_s: float
    words_per_s: float

def extract_and_clean(prompt: str) -> str:
    """
    Process a prompt through the model and clean the response.
    
    Args:
        prompt: The input prompt to process
        
    Returns:
        str: Cleaned model response with function calls split onto separate lines
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(DEVICE)
    
    out_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=5,
        do_sample=True,
        temperature=0.1,
        top_p=0.9
    )
    
    raw = tokenizer.decode(
        out_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    # Split any concatenated calls onto separate lines
    calls = re.findall(r'(?:#\w+\s*)?[A-Za-z_]+\([^)]*\)', raw)
    return "\n".join(calls) if calls else raw.strip()

def is_function_call(response: str) -> str | None:
    """
    Check if a response is a function call and return the function name.
    
    Args:
        response (str): The model's response to check
        
    Returns:
        str | None: Function name if response is a function call, None otherwise
    """
    match = re.match(r"(\w+)\(", response)
    return match.group(1) if match else None

def compare_expected(response: str, expected: str) -> bool:
    """
    Compare a response against the expected output.
    
    Args:
        response (str): The model's response
        expected (str): The expected response
        
    Returns:
        bool: True if response matches expected format, False otherwise
    """
    if expected.startswith("CALL:"):
        # Must be function call and match function name
        call_type = is_function_call(response)
        exp_func = re.match(r"CALL:\s*(\w+)\(", expected)
        return call_type == (exp_func.group(1) if exp_func else None)
    else:
        # Expects a brief technical explanation (not a function call)
        return not response.strip().startswith("CALL:")

def save_results(results: List[TestResult], csv_path: str) -> None:
    """
    Save test results to a CSV file.
    
    Args:
        results: List of test results
        csv_path: Path to save CSV file
    """
    fieldnames = [
        "prompt", "output",
        "latency_s", "words_per_s"
    ]
    
    write_header = not Path(csv_path).exists()
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows([vars(r) for r in results])

def main() -> None:
    """Main execution function that runs the benchmark and saves results."""
    global tokenizer, model
    
    fieldnames = [
        "prompt", "output",
        "latency_s", "words_per_s"
    ]
    results = []
    total_latency, total_words, n = 0.0, 0, 0

    # Load model & tokenizer
    print(f"🖥  Using device: {DEVICE}\n")
    print(f"==> Loading model from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)

    # Print model configuration
    print("\nModel Configuration:")
    print(f"Model type: {model.__class__.__name__}")
    print(f"Model config: {model.config}")
    print(f"Vocab size: {model.config.vocab_size}")
    print(f"Decoder start token: {model.config.decoder_start_token_id}")
    print(f"EOS token: {model.config.eos_token_id}")
    print(f"Pad token: {model.config.pad_token_id}")
    print("\nTokenizer Configuration:")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    print("\n" + "="*50 + "\n")

    # Process each test case
    for i, prompt in enumerate(PROMPT_DATA):
        print(f"Prompt {i+1}")
        prompt_text = f"{SYSTEM_PROMPT}\n\n### Instruction:\n{prompt}\n### Answer:\n"
        
        try:
            t0 = time.perf_counter()
            gen = extract_and_clean(prompt_text)
            secs = time.perf_counter() - t0
            wps = len(gen.split()) / secs if secs else 0.0
            
            print(f"Prompt: {prompt}")
            print(f"LLM Output: {gen}")
            print(f"Latency: {secs:.3f}s | WPS: {wps:.1f}\n")
            
            result = TestResult(
                prompt=prompt,
                output=gen.replace("\n", "\\n"),
                latency_s=secs,
                words_per_s=wps
            )
            results.append(result)
            
            total_latency += secs
            total_words += len(gen.split())
            n += 1
            
        except Exception as exc:
            print(f"⚠️  Error on prompt: {exc}")
            result = TestResult(
                prompt=prompt,
                output=f"ERROR: {exc}",
                latency_s=0.0,
                words_per_s=0.0
            )
            results.append(result)

    # Calculate and print final statistics
    avg_lat = total_latency / n if n else 0
    avg_wps = total_words / total_latency if total_latency else 0
    print(f"Results: avg_latency={avg_lat:.2f}s avg_wps={avg_wps:.1f}\n")

    # Save results to CSV
    if CSV_PATH:
        save_results(results, CSV_PATH)
    print(f"📄 Results saved to {CSV_PATH}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user")