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
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
    Pipeline
)

# Configuration Constants
MODELS = [
    "google/flan-t5-small",
    "google/flan-t5-base",
    "t5-small",
    "distilgpt2",
]

SYSTEM_PROMPT = (
    "You are an assistant for a communication simulation tool. Convert user queries into function calls or short answers. "
    "Scope: modulation, error probability/exponent, and performance plots. "
    "Available functions: "
    "computeErrorProbability(modulation, snr, rate, quadrature_nodes, n), "
    "computeErrorExponent(modulation, snr, rate, quadrature_nodes), "
    "plotFromFunction(y, x, min, max, points, typeModulation, M, N, SNR, Rate). "
    "Prefix any computation with 'CALL:' followed by the function name and arguments. "
    "Answer theoretical questions in less than 2 sentences (under 100 words). "
    "If the query is outside scope, reply: I'm sorry, that's outside my domain."
)

MAX_NEW_TOKENS = 64
USE_FP16 = True
CSV_PATH = "results.csv"

# Test Cases
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
    model: str
    prompt: str
    output: str
    latency_s: float
    words_per_s: float

def pick_task(cfg: AutoConfig) -> str:
    """
    Determine the appropriate task type based on model configuration.
    
    Args:
        cfg: Model configuration object
        
    Returns:
        str: Task type ("text2text-generation" or "text-generation")
    """
    return "text2text-generation" if cfg.is_encoder_decoder else "text-generation"

def load_pipe(model_name: str, device: torch.device, fp16: bool) -> Pipeline:
    """
    Load and configure the model pipeline.
    
    Args:
        model_name: Name or path of the model to load
        device: Torch device to use
        fp16: Whether to use half-precision
        
    Returns:
        Pipeline: Configured HuggingFace pipeline
    """
    cfg = AutoConfig.from_pretrained(model_name)
    task = pick_task(cfg)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    model_cls = AutoModelForSeq2SeqLM if task == "text2text-generation" else AutoModelForCausalLM
    model = model_cls.from_pretrained(
        model_name,
        torch_dtype=(torch.float16 if fp16 and device.type == "cuda" else None),
    ).to(device)
    
    return pipeline(
        task,
        model=model,
        tokenizer=tok,
        device=device.index if device.type == "cuda" else -1
    )

def extract_text(response_dict: Dict) -> str:
    """
    Extract the generated text from the model's response dictionary.
    
    Args:
        response_dict: Model output dictionary
        
    Returns:
        str: Extracted text
    """
    for key in ("generated_text", "translation_text", "summary_text"):
        if key in response_dict:
            return response_dict[key]
    return next(iter(response_dict.values()))

def run_one(pipe: Pipeline, prompt: str) -> Tuple[str, float]:
    """
    Run a single inference test.
    
    Args:
        pipe: Model pipeline
        prompt: Input prompt
        
    Returns:
        Tuple[str, float]: Generated text and latency in seconds
    """
    t0 = time.perf_counter()
    out = pipe(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.1,
        top_p=0.9
    )
    latency = time.perf_counter() - t0
    text = extract_text(out[0])
    return text, latency

def save_results(results: List[TestResult], csv_path: str) -> None:
    """
    Save test results to a CSV file.
    
    Args:
        results: List of test results
        csv_path: Path to save CSV file
    """
    fieldnames = [
        "model", "prompt", "output",
        "latency_s", "words_per_s"
    ]
    
    write_header = not Path(csv_path).exists()
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows([vars(r) for r in results])

def main() -> None:
    """Main execution function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Using device: {device}\n")
    global_results = []

    for model_id in MODELS:
        print(f"==> {model_id}")
        try:
            pipe = load_pipe(model_id, device, USE_FP16)
        except Exception as exc:
            print(f"⚠️  Error loading {model_id}: {exc}\n")
            continue

        per_model = []

        for i, prompt in enumerate(PROMPT_DATA):
            print(f"Prompt {i+1}")
            prompt_text = f"{SYSTEM_PROMPT}\n\n### Instruction:\n{prompt}\n### Answer:\n"
            
            try:
                gen, secs = run_one(pipe, prompt_text)
                wps = len(gen.split()) / secs if secs else 0.0
                
                print(f"Prompt: {prompt}")
                print(f"LLM Output: {gen}")
                
                result = TestResult(
                    model=model_id,
                    prompt=prompt,
                    output=gen.replace("\n", "\\n"),
                    latency_s=secs,
                    words_per_s=wps
                )
                per_model.append(result)
                
            except Exception as exc:
                print(f"⚠️  Error on prompt: {exc}")
                result = TestResult(
                    model=model_id,
                    prompt=prompt,
                    output=f"ERROR: {exc}",
                    latency_s=0.0,
                    words_per_s=0.0
                )
                per_model.append(result)

        global_results.extend(per_model)
        if CSV_PATH:
            save_results(per_model, CSV_PATH)
            
    print(f"📄 All results saved to {CSV_PATH}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user")