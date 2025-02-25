#!/usr/bin/env python3
"""
Ollama Model Benchmark Tool with CSV Logging

Features:
- Measures LLM performance metrics via Ollama
- Saves results to 'benchmark_results.csv'
- Handles multiple prompts and models

Usage:
    python benchmark.py [-v] [-m MODEL_NAMES...] [-p PROMPTS...]
"""

import argparse
import csv
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
import ollama

# CSV File Path
CSV_FILENAME = "./benchmark_results.csv"


class Message(BaseModel):
    role: str
    content: str


class OllamaResponse(BaseModel):
    model: str
    created_at: Optional[datetime] = None
    message: Message
    done: bool
    total_duration: int = Field(default=0)
    load_duration: int = Field(default=0)
    prompt_eval_count: int = Field(default=0)
    prompt_eval_duration: int = Field(default=0)
    eval_count: int = Field(default=0)
    eval_duration: int = Field(default=0)

    @classmethod
    def from_chat_response(cls, response) -> "OllamaResponse":
        return cls(
            model=response.model,
            message=Message(
                role=response.message.role,
                content=response.message.content
            ),
            done=response.done,
            total_duration=getattr(response, 'total_duration', 0),
            load_duration=getattr(response, 'load_duration', 0),
            prompt_eval_count=getattr(response, 'prompt_eval_count', 0),
            prompt_eval_duration=getattr(response, 'prompt_eval_duration', 0),
            eval_count=getattr(response, 'eval_count', 0),
            eval_duration=getattr(response, 'eval_duration', 0)
        )


def run_benchmark(model_name: str, prompt: str, verbose: bool) -> Optional[OllamaResponse]:
    messages = [{"role": "user", "content": prompt}]
    try:
        response = ollama.chat(model=model_name, messages=messages)
        if not response.message.content.strip():
            print(f"Error: {model_name} returned an empty response.")
            return None
        return OllamaResponse.from_chat_response(response)
    except Exception as e:
        print(f"Error running benchmark for {model_name}: {e}")
        return None


def save_to_csv(results: List[OllamaResponse]):
    fieldnames = [
        "Model", "Prompt Processing (tokens/sec)", "Generation Speed (tokens/sec)",
        "Combined Speed (tokens/sec)", "Input Tokens", "Generated Tokens",
        "Model Load Time (s)", "Processing Time (s)", "Generation Time (s)", "Total Time (s)"
    ]
    
    with open(CSV_FILENAME, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if file.tell() == 0:
            writer.writeheader()
        
        for res in results:
            writer.writerow({
                "Model": res.model,
                "Prompt Processing (tokens/sec)": res.prompt_eval_count / (res.prompt_eval_duration / 1_000_000_000) if res.prompt_eval_duration else 0,
                "Generation Speed (tokens/sec)": res.eval_count / (res.eval_duration / 1_000_000_000) if res.eval_duration else 0,
                "Combined Speed (tokens/sec)": (res.prompt_eval_count + res.eval_count) / (res.total_duration / 1_000_000_000) if res.total_duration else 0,
                "Input Tokens": res.prompt_eval_count,
                "Generated Tokens": res.eval_count,
                "Model Load Time (s)": res.load_duration / 1_000_000_000,
                "Processing Time (s)": res.prompt_eval_duration / 1_000_000_000,
                "Generation Time (s)": res.eval_duration / 1_000_000_000,
                "Total Time (s)": res.total_duration / 1_000_000_000,
            })
    print(f"Benchmark results saved to {CSV_FILENAME}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-m", "--models", nargs="*", default=[], help="Specify models to test")
    parser.add_argument("-p", "--prompts", nargs="*", default=[
            "Explain the process of photosynthesis in plants, including the key chemical reactions and energy transformations involved.",
            "Write a detailed story about a time traveler who visits three different historical periods. Include specific details about each era and the protagonist's interactions.",
            "Analyze the potential impact of artificial intelligence on global employment over the next decade. Consider various industries, economic factors, and potential mitigation strategies. Provide specific examples and data-driven reasoning.",
            "Write a Python function that implements a binary search tree with methods for insertion, deletion, and traversal. Include comments explaining the time complexity of each operation.",
            "Create a detailed business plan for a renewable energy startup. Include sections on market analysis, financial projections, competitive advantages, and risk assessment. Format the response with clear headings and bullet points.",
        ], help="Specify prompts")
    
    args = parser.parse_args()
    
    models = args.models if args.models else ["qwen2.5:72b", "qwen2.5:72b_nc_8192"]  # Default models
    results = []
    
    for model in models:
        for prompt in args.prompts:
            print(f"Benchmarking {model} with prompt: {prompt}")
            response = run_benchmark(model, prompt, args.verbose)
            if response:
                results.append(response)
    
    if results:
        save_to_csv(results)
    else:
        print("No valid responses received.")


if __name__ == "__main__":
    main()
