from tools.codemanager.helpers.code_manager_cus import CodeManager
from tools.dataframe import generate_dataframes
import re
import torch
from transformers import GenerationConfig

def code_prompt(df_dec, question):
    system_prompt = """You are a world-class python programmer that can complete any data analysis tasks by coding."""

    prompt = f"""You are provided with a following pandas dataframe (`df`):

{df_dec}

Using the provided dataframe (`df`), update the following python code and complete the function (analyze_data) that returns the answer to question: \"{question}\"

This is the initial python code to be updated:
```python
# TODO import all the dependencies required
import pandas as pd
import numpy as np

def analyze_data(df: pd.DataFrame) -> str:
    \"\"\"
    Analyze the data and return the answer of question: \"{question}\"
    1. Prepare: Preprocessing and cleaning data if necessary
    2. Process: Manipulating data for analysis (grouping, filtering, aggregating, etc.)
    3. Analyze: Conducting the actual analysis
    4. Output: Returning the answer as a string
    \"\"\"
```"""
    return system_prompt, prompt


def generate_prompt(instruction):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"""

def extract_code(code):
    if "```python" in code:
        try:
            code = re.findall(r"```python(.*?)```",code, re.DOTALL)[0].strip()
        except:
            code = re.findall(r"```python(.*?)$", code, re.DOTALL)[0].strip()
    elif "```" in code:
        try:
            code = re.findall(r"```(.*?)```",code, re.DOTALL)[0].strip()
        except:
            start = code.find("```")
            code = code[start+3:].strip()
    return code

def run_one_code(code, codemanager):
    code = extract_code(code)
    result, code_to_run = codemanager.execute_code(code)
    assert result, f"code running error: \n{code_to_run}"
    return result, code_to_run

def evaluate(
        instruction,
        model,
        tokenizer,
        generation_config,
        max_new_tokens=2048,
):
    prompts = generate_prompt(instruction)
    inputs = tokenizer(prompts, return_tensors="pt", max_length=4096, truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(model.device)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences
    output = tokenizer.batch_decode(s, skip_special_tokens=True)
    return output

def infer(df,
          question,
          model,
          tokenizer,
          df_name = None,
          df_description = None):

    df_dec = generate_dataframes(df, df_name, df_description)

    _ ,instruction = code_prompt(df_dec, question)

    generation_config = GenerationConfig(
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    _output = evaluate(instruction,model,tokenizer,generation_config)
    anly_codemanager = CodeManager(df=df, func_name="analyze_data")

    try:
        final_output = _output[0].split("### Response:")[1].strip()
    except:
        print("input prompt is too long, return empty answer")
        return "", ""

    answer, code_to_run = run_one_code(final_output, anly_codemanager)
    return answer, code_to_run