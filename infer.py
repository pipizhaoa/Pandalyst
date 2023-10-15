import pandas as pd

from tools.codemanager.helpers.code_manager_cus import CodeManager
from tools.dataframe import generate_dataframes, code_prompt, plot_code_prompt
import re
import torch
from transformers import GenerationConfig

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
    return result, code_to_run

def evaluate(
        instruction,
        model,
        tokenizer,
        generation_config,
        try_n = 1,
        max_new_tokens=2048,
):
    prompts = generate_prompt(instruction)
    inputs = tokenizer([prompts] * try_n, return_tensors="pt", max_length=4096, truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(model.device)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
            max_new_tokens=max_new_tokens,
            num_return_sequences=try_n,
        )
    s = generation_output.sequences
    output = tokenizer.batch_decode(s, skip_special_tokens=True)
    return output

def infer(df: pd.DataFrame,
          question: str,
          model,
          tokenizer,
          df_name: str = None,
          df_description: str = None,
          try_n: int = 1,
          plot: bool = False,
          save_img: str = None):
    """
    :param df: pd.DataFrame
    :param question:
    :param model:
    :param tokenizer:
    :param df_name:
    :param df_description:
    :param try_n: batch size
    :param plot: if ask model to plot chart
    :param save_img: the path_name to save chart
    :return:
    """

    df_dec = generate_dataframes(df, df_name, df_description)

    if plot:
        _, instruction = plot_code_prompt(df_dec, question)
    else:
        _ ,instruction = code_prompt(df_dec, question)

    if try_n == 1:
        do_sample = False
    else:
        do_sample = True

    generation_config = GenerationConfig(
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_p=0.95,
    )

    _output = evaluate(instruction,model,tokenizer,generation_config,try_n)

    if plot:
        anly_codemanager = CodeManager(df=df, func_name="plot_chart")
    else:
        anly_codemanager = CodeManager(df=df, func_name="analyze_data")

    try:
        final_output = _output[0].split("### Response:")[1].strip()
    except:
        print("input prompt is too long, return empty answer")
        return "", ""

    for output_code in _output:

        if plot and save_img:
            output_code.replace("./temp_chart.png", save_img)

        answer, code_to_run = run_one_code(final_output, anly_codemanager)

        if code_to_run:
            print("code running successfully")
            return answer, code_to_run

    return None, None