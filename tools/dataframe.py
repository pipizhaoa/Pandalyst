import pandas as pd
import numpy as np
import random

def got_type(list_):
    def judge(string):
        try:
            int(string)
            return "int"
        except:
            try:
                float(string)
                return "float"
            except:
                return "string"
    return [judge(x) for x in list_]

def column_des(df):

    def single_des(name,data):
        description = "{\"Column Name\": \"" + name + "\"" + ", "
        find_valid = False
        for s in data:
            if not pd.isnull(s):
                find_valid = True
                break

        if not find_valid:
            return ""
        data = [x for x in data if not pd.isnull(x)]
        types = got_type(data)
        if "string" in types:
            type_ = "string"
            data = [str(x) for x in data]
        elif "float" in types:
            type_ = "float"
            data = np.array([float(x) for x in data])
        else:
            type_ = "int"
            data = np.array([int(x) for x in data])

        description = description + "\"Type\": \"" + type_ + "\", "
        if type_ in ["int", "float"]:
            min_ = data.min()
            max_ = data.max()
            description = description + "\"MIN\": " + str(min_) + ", \"MAX\": " + str(max_)
        elif type_ == "string":
            values = list(set(["\"" + str(x).strip() + "\"" for x in data if not pd.isnull(x)]))
            if len(values) >= 20: ## to adjust: Max 20 Enumerated Values
                values = values[:20]
            numerates = ", ".join(values)
            description = description + "\"Enumerated Values\": [" + numerates + "]"
        return description + "}"

    columns_dec = [single_des(c,df[c]) for c in df.columns]
    random.shuffle(columns_dec)
    return "\n".join([x for x in columns_dec if x])

def generate_dataframes(df,df_name=None,df_desc=None):
    rows_count = len(df)
    columns_count = len(df.columns)
    description = "Dataframe "
    if df_name is not None:
        description += f"Name: {df_name}"
    description += (
        f", with {rows_count} rows and {columns_count} columns."
    )
    if df_desc is not None:
        description += f"\nDescription: {df_desc}"
    description += f"""
Columns: {", ".join(df.columns)}
Here are the descriptions of the columns of the dataframe:
{column_des(df)}"""  # noqa: E501
    return description


def build_prompt(df_dec, question):
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