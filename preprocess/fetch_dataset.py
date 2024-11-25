#%%
from datasets import load_dataset, concatenate_datasets
#%%
def convert_to_conversational_format(x, input, output):
    """Convert a pair of input/output (typically question/answer) to conversational format."""

    return {"messages": [{"role": "user", "content": x[input]}, {"role": "assistant", "content": x[output]}]}


ds_1 = load_dataset("jrobador/mathinstruct_es", split="train")
#Two questions has no anwser: IDs = [18865, 143408]
remove_ids = [18865, 143408]
ds_1 = ds_1.filter(lambda _, idx: idx not in remove_ids, with_indices=True)
ds_2 = load_dataset("jrobador/gsm8k_es", split="train")
ds_1 = ds_1.rename_columns({"instruction": "question", "output": "answer"})
dataset = concatenate_datasets([ds_1, ds_2])


dataset = dataset.map(lambda x: convert_to_conversational_format(x, "question", "answer"), remove_columns =["question", "answer", "Unnamed: 0", "source"])

dataset.to_json("../data/processed/MathSet_spanish.json", orient="records", lines=True)

