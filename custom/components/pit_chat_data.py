import functools
from torchtune.datasets import instruct_dataset, chat_dataset
from torchtune.data import PromptTemplate, ChatFormat
from torchtune.models.llama2 import Llama2Tokenizer, Llama2ChatTemplate


Llama2InstructTemplate = functools.partial(
    PromptTemplate,
    template={
        "user": ("[INST] ", "[/INST]"),
    },
)

if __name__ == "__main__":
    tokenizer = Llama2Tokenizer(path="models/Llama-2-7b-hf/tokenizer.model")
    dataset   = instruct_dataset(
        tokenizer=tokenizer,
        source="json",
        data_files="data/PIT/wiki2023_film_test/qa.jsonl",
        column_map={"input": "question", "output": "answer"},
        split="train",
    )
