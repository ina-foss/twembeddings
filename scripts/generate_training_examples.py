import torch
from transformers import pipeline
import csv

pipe = pipeline("text-generation",
                model="HuggingFaceH4/zephyr-7b-alpha", torch_dtype=torch.bfloat16, device_map="auto")

with open("../data/tweets_pairs_valid_scores.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)
    for i, row in enumerate(reader):

        title = row[1]

        # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
        messages = [
            {
                "role": "system",
                "content": "You are a French journalist writing on Twitter in a quite format tone",
            },
            {
                "role": "user",
                "content": "Write three different headlines in French focusing on different aspects of the following tweet: '{}'".format(title)},
        ]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.8, top_k=20, top_p=0.95)
        print(outputs[0]["generated_text"])

        if i == 10:
            break

