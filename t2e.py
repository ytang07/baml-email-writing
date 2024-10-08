import dotenv
dotenv.load_dotenv()

from typing import List
from baml_client.async_client import b
from baml_client.types import Question
import tqdm

import tiktoken

def split_large_text(large_text: str, max_tokens: int) -> List[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokenized_text = enc.encode(large_text)

    chunks = []
    current_chunk = []
    current_length = 0

    for token in tokenized_text:
        current_chunk.append(token)
        current_length += 1

        if current_length >= max_tokens:
            chunks.append(enc.decode(current_chunk).rstrip(' .,;'))
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunks.append(enc.decode(current_chunk).rstrip(' .,;'))

    return chunks

async def process_transcript(transcript: str):
    chunks = split_large_text(transcript, 2000)
    qs: List[Question] = []
    for c in (pbar := tqdm.tqdm(chunks)):
        stream = b.stream.ExtractQuestions(c)
        num_questions = 0
        async for chunk in stream:
            num_questions = len(chunk)
            pbar.set_description(f'{num_questions} questions')

        qs.extend(await stream.get_final_response())
    with open("questions.jsonl", "w") as f:
        for q in qs:
            f.write(q.model_dump_json() + "\n")

# print(example("This is not a resume"))

transcript = open('transcript.txt','r').read()

import asyncio
asyncio.run(process_transcript(transcript))
