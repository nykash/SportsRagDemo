import pandas as pd
import tqdm
import ollama

def encode_text(text, previous_commentary):
    response = ollama.chat(
        model="llama3:8b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that listens to Volleyball commentary and summarizes the events that happened in the video. The previous commentary is: " + previous_commentary + ". Now you will be given a commentary that comes after the previous commentary. You will need to summarize the events that happened in the commentary. Only output the summary as a paragraph with no line breaks, no other text. Never mention the word commentary, just describe the events that happened."},
            {"role": "user", "content": "Now summarize the following commentary:\n" + text},

        ]
    )
    return response['message']['content']

df = pd.read_csv('long_video_time_grouped.csv')

encoded_text = []

for index, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    if index == 0:
        previous_commentary = ""
    else:
        previous_commentary = df.iloc[index - 1]['text']
    encoded_text.append(encode_text(row['text'], previous_commentary))

df['encoded_text'] = encoded_text
df.to_csv('long_video_time_grouped_encoded.csv', index=False)
print(df['encoded_text'])