import pandas as pd
import re
import tqdm

df = pd.read_csv('long_video.tsv', sep='\t')

df_time_grouped = {
    "start": [],
    "end": [],
    "text": [],
}

last_start = 0
time_threshold_for_new_segment = 20 * 1000 # 20 seconds

for index, row in tqdm.tqdm(df.iterrows()):
    if index == 0:
        df_time_grouped['start'].append(row['start'])
        df_time_grouped['end'].append(row['end'])
        df_time_grouped['text'].append(row['text'])
        last_start = row['start']
        continue
    if row['start'] - last_start < time_threshold_for_new_segment:
        df_time_grouped['end'][-1] = row['end']
        df_time_grouped['text'][-1] += row['text']
    else:
        df_time_grouped['start'].append(row['start'])
        df_time_grouped['end'].append(row['end'])
        df_time_grouped['text'].append(row['text'])
        last_start = row['start']

df_time_grouped = pd.DataFrame(df_time_grouped)
df_time_grouped['duration_minutes'] = (df_time_grouped['end'] - df_time_grouped['start'] ) / 1000 / 60
df_time_grouped.to_csv('long_video_time_grouped.csv', index=False)
print(df_time_grouped['duration_minutes'])
# df_grouped = {
#     "start": [],
#     "end": [],
#     "text": [],
# }

# for index, row in tqdm.tqdm(df.iterrows()):
#     if index == 0:
#         df_grouped['start'].append(row['start'])
#         df_grouped['end'].append(row['end'])
#         df_grouped['text'].append([row['text']])
#         continue

#     if re.search(r'\d-\d', row['text']):
#         df_grouped['start'].append(row['start'])
#         df_grouped['end'].append(row['end'])
#         df_grouped['text'][-1] = " ".join(df_grouped['text'][-1])
#         df_grouped['text'].append([row['text']])
#     else:
#         df_grouped['text'][-1].append(row['text'])
#         df_grouped['end'][-1] = row['end']

# if isinstance(df_grouped['text'][-1], list):
#     df_grouped['text'][-1] = " ".join(df_grouped['text'][-1])

# df_grouped = pd.DataFrame(df_grouped)
# df_grouped['duration_minutes'] = (df_grouped['end'] - df_grouped['start'] ) / 1000 / 60
# df_grouped.to_csv('long_video_grouped.csv', index=False)

# print(df_grouped['duration_minutes'])