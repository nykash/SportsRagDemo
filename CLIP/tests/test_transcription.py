from utils import process_video as pv
import pickle
import os


def test_transcription(video_path):
    raw_csv_path = pv.transcribe_video(video_path)
    
    if not raw_csv_path:
        print("Transcription failed.")
        return

    print("\n--- GROUPING TEXT ---")
    grouped_csv_path = pv.group_transcriptions_csv(raw_csv_path, window_size=5, step_size=2)

    print("\n--- SPLITTING VIDEO ---")
    # 3. Create Clips using the GROUPED csv
    # Use the video name for the folder (e.g. "germany_v_japan_clips")
    folder_name = os.path.splitext(os.path.basename(video_path))[0] + "_clips"
    pv.create_video_clips(grouped_csv_path, video_path, output_folder_name=folder_name)



def test_chunk_transcription(path):
    transcription = pv.transcribe_video(video_path)
    chunks = pv.chunk_transcription(transcription)
    print(chunks)

def test_summarize_transcription(path):
    transcription = pv.transcribe_video(video_path)
    chunks = pv.chunk_transcription(transcription)
    summaries = pv.summarize_transcription(chunks)
    print(summaries)

def test_rag_summarized_transcriptions(path):
    print(f"Processing video: {path}")
    # transcription = pv.transcribe_video(video_path)
    # with open("transcription.pkl", "wb") as f:
    #     pickle.dump(transcription, f)
    print("Transcription complete")
    # chunks = pv.chunk_transcription(transcription)
    # with open("chunks.pkl", "wb") as f:
    #     pickle.dump(chunks, f)
    print("Chunking complete")
    chunks = pickle.load(open("chunks.pkl", "rb"))
    summaries = pv.summarize_transcription(chunks)
    with open("summaries.pkl", "wb") as f:
        pickle.dump(summaries, f)
    print("Summarization complete")
    pv.rag_summarized_transcriptions(summaries)
    print("RAG complete")

if __name__ == "__main__":
    # List of videos to process
    videos = [
        "/home/richard/Desktop/workspace/SportsRagDemo/experiments/germany_v_japan.mp4",
        "/home/richard/Desktop/workspace/SportsRagDemo/experiments/canada_v_usa.mp4",
        "/home/richard/Desktop/workspace/SportsRagDemo/experiments/brazil_v_usa.mp4",
        "/home/richard/Desktop/workspace/SportsRagDemo/experiments/brazil_v_japan.mp4",
        "/home/richard/Desktop/workspace/SportsRagDemo/experiments/france_v_japan.mp4",
        "/home/richard/Desktop/workspace/SportsRagDemo/experiments/germany_v_usa.mp4"
    ]

    for video in videos:
        if os.path.exists(video):
            test_transcription(video)
        else:
            print(f"File not found: {video}")
    
    # test_chunk_transcription()
    # test_summarize_transcription()
    # test_rag_summarized_transcriptions(video_path)