import utils.process_video as pv
import pickle

def test_transcription(video_path):         
    transcription = pv.transcribe_video(video_path)
    print(transcription)

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
    video_path = "../experiments/long_video.mp4"
    #test_transcription()
    #test_chunk_transcription()
    #test_summarize_transcription()
    test_rag_summarized_transcriptions(video_path)