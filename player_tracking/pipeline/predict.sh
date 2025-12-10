#!/bin/bash

# Configuration
DATA_DIR="data"
SPLICES_DIR="data/splices"
OUTPUT_QUAD_DIR="output_quadrilaterals"
OUTPUT_QUAD_WITH_JERSEYS_DIR="output_quadrilaterals_with_jerseys"
OUTPUT_STATS_DIR="output_stats"
JERSEY_CHECKPOINT="runs/jersey_classifier/best_model.pt"

# List of data files to process (without .mp4 extension)
DATA_FILES=("1" "2" "3" "4" "5")

# Activate virtual environment
source venv/bin/activate
pip install -r requirements.txt

echo "=========================================="
echo "Starting Pipeline for ${#DATA_FILES[@]} videos"
echo "=========================================="

# Step 1: Detect clips (creates timestamp JSON files)
echo ""
echo "Step 1: Detecting clips..."
for video_id in "${DATA_FILES[@]}"; do
    video_path="${DATA_DIR}/${video_id}.mp4"
    if [ -f "$video_path" ]; then
        echo "  Processing: $video_path"
        python player_tracking/detect_clips.py --video_path "$video_path"
    else
        echo "  Warning: $video_path not found, skipping..."
    fi
done

# Step 2: Segment videos into clips
echo ""
echo "Step 2: Segmenting videos into clips..."
python player_tracking/segment_videos_into_clips.py

# Step 3: Find jerseys (processes videos in splices directory)
echo ""
echo "Step 3: Finding jerseys..."
python player_tracking/find_jerseys.py

# Step 4: Recognize jersey numbers
echo ""
echo "Step 4: Recognizing jersey numbers..."
python player_tracking/recognize_jersey_numbers.py \
    --quad-dir "$OUTPUT_QUAD_DIR" \
    --output-dir "$OUTPUT_QUAD_WITH_JERSEYS_DIR" \
    --checkpoint "$JERSEY_CHECKPOINT"

# Step 5: Generate player stats
echo ""
echo "Step 5: Generating player stats..."
python player_tracking/generate_player_stats.py \
    --input-dir "$SPLICES_DIR" \
    --jersey-quads-dir "$OUTPUT_QUAD_WITH_JERSEYS_DIR" \
    --output-dir "$OUTPUT_STATS_DIR" \
    --verbose

echo ""
echo "=========================================="
echo "Pipeline completed!"
echo "Results saved to: $OUTPUT_STATS_DIR"
echo "=========================================="