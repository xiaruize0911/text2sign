import os
import subprocess
import zipfile
from pathlib import Path
from typing import List
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from datasets import load_dataset

def convert_video_to_gif(video_path: str, gif_path: str, size: tuple = (128, 128), num_frames: int = 10) -> None:
    """
    Converts a video file (MP4) to a GIF with specified size and number of frames using OpenCV.

    :param video_path: Path to the input video (MP4).
    :param gif_path: Path to save the output GIF.
    :param size: Desired size for the GIF (default is 128x128).
    :param num_frames: The number of frames to sample for the GIF (default is 10).
    """
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame
                frame_resized = cv2.resize(frame_rgb, size, interpolation=cv2.INTER_AREA)
                
                # Convert to PIL Image
                pil_frame = Image.fromarray(frame_resized)
                frames.append(pil_frame)
        
        cap.release()
        
        if frames:
            # Calculate duration for smooth animation
            # For sign language, 200ms per frame gives good visibility
            frame_duration = 200  # 200ms per frame = 5 FPS
            
            # Save as GIF
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=frame_duration,
                loop=0,
                optimize=True  # Optimize file size
            )
            print(f"Converted {video_path} to GIF and saved as {gif_path} ({len(frames)} frames)")
        else:
            print(f"No frames extracted from {video_path}")
            
    except Exception as e:
        print(f"Error converting video {video_path} to GIF: {e}")


def create_training_data(df: pd.DataFrame, videos_dir: str, output_dir: str, size: tuple = (128, 128), num_frames: int = 10) -> None:
    """
    Creates a training folder containing GIFs and corresponding caption text files.

    :param df: DataFrame containing video data.
    :param videos_dir: Directory where videos are stored.
    :param output_dir: Directory where the training data will be saved.
    :param size: Desired size for the GIF (default is 128x128).
    :param num_frames: The number of frames to sample for the GIF (default is 10).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Starting the conversion of videos to GIFs and creating caption text files...")
    
    # Use tqdm to show a progress bar while iterating over the rows of the DataFrame
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Videos", ncols=100):
        video_id = row['SENTENCE_NAME']
        caption = row['SENTENCE']
        
        # Define paths
        video_path = os.path.join(videos_dir, f"{video_id}.mp4")
        gif_path = os.path.join(output_dir, f"{video_id}.gif")
        caption_path = os.path.join(output_dir, f"{video_id}.txt")
        
        # Convert video to GIF with size and frame limit
        convert_video_to_gif(video_path, gif_path, size=size, num_frames=num_frames)
        
        # Save the caption in a text file
        with open(caption_path, 'w') as caption_file:
            caption_file.write(caption)

        print(f"Processed video {video_id}")

    print(f"Training data successfully created in {output_dir}")


lst_of_data = [('test_rgb_front_clips', 'how2sign_realigned_test.csv'), ('val_rgb_front_clips', 'how2sign_realigned_val.csv')]

def main():
    # Step 1: Download the Kaggle dataset
    download_dir = '../../../raw_data'

    # Step 3: Define the path to the TrainValVideo directory where the videos are located
    for source_folder, csv_path in lst_of_data:
        videos_dir = os.path.join(download_dir, source_folder)

        basename = os.path.basename(os.getcwd())
        output_dir = "../../training_data" 
        csv_data = pd.read_csv(os.path.join(download_dir, csv_path), sep='\t')

        create_training_data(csv_data, videos_dir, output_dir, size=(128, 128), num_frames=28)
        print(f"Finished processing {source_folder}")


if __name__ == "__main__":
    main()
