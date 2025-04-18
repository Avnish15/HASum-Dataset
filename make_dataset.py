import argparse
import pandas as pd
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm

import feature_google
import feature_clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir', type=str, default='videos/Surfing')
    parser.add_argument('--annotation-file', type=str, default='videos/Surfing/Surfing_metadata.xlsx')
    parser.add_argument('--sample-rate', type=int, default=15)
    parser.add_argument('--save-path', type=str, default='videos/Surfing/eccv16_dataset_custom_google_pool5.h5')
    parser.add_argument('--feature-extractor', type=str, choices=['google', 'clip'], default='google')
    args = parser.parse_args()

    out_dir = Path(args.save_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    annotation_file = Path(args.annotation_file)

    print(f'Loading {args.feature_extractor} feature extractor ...')
    video_proc = feature_google.VideoPreprocessor(
        args.sample_rate) if args.feature_extractor == 'google' else feature_clip.VideoPreprocessor(args.sample_rate)

    video_paths = sorted(Path(args.video_dir).glob('*.mp4'))
    print(f'Processing {len(video_paths)} videos ...')

    with h5py.File(args.save_path, 'a') as h5out:  # Open in append mode
        df = pd.read_excel(annotation_file)

        for idx, video_path in tqdm(list(enumerate(video_paths))):
            video_name = video_path.stem
            video_key = f'{video_name}'

            n_frames, features, cps, nfps, picks = video_proc.run(video_path)
            video_annotations = df[df["Video_ID"].str.lower() == video_name.lower()]

            if video_annotations.empty:
                raise ValueError(f"No annotations found for video: {video_name}")

            # Extract video category (tags)
            video_tags = str(video_annotations["Video_Tags"].values[0]).strip("[]")

            # # Extract annotations
            # annotation_columns = [col for col in df.columns if col.startswith("Annotation")]
            # user_summary = video_annotations[annotation_columns].values[0]
            # user_summary = np.array([eval(a) for a in user_summary], dtype=np.float32)
            #
            # _, label_n_frames = user_summary.shape
            # assert label_n_frames == n_frames, f'Invalid label of size {label_n_frames}: expected {n_frames}'
            #
            # gtscore = np.mean(user_summary[:, ::args.sample_rate], axis=0)

            # Save features and labels for this video
            h5out.create_dataset(f'{video_key}/features', data=features)
            # h5out.create_dataset(f'{video_key}/gtscore', data=gtscore)
            # h5out.create_dataset(f'{video_key}/user_summary', data=user_summary)
            h5out.create_dataset(f'{video_key}/change_points', data=cps)
            h5out.create_dataset(f'{video_key}/n_frame_per_seg', data=nfps)
            h5out.create_dataset(f'{video_key}/n_frames', data=n_frames)
            h5out.create_dataset(f'{video_key}/picks', data=picks)
            h5out.create_dataset(f'{video_key}/video_name', data=video_name)
            h5out.create_dataset(f'{video_key}/video_tags', data=video_tags)


    print(f'Dataset saved to {args.save_path}')


if __name__ == '__main__':
    main()
