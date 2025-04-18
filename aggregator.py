import argparse
import h5py
import shutil
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

MAX_SUMMARY_RATIO = 0.15  # 15% summary limit

def play_shot(video_path, start, end):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    gui_available = hasattr(cv2, 'imshow')

    for i in range(start, end):
        ret, frame = cap.read()
        if not ret:
            break
        if gui_available:
            try:
                cv2.imshow('Shot', frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            except cv2.error:
                print("⚠️ GUI not available (cv2.imshow failed), printing frame count instead.")
                gui_available = False
        if not gui_available:
            print(f"🖼️ Frame {i}", end='\r')
    cap.release()
    if gui_available:
        cv2.destroyAllWindows()

def annotate_video(video_id, change_points, total_frames, picks, video_path):
    shot_labels = np.zeros(len(picks), dtype=np.float32)
    selected = 0
    limit = int(MAX_SUMMARY_RATIO * total_frames)

    for idx, (start, end) in enumerate(change_points):
        start_f = picks[start]
        end_f = picks[end] if end < len(picks) else total_frames

        print(f"\nShot {idx + 1}: Frame {start_f} to {end_f}")
        play_shot(video_path, start_f, end_f)

        decision = input("Include this shot in summary? [y/N]: ").strip().lower()
        if decision == 'y':
            if selected + (end_f - start_f) <= limit:
                shot_labels[start:end] = 1
                selected += (end_f - start_f)
                print(f"✅ Added (selected {selected}/{limit})")
            else:
                print("⚠️ Skipped: would exceed 15% summary limit")
        else:
            print("❌ Skipped")

    return shot_labels

def add_video_to_excel(df, video_id, category):
    if video_id not in df["Video_ID"].values:
        new_row = {
            "Video_ID": video_id,
            "Video_Tags": category
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        print(f"➕ Added {video_id} to Annotations.xlsx")
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="videos", type=Path)
    parser.add_argument("--output_dir", default="data", type=Path)
    parser.add_argument("--sample_rate", type=int, default=15)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_dir = output_dir / "videos"
    output_video_dir.mkdir(parents=True, exist_ok=True)

    annotation_excel_path = output_dir / "Annotations.xlsx"
    merged_h5_path = output_dir / "eccv16_dataset_custom_google_pool5.h5"

    # Load or initialize Excel
    if annotation_excel_path.exists():
        df = pd.read_excel(annotation_excel_path)
    else:
        df = pd.DataFrame(columns=["Video_ID", "Video_Tags"])

    # Ask for annotation column
    user_col = input("Enter user annotation column (e.g., Annotation_1): ").strip()
    if user_col not in df.columns:
        df[user_col] = None

    new_video_index = 1

    for category_path in tqdm(sorted(input_dir.iterdir())):
        if not category_path.is_dir():
            continue

        metadata_path = category_path / f"{category_path.name}_metadata.xlsx"
        h5_path = category_path / "eccv16_dataset_custom_google_pool5.h5"
        if not metadata_path.exists() or not h5_path.exists():
            print(f"❌ Missing metadata or H5 for {category_path.name}")
            continue

        df_meta = pd.read_excel(metadata_path)
        with h5py.File(h5_path, "r") as h5_file:
            df_meta["cp_count"] = df_meta["Video_ID"].apply(
                lambda vid: len(h5_file[vid]["change_points"]) if vid in h5_file else 0
            )
        df_meta = df_meta[df_meta["cp_count"] > 0]
        valid_videos = df_meta.sort_values("cp_count", ascending=False)

        if len(valid_videos) < 10:
            print(f"⏭️ Skipping {category_path.name} (only {len(valid_videos)} valid videos)")
            continue

        top10_videos = valid_videos.head(10)

        with h5py.File(h5_path, "r") as h5_file:
            for _, row in top10_videos.iterrows():
                orig_vid = row["Video_ID"]
                new_vid = f"Video_{new_video_index}"
                new_video_index += 1

                # Copy video file
                src_video = category_path / f"{orig_vid}.mp4"
                dst_video = output_video_dir / f"{new_vid}.mp4"
                if src_video.exists():
                    shutil.copy(src_video, dst_video)
                else:
                    print(f"⚠️ Video file not found for {orig_vid}")
                    continue

                df = add_video_to_excel(df, new_vid, category_path.name)

                video_path = dst_video
                group = h5_file[orig_vid]
                change_points = np.array(group["change_points"])
                total_frames = int(group["n_frames"][()])
                picks = np.array(group["picks"])

                print(f"\n🎬 Annotating {new_vid}")
                summary = annotate_video(new_vid, change_points, total_frames, picks, str(video_path))
                df.loc[df["Video_ID"] == new_vid, user_col] = str(summary.tolist())

                user_annotations = [
                    eval(x) for x in df.loc[df["Video_ID"] == new_vid, df.columns[df.columns.str.startswith("Annotation_")]]
                    if pd.notnull(x).values[0]
                ]
                user_summary = np.vstack(user_annotations)
                gtscore = np.mean(user_summary[:, ::args.sample_rate], axis=0)

                with h5py.File(merged_h5_path, "a") as final_h5:
                    if new_vid not in final_h5:
                        h5_file.copy(orig_vid, final_h5, name=new_vid)
                    group = final_h5[new_vid]
                    for key in ["user_summary", "gtscore"]:
                        if key in group:
                            del group[key]
                    group.create_dataset("user_summary", data=user_summary.astype(np.float32))
                    group.create_dataset("gtscore", data=gtscore.astype(np.float32))

                print(f"✅ Saved manual annotation for {new_vid}")

                cont = input("➡️ Continue to next video? [Y/n]: ").strip().lower()
                if cont == 'n':
                    print("👋 Exiting...")
                    df.to_excel(annotation_excel_path, index=False)
                    return

    df.to_excel(annotation_excel_path, index=False)
    print(f"\n📝 Excel saved at {annotation_excel_path}")
    print(f"📁 Final HDF5 saved at {merged_h5_path}")

if __name__ == "__main__":
    main()
