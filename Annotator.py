import argparse
import h5py
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

MAX_SUMMARY_RATIO = 0.15  # 15% summary constraint

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
                print("⚠️ Cannot display frames (cv2.imshow not supported). Showing frame count only.")
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
                print(f"✅ Added (total selected: {selected}/{limit})")
            else:
                print("⚠️ Skipped: would exceed 15% limit")
        else:
            print("❌ Skipped")

    return shot_labels

def add_new_video_to_excel(df, video_id):
    if video_id not in df["Video_ID"].values:
        new_row = {col: None for col in df.columns}
        new_row["Video_ID"] = video_id
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        print(f"➕ Added {video_id} to Excel.")
    return df

def check_annotation_status(df, hdf, column):
    print(f"\n📊 Checking annotation completeness and n_frame alignment for column: {column}")
    all_okay = True

    for video_id in hdf.keys():
        group = hdf[video_id]
        n_frames = int(group["n_frames"][()])
        row = df[df["Video_ID"] == video_id]

        if column not in row or pd.isnull(row[column].values[0]):
            print(f"❌ Missing annotation for {video_id}")
            all_okay = False
            continue

        try:
            summary = np.array(eval(row[column].values[0]))
            if len(summary) != len(group["picks"]):
                print(f"⚠️ Mismatch in length for {video_id}: len(summary)={len(summary)}, len(picks)={len(group['picks'])}")
                all_okay = False
        except Exception as e:
            print(f"❌ Error parsing annotation for {video_id}: {e}")
            all_okay = False

    if all_okay:
        print("✅ All videos are annotated correctly and frame lengths match.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_excel", default="data/Annotations.xlsx", type=Path)
    parser.add_argument("--input_h5", default="data/eccv16_dataset_custom_google_pool5.h5")
    parser.add_argument("--video_dir", default="data/videos", type=Path)
    parser.add_argument("--sample_rate", default=15, type=int)
    parser.add_argument("--add_column", help="Add new column to annotation Excel")
    parser.add_argument("--annotate_column", help="Annotate using this column")
    parser.add_argument("--check_status", action='store_true', help="Check annotation completeness and frame match")
    args = parser.parse_args()

    df = pd.read_excel(args.annotation_excel)

    # Add a new column
    if args.add_column:
        if args.add_column not in df.columns:
            df[args.add_column] = None
            df.to_excel(args.annotation_excel, index=False)
            print(f"✅ Added new column: {args.add_column}")
        else:
            print(f"⚠️ Column already exists: {args.add_column}")

    # Check annotations
    if args.check_status:
        with h5py.File(args.input_h5, "r") as hdf:
            check_annotation_status(df, hdf, args.annotate_column if args.annotate_column else "Annotation_1")
        return

    # Annotate videos
    if args.annotate_column:
        if args.annotate_column not in df.columns:
            df[args.annotate_column] = None

        with h5py.File(args.input_h5, "r+") as hdf:
            for video_id in sorted(hdf.keys(), key=lambda x: -len(hdf[x]['change_points'])):
                df = add_new_video_to_excel(df, video_id)

                if not df[(df["Video_ID"] == video_id) & (df[args.annotate_column].isnull())].empty:
                    video_path = args.video_dir / f"{video_id}.mp4"
                    if not video_path.exists():
                        print(f"⚠️ Skipping {video_id}: video file not found")
                        continue

                    print(f"\n🎬 Annotating: {video_id}")
                    group = hdf[video_id]
                    change_points = np.array(group["change_points"])
                    total_frames = int(group["n_frames"][()])
                    picks = np.array(group["picks"])

                    summary = annotate_video(video_id, change_points, total_frames, picks, str(video_path))
                    df.loc[df["Video_ID"] == video_id, args.annotate_column] = str(summary.tolist())

                    # Save user_summary and gtscore
                    user_annotations = [
                        eval(x) for x in df.loc[df["Video_ID"] == video_id, df.columns[df.columns.str.startswith("Annotation_")]]
                        if pd.notnull(x).values[0]
                    ]
                    user_summary = np.vstack(user_annotations)
                    gtscore = np.mean(user_summary[:, ::args.sample_rate], axis=0)

                    if "user_summary" in group:
                        del group["user_summary"]
                    if "gtscore" in group:
                        del group["gtscore"]
                    group.create_dataset("user_summary", data=user_summary.astype(np.float32))
                    group.create_dataset("gtscore", data=gtscore.astype(np.float32))

                    print(f"✅ Annotation and gtscore saved for {video_id}")

                    cont = input("\n➡️ Continue to next video? [Y/n]: ").strip().lower()
                    if cont == 'n':
                        print("👋 Exiting annotation loop.")
                        break

        df.to_excel(args.annotation_excel, index=False)
        print(f"\n📝 Final annotations saved to: {args.annotation_excel}")
    else:
        print("ℹ️ No --annotate_column specified. Use --help for options.")

if __name__ == "__main__":
    main()
