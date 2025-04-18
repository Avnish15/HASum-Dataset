import h5py
import argparse
import numpy as np

def visualize_hdf5(file_path):
    identical_count = 0
    total_checked = 0

    with h5py.File(file_path, 'r') as h5in:
        print(f"\n🔍 Reading HDF5 file: {file_path}\n")
        for video_key in h5in.keys():
            video_data = h5in[video_key]

            print(f"🎞 Video: {video_key}")
            for dataset_key, data in video_data.items():
                try:
                    if hasattr(data, 'shape') and data.shape == ():
                        print(f"  - {dataset_key:<20}: {data[()]}")
                    else:
                        shape_str = f"{data.shape}" if hasattr(data, 'shape') else "N/A"
                        print(f"  - {dataset_key:<20}: shape {shape_str}")
                except Exception as e:
                    print(f"  - {dataset_key:<20}: Error reading data ({e})")

            try:
                user_summary = video_data["user_summary"][:]
                if user_summary.shape[0] == 3:
                    total_checked += 1
                    all_equal = np.allclose(user_summary, user_summary[0:1], atol=1e-6)
                    print(f"  🔁 Identical summaries: {'✅ Yes' if all_equal else '❌ No'}")
                    if all_equal:
                        identical_count += 1
            except Exception as e:
                print(f"  ⚠️ Error checking user_summary for {video_key}: {e}")
            print()

    print(f"📊 Total videos checked: {total_checked}")
    print(f"✅ Videos with identical summaries: {identical_count}")
    print(f"❌ Videos with differing summaries: {total_checked - identical_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, default="data/eccv16_dataset_hasum_google_pool5.h5", nargs="?", help="Path to the HDF5 file.")
    visualize_hdf5(parser.parse_args().file)
