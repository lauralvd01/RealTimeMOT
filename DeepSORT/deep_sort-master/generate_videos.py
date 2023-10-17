# vim: expandtab:ts=4:sw=4
import os
import argparse
import show_results


def convert(filename_in, filename_out, ffmpeg_executable="ffmpeg"):
    import subprocess
    command = [ffmpeg_executable, "-i", filename_in, "-c:v", "libx264",
               "-preset", "slow", "-crf", "21", filename_out]
    subprocess.call(command)


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Siamese Tracking")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        required=True)
    parser.add_argument(
        "--result_file", help="Path to the file with tracking output.",
        required=True)
    parser.add_argument(
        "--output_dir", help="Folder to store the videos in. Will be created "
        "if it does not exist.",
        required=True)
    parser.add_argument(
        "--convert_h264", help="If true, convert videos to libx264 (requires "
        "FFMPEG", default=False)
    parser.add_argument(
        "--update_ms", help="Time between consecutive frames in milliseconds. "
        "Defaults to the frame_rate specified in seqinfo.ini, if available.",
        default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    update_ms = args.update_ms
    video_filename = args.output_dir + "/deep_sort_bateau_1_0.5.avi"

    print("Saving to %s." % (video_filename))
    show_results.run(args.mot_dir, args.result_file, False, None, update_ms, video_filename)

    if args.convert_h264:
        filename_out = args.output_dir + "/deep_sort_bateau_1_0.5.mp4"
        convert(video_filename, filename_out)
