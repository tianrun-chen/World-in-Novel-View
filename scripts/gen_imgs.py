import os
import glob
import argparse
from skimage import io
from skimage.transform import resize
import yt_dlp

from multiprocessing import Pool
import shutil
from concurrent.futures import ThreadPoolExecutor
import argparse

num_workders_video = 8
num_workers_download = 1


def try_download(url):
    videoname = url.split("=")[-1]
    print(f"[INFO] Downloading {videoname} ...")
    try:
        ydl_opts = {
            "format": "bestvideo[height<=480]",
            "outtmpl": f"./{videoname}",
            "cookies": "./cookies.txt",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as err:
        print(f"[ERROR] Failed to download {url}: {err}")
        return False
    return True


def download_and_process(data, mode, output_root):
    failure_log_file = f"failed_videos_{mode}.txt"
    assert os.path.exists(failure_log_file), f"{failure_log_file} does not exist"

    videoname = data.url.split("=")[-1]
    try_download(data.url)

    # Download successful, proceed to process
    for seq_id in range(len(data)):
        wrap_process((data, seq_id, videoname, output_root))

    if os.path.exists("./" + videoname):
        os.system("rm ./" + videoname)  # remove videos

    # with Pool(processes=num_workers_download) as pool:
    # pool.map(
    #     wrap_process,
    #     [(data, seq_id, videoname, output_root) for seq_id in range(len(data))],
    # )


class Data:
    def __init__(self, url, seqname, list_timestamps):
        self.url = url
        self.list_seqnames = []
        self.list_list_timestamps = []

        self.list_seqnames.append(seqname)
        self.list_list_timestamps.append(list_timestamps)

    def add(self, seqname, list_timestamps):
        self.list_seqnames.append(seqname)
        self.list_list_timestamps.append(list_timestamps)

    def __len__(self):
        return len(self.list_seqnames)


def process(data, seq_id, videoname, output_root):
    seqname = data.list_seqnames[seq_id]
    if not os.path.exists(output_root + seqname):
        os.makedirs(output_root + seqname)
    else:
        print(f"[WARNING] The output dir {output_root + seqname} has already existed.")
        return True

    list_str_timestamps = []
    for timestamp in data.list_list_timestamps[seq_id]:
        timestamp = int(timestamp / 1000)
        str_hour = str(int(timestamp / 3600000)).zfill(2)
        str_min = str(int(int(timestamp % 3600000) / 60000)).zfill(2)
        str_sec = str(int(int(int(timestamp % 3600000) % 60000) / 1000)).zfill(2)
        str_mill = str(int(int(int(timestamp % 3600000) % 60000) % 1000)).zfill(3)
        _str_timestamp = f"{str_hour}:{str_min}:{str_sec}.{str_mill}"
        list_str_timestamps.append(_str_timestamp)

    # Extract frames from a video
    for idx, str_timestamp in enumerate(list_str_timestamps):
        if os.path.exists(
            f"{output_root}{seqname}/{data.list_list_timestamps[seq_id][idx]}.png"
        ):
            continue
        command = (
            f"ffmpeg -loglevel error -ss {str_timestamp} -i {videoname} -vframes 1 "
            f"-f image2 {output_root}{seqname}/{data.list_list_timestamps[seq_id][idx]}.png"
        )
        try:
            os.system(command)
        except Exception as err:
            print(f"[ERROR] Failed to process {data.url}: {err}")
            # shutil.rmtree(output_root + seqname)  # delete the output dir
            return True
    return False


def wrap_process(list_args):
    return process(*list_args)


class DataDownloader:
    def __init__(self, dataroot, mode="test", failure=False):
        self.failure = failure
        self.dataroot = dataroot
        self.mode = mode
        self.output_root = "./" + mode + "/"
        os.makedirs(self.output_root, exist_ok=True)

        # Read failed URLs if failure is True
        failed_urls = []
        if self.failure:
            failure_log_file = f"failed_videos_{self.mode}.txt"
            if os.path.exists(failure_log_file):
                with open(failure_log_file, "r") as f:
                    failed_urls = [line.strip() for line in f if line.strip()]
            else:
                print("[INFO] No failure log file found.")
                failed_urls = []

        # Load data list
        print("[INFO] Loading data list ... ", end="")
        # self.list_seqnames = [
        #     "/scratch/partial_datasets/realestate10k/RealEstate10K/test/c63c6fd4250123f8.txt"
        # ]
        self.list_seqnames = sorted(glob.glob(dataroot + "/*.txt"))
        self.list_data = []
        for txt_file in self.list_seqnames:
            dir_name = txt_file.split("/")[-1]
            seq_name = dir_name.split(".")[0]

            # Extract info from txt
            with open(txt_file, "r") as seq_file:
                lines = seq_file.readlines()
            youtube_url = ""
            list_timestamps = []
            for idx, line in enumerate(lines):
                if idx == 0:
                    youtube_url = line.strip()
                else:
                    timestamp = int(line.split(" ")[0])
                    list_timestamps.append(timestamp)
                # if youtube_url == "https://www.youtube.com/watch?v=2XXlfePPyUo":
                #     print(txt_file)
                #     exit()

            # If failure is True, only include data whose URL is in failed_urls
            if self.failure and youtube_url not in failed_urls:
                # print(f"[INFO] Skipping {youtube_url} because it is not in failed_urls")
                continue  # Skip this data

            isRegistered = False
            for i in range(len(self.list_data)):
                if youtube_url == self.list_data[i].url:
                    isRegistered = True
                    self.list_data[i].add(seq_name, list_timestamps)
                    break
            if not isRegistered:
                self.list_data.append(Data(youtube_url, seq_name, list_timestamps))

        print(f"{len(self.list_data)} movies are used in {self.mode} mode")

    def run(self):
        num_videos = len(self.list_data)
        print(f"[INFO] Start downloading {num_videos} movies")
        with ThreadPoolExecutor(max_workers=num_workders_video) as executor:
            futures = [
                executor.submit(
                    download_and_process,
                    data,
                    self.mode,
                    self.output_root,
                )
                for vid, data in enumerate(self.list_data)
            ]
            for future in futures:
                future.result()
        print("[INFO] Done!")

    def show(self):
        print("########################################")
        global_count = 0
        # seq_ids = set(os.listdir(self.output_root))
        for data in self.list_data:
            # url_list.append(data.url)
            for idx in range(len(data)):
                # print(f" SEQ_{idx} : {data.list_seqnames[idx]}")
                global_count += 1
        # print(f"URL count: {len(set(url_list))}")
        print(f"TOTAL : {global_count} sequences")
        print("########################################")

    def regenerate_failed_urls(self):
        seq_ids = set(os.listdir(f"./{self.mode}"))
        failed_urls = []
        for data in self.list_data:
            for idx in range(len(data)):
                if data.list_seqnames[idx] not in seq_ids:
                    # print(f"SEQ_{idx} : {data.list_seqnames[idx]}")
                    failed_urls.append(data.url)
        failed_urls = sorted(list(set(failed_urls)))
        with open(f"failed_videos_{self.mode}.txt", "w") as f:
            f.writelines([url + "\n" for url in failed_urls])
        print(
            f"{len(failed_urls)} failed URLs are regenerated in failed_videos_{self.mode}.txt"
        )

    def regenerate_incomplete_urls(self):
        seq_ids = set(os.listdir(f"./{self.mode}"))
        incomplete_urls = []
        for data in self.list_data:
            for idx in range(len(data)):
                if data.list_seqnames[idx] not in seq_ids:
                    continue
                # seqname exists, check if all the required timestamps are generated
                tstmp_count = len(data.list_list_timestamps[idx])
                png_count = len(
                    glob.glob(f"{self.output_root}{data.list_seqnames[idx]}/*.png")
                ) + len(
                    glob.glob(
                        f"{self.output_root}{data.list_seqnames[idx]}/images/*.png"
                    )
                )
                if png_count < tstmp_count:
                    incomplete_urls.append(data.url)
                    print(data.url, tstmp_count, png_count)
        incomplete_urls = sorted(list(set(incomplete_urls)))
        with open(f"failed_videos_{self.mode}.txt", "w") as f:
            f.writelines([url + "\n" for url in incomplete_urls])
        print(
            f"{len(incomplete_urls)} incomplete URLs are regenerated in failed_videos_{self.mode}.txt"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process dataset.")
    parser.add_argument(
        "mode", choices=["test", "train"], help="Mode: 'test' or 'train'"
    )

    args = parser.parse_args()
    mode = args.mode
    dataroot = f"./RealEstate10K/{mode}"
    DataDownloader(dataroot, mode, failure=False).run()
    DataDownloader(dataroot, mode, failure=False).show()
    DataDownloader(dataroot, mode, failure=False).regenerate_failed_urls()
    DataDownloader(dataroot, mode, failure=False).regenerate_incomplete_urls()
    DataDownloader(dataroot, mode, failure=True).run()
