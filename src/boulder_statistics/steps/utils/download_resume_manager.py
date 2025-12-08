import os
import shutil


class DownloadResumeManager:
    @staticmethod
    def combine_parts(file_path: str, n_parts: int, resume_file: str) -> None:
        tmp_path = file_path + ".part"
        with open(tmp_path, "wb") as outfile:
            for index in range(n_parts):
                part_path: str = f"{file_path}.part{index}"
                with open(part_path, "rb") as infile:
                    shutil.copyfileobj(infile, outfile)
                os.remove(part_path)
        os.replace(tmp_path, file_path)
        if os.path.exists(resume_file):
            try:
                os.remove(resume_file)
            except Exception:
                pass

    @staticmethod
    def invalidate_cache(file_path: str, resume_file: str,
                         extract_dir: str) -> None:
        if os.path.exists(extract_dir):
            return

        base_dir: str = os.path.dirname(file_path)
        base_name: str = os.path.basename(file_path)

        try:
            for filename in os.listdir(base_dir):
                if filename.startswith(base_name) and ".part" in filename:
                    try:
                        os.remove(os.path.join(base_dir, filename))
                    except Exception:
                        pass
        except FileNotFoundError:
            pass

        if os.path.exists(resume_file):
            try:
                os.remove(resume_file)
            except Exception:
                pass
