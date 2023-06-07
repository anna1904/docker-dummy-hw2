from pathlib import Path
import boto
import sys, os
from boto.s3.key import Key
from boto.exception import S3ResponseError


def backup_s3_folder(path_to_save):
    BUCKET_NAME = "war-images"
    AWS_ACCESS_KEY_ID = os.getenv("AWS_KEY_ID")  # set your AWS_KEY_ID  on your environment path
    AWS_ACCESS_SECRET_KEY = os.getenv("AWS_ACCESS_KEY")  # set your AWS_ACCESS_KEY  on your environment path
    conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_ACCESS_SECRET_KEY)
    bucket = conn.get_bucket(BUCKET_NAME)

    # goto through the list of files
    bucket_list = bucket.list()

    for l in bucket_list:
        key_string = str(l.key)
        s3_path = path_to_save + key_string
        try:
            l.get_contents_to_filename(s3_path)
        except (OSError, S3ResponseError) as e:
            pass
            # check if the file has been downloaded locally
            if not os.path.exists(s3_path):
                try:
                    os.makedirs(s3_path)
                except OSError as exc:
                    # let guard againts race conditions
                    import errno
                    if exc.errno != errno.EEXIST:
                        raise


def load_data(path_to_save: Path):
    if not os.path.exists(path_to_save):
        print("Making download directory")
        os.mkdir(path_to_save)

    backup_s3_folder(str(path_to_save) + '/train/')
