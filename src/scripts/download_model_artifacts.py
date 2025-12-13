import boto3
import os
from settings import settings


def try_download_file(s3, bucket_name, key, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {key}...")
        try:
            s3.download_file(bucket_name, key, local_path)
            print(f"Downloaded {key}")
        except Exception as e:
            print(f"Error downloading {key}: {e}")
    else:
        print(f"{key} already exists")


def download_folder(s3, bucket_name, prefix, local_dir):
    print(f"Downloading folder {prefix}...")
    
    # list all objects with the given prefix
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
    downloaded_count = 0
    for page in pages:
        if 'Contents' not in page:
            print(f"No files found in {prefix}")
            return
            
        for obj in page['Contents']:
            key = obj['Key']
            if key.endswith('/'):
                continue
                
            # create local file path
            relative_path = os.path.relpath(key, prefix)
            local_file_path = os.path.join(local_dir, relative_path)
            
            # create local directory if needed
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            if not os.path.exists(local_file_path):
                print(f"  Downloading {key}...")
                try:
                    s3.download_file(bucket_name, key, local_file_path)
                    downloaded_count += 1
                except Exception as e:
                    print(f"  Error downloading {key}: {e}")
            else:
                print(f"  {key} already exists")
    
    print(f"Downloaded {downloaded_count} files from {prefix}")


def download_artifacts(settings=settings):
    s3 = boto3.client("s3")
    bucket_name = settings.s3_bucket_name

    print(f"Downloading artifacts from bucket: {bucket_name}")

    os.makedirs(settings.model_dir, exist_ok=True)

    # download classifier file
    classifier_key = "classifier.joblib"
    local_classifier_path = settings.classifier_joblib_path

    try_download_file(s3, bucket_name, classifier_key, local_classifier_path)

    # download sentence_transformer folder
    sentence_transformer_prefix = "sentence_transformer.model/"
    local_model_dir = settings.sentence_transformer_dir
    os.makedirs(local_model_dir, exist_ok=True)
    
    download_folder(s3, bucket_name, sentence_transformer_prefix, local_model_dir)

    print("Download complete.")
