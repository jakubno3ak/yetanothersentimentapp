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

def download_artifacts(settings=settings):
    s3 = boto3.client("s3")
    bucket_name = settings.s3_bucket_name
    
    print(f"Downloading artifacts from bucket: {bucket_name}")

    os.makedirs(settings.onnx_model_path, exist_ok=True)
    
    classifier_key = "classifier.joblib"
    local_classifier_path = settings.classifier_path
    
    try_download_file(s3, bucket_name, classifier_key, local_classifier_path)
    
    sentence_transformer_key = "sentence_transformer.model"
    local_model_dir = settings.model_path 
    
    try_download_file(s3, bucket_name, sentence_transformer_key, local_model_dir)

    print("Download complete.")