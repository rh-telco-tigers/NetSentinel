import os
import boto3
import argparse
import logging
from botocore.exceptions import NoCredentialsError

def setup_logging(log_level):
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def upload_to_s3(file_path, bucket_name, s3_key):
    """
    Upload a file to an S3 bucket.

    Args:
        file_path (str): Path to the file to upload.
        bucket_name (str): The name of the S3 bucket.
        s3_key (str): The S3 key (filename in the bucket).

    Returns:
        str: The S3 URL of the uploaded file.
    """
    try:
        s3 = boto3.client('s3')
        s3.upload_file(file_path, bucket_name, s3_key)
        s3_url = f"s3://{bucket_name}/{s3_key}"
        logging.info(f"File uploaded to S3: {s3_url}")
        return s3_url
    except FileNotFoundError:
        logging.error(f"File {file_path} not found.")
        return None
    except NoCredentialsError:
        logging.error("AWS credentials not available.")
        return None
    except Exception as e:
        logging.error(f"Error uploading file to S3: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Upload file to S3")
    parser.add_argument('--file_path', type=str, default='./models/predictive_model/model.onnx', help='Path to the file to upload (default: "./models/predictive_model/model.onnx")')
    parser.add_argument('--bucket_name', type=str, default='my-default-bucket', help='S3 bucket name (default: "my-default-bucket")')
    parser.add_argument('--s3_key', type=str, default='model.onnx', help='S3 key for the uploaded file (default: "model.onnx")')
    parser.add_argument('--log_level', type=str, default='INFO', help='Log level (default: INFO)')
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Upload the file to S3
    s3_url = upload_to_s3(args.file_path, args.bucket_name, args.s3_key)

    if s3_url:
        logging.info(f"Upload successful. File available at {s3_url}")
    else:
        logging.error("Upload failed.")

if __name__ == "__main__":
    main()
