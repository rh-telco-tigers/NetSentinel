import kfp
from kfp import dsl
from kfp.dsl import component, InputPath

@component(
    packages_to_install=['boto3'],
)
def upload_to_s3_component(
    file_path: InputPath(),
    bucket_name: str,
    s3_key: str,
    endpoint_url: str
):
    import boto3
    import logging
    import os
    from botocore.exceptions import NoCredentialsError, ClientError

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    def upload_to_s3(file_path, bucket_name, s3_key):
        try:

            region_name = 'us-east-1'

            s3 = boto3.client(
                's3',
                endpoint_url=endpoint_url,
                region_name=region_name,
            )
            s3.upload_file(file_path, bucket_name, s3_key)
            s3_url = f"s3://{bucket_name}/{s3_key}"
            logging.info(f"File uploaded to MinIO: {s3_url}")
            return s3_url
        except KeyError as e:
            logging.error(f"Environment variable {e} not set.")
            return None
        except FileNotFoundError:
            logging.error(f"File {file_path} not found.")
            return None
        except NoCredentialsError:
            logging.error("Credentials not available.")
            return None
        except ClientError as e:
            logging.error(f"Client error: {e}")
            return None
        except Exception as e:
            logging.error(f"Error uploading file to MinIO: {e}")
            return None

    # Assume that file_path is a directory containing 'model.onnx'
    onnx_model_path = os.path.join(file_path, 'model.onnx')
    s3_url = upload_to_s3(onnx_model_path, bucket_name, s3_key)

    if s3_url:
        logging.info(f"Upload successful. File available at {s3_url}")
    else:
        logging.error("Upload failed.")
