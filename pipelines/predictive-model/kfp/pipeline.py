import kfp
from kfp import dsl

from download_dataset_component import download_dataset_component
from preprocess_data_component import preprocess_data_component
from train_model_component import train_model_component
from evaluate_model_component import evaluate_model_component
from export_model_to_onnx_component import export_model_to_onnx_component
from upload_to_s3_component import upload_to_s3_component

@dsl.pipeline(
    name='Predictive Model Pipeline',
    description='A pipeline that downloads data, preprocesses it, trains a model, evaluates it, exports it to ONNX format, and uploads it to S3.'
)
def predictive_model_pipeline(
    dataset: str = 'mrwellsdavid/unsw-nb15',
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = -1,
    bucket_name: str = 'predictive-model-training',
    s3_key: str = 'model.onnx',
    endpoint_url: str = 'http://minio-service.netsentenial:9000',
):
    # Step 1: Download dataset
    download_task = download_dataset_component(
        dataset=dataset,
    )

    # Mount the Kaggle secret as a volume
    kaggle_secret_name = 'kaggle-secret'  # The name of your Kubernetes Secret

    # Use kubernetes.use_secret_as_volume to mount the secret
    from kfp import kubernetes

    kubernetes.use_secret_as_volume(
        task=download_task,
        secret_name=kaggle_secret_name,
        mount_path='/.config/kaggle',  # Kaggle API expects kaggle.json here
    )

    # Step 2: Preprocess data
    preprocess_task = preprocess_data_component(
        raw_data_path=download_task.outputs['download_path'],
    )

    # Step 3: Train model
    train_task = train_model_component(
        processed_data_path=preprocess_task.outputs['processed_data_path'],
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    # Step 4: Evaluate model
    evaluate_task = evaluate_model_component(
        model_input=train_task.outputs['model_output'],
        processed_data_path=preprocess_task.outputs['processed_data_path'],
    )

    # Step 5: Export model to ONNX
    export_task = export_model_to_onnx_component(
        model_input=train_task.outputs['model_output'],
        processed_data_path=preprocess_task.outputs['processed_data_path'],
    )

    # Step 6: Upload to S3
    upload_task = upload_to_s3_component(
        file_path=export_task.outputs['onnx_model_output'],
        bucket_name=bucket_name,
        s3_key=s3_key,
        endpoint_url=endpoint_url,
    )
    kubernetes.use_secret_as_volume(
        task=upload_task,
        secret_name='aws-credentials',
        mount_path='/root/.aws',
    )

if __name__ == '__main__':
    from kfp import compiler
    compiler.Compiler().compile(predictive_model_pipeline, 'predictive_model_pipeline.yaml')
