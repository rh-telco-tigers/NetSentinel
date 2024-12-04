# Setting Up MinIO and Models for NetSentinel

- Ensure your MinIO service is up and running.

```
oc get pods -l  app=minio -n netsentinel
NAME                     READY   STATUS    RESTARTS   AGE
minio-79f8869bf5-tntxq   1/1     Running   0          3h14m
```

Any S3-compatible bucket will work, but we assume the default credentials are `minio` (username) and `minio123` (password) with the following S3 endpoint. Update these settings in the appropriate locations if needed.

Get the minio endpoints as follows.

```
oc get routes minio-api -n netsentinel
NAME        HOST/PORT                                                               PATH   SERVICES        PORT   TERMINATION     WILDCARD
minio-api   minio-api-netsentinel.apps.cluster-bbgs4.bbgs4.sandbox592.opentlc.com          minio-service   api    edge/Redirect   None
```

## Install AWS CLI

```
pip install awscli
```

## Configure AWS CLI

```
aws configure
AWS Access Key ID [None]: minio
AWS Secret Access Key [None]: minio123
Default region name [None]: us-east-1
Default output format [None]:
```

## Create Alias and S3 Bucket

Minio endpoints from above.

```
alias s3="aws s3 --endpoint-url https://minio-api-netsentinel.apps.cluster-bbgs4.bbgs4.sandbox592.opentlc.com"
s3 mb s3://netsentinel
```

---

## Download and Upload Predictive Model

Download the predictive model from Hugging Face and upload it to MinIO:

```
git clone https://huggingface.co/bkpandey/netsentinel netsentinel-model
cd netsentinel-model/
s3 cp v1/config.pbtxt s3://netsentinel/predictive-model/config.pbtxt
s3 cp v1/1/model.onnx s3://netsentinel/predictive-model/1/model.onnx
```

---

## Download and Upload Generative Model (Optional)

If you are using [Models as a Service on OpenShift AI](https://maas.apps.prod.rhoai.rh-aiservices-bu.com/) skip this step and refer to the relevant [documentation](./model-as-a-service.md). Otherwise, download the generative model from Hugging Face and upload it to MinIO:

```
git clone https://huggingface.co/ibm-granite/granite-8b-code-instruct-128k
cd granite-8b-code-instruct-128k
s3 cp . s3://netsentinel/llm-model/1/ --recursive --exclude ".*"
```
