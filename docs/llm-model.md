## Download and Upload Generative Model (Optional)

If you are using [Models as a Service on OpenShift AI](https://maas.apps.prod.rhoai.rh-aiservices-bu.com/) skip this step and refer to the relevant [documentation](./model-as-a-service.md). Otherwise, download the generative model from Hugging Face and upload it to MinIO:

```
git clone https://huggingface.co/ibm-granite/granite-8b-code-instruct-128k
cd granite-8b-code-instruct-128k
s3 cp . s3://netsentinel/llm-model/1/ --recursive --exclude ".*"
```

> Note: This section is not properly tested for now use [Models as a Service on OpenShift AI](https://maas.apps.prod.rhoai.rh-aiservices-bu.com/)
