# Steps To Run Netsentinel on RHDP instance

## 1. Order OpenShift Environment

- Any OpenShift environment should work technically, provided there are no operator conflicts. To avoid issues, it’s recommended to start with a clean environment since the project requires installing multiple operators and configurations.
- For testing purposes, we are using the following environment.
  - Order an OCP demo cluster via this [URL](https://catalog.demo.redhat.com/catalog?item=babylon-catalog-prod/sandboxes-gpte.ocp-wksp.prod&utm_source=webapp&utm_medium=share-link)
  - Select **OpenShift Version 4.17** during setup.
  - Only a single control plane is sufficient.
  - If you are using **Model as a Service** for the LLM model, a CPU-only setup is adequate for deploying this project.

- Wait for Cluster to be provisioned. It takes around 30-60 mins.


## 2. Clone the RHDP Bootstrap Repository
- Clone the bootstrap repository using the below command

        git clone https://github.com/rh-telco-tigers/rhpds-bootstrap.git


## 3. Deploy using Helm Commamd
- Install helm cli via this [URL](https://helm.sh/docs/intro/install/)
- After the cluster is provisioned, deploy your application and its dependencies using Helm, please run the following commands in sequence after cloning the **bootstrap repository**:

    
    - ### Install GitOps Operator (e.g., Argo CD and related CRDs)
      
    helm template netsentinel bootstrap/ --set=gitops.operator.install=true | oc apply -f -
    
    - ### Step 2: Override argocd instance
      
        helm template netsentinel bootstrap/ --set=gitops.operator.install=true,gitops.argocd.install=true | oc apply -f -

    - ### Step 3: Deploy the main application components
      
        helm template netsentinel bootstrap/ --set=gitops.operator.install=true,gitops.argocd.install=true,apps.netsentinel.enabled=true | oc apply -f -

