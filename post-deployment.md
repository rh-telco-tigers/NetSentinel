## Post Deployment Steps for Netsentinel
After setting up the cluster, you need to manually create the following secret:


	 {{- if .Values.createAppSecret }}
	apiVersion: v1
	kind: Secret
	metadata:
	  name: app-config-secret
	  annotations:
	    argocd.argoproj.io/compare-options: IgnoreExtraneous
	    argocd.argoproj.io/sync-options: Prune=false
	  labels:
	    app.kubernetes.io/instance: netsentinel-apps
	type: Opaque
	stringData:
	  secret.yaml: |
	    models:
	      llm:
	        token: "replace-with-your-llm-token"
	    slack:
	      bot_token: "replace-with-your-bot-token"
	      signing_secret: "replace-with-your-signing-secret"
	{{- end }}



In order to get the llm token, bot_token and signing_secret follow the below procedure:



### 1. Create LLM Token

I. Visit the following URL to access the "Models as a Service" applications admin page: https://maas.apps.prod.rhoai.rh-aiservices-bu.com/admin/applications

 Click on "Create new application":


![Create New Application](./docs/images/maas/001-create-new-app.png)



II. Select a Model From the list of available models, select "Granite-8B-Code-Instruct".

- Visit the following URL to access the "Models as a Service" applications admin page: https://maas.apps.prod.rhoai.rh-aiservices-bu.com/admin/applications

- Select a Model From the list of available models, select "Granite-8B-Code-Instruct".
  ![Select Model](./docs/images/maas/002-granite-family.png)
              

- Fill Out the Application Form. Provide the following details:
	- Name: netsentinel-demo
   
	- Description: LLM endpoints for netsentinel agent


III. Retrieve API Information After creating the application, you'll be presented with the following details: 

- URL:https://granite-8b-code-instruct-maas-apicast-production.apps.prod.rhoai.rh-aiservices-bu.com:443
- Model Name: granite-8b-code-instruct-128k 
- API Key: (e.g., ••••••••••••••••••••••••••••••••••••••••)

  ![API Information](./docs/images/maas/004-llm-credentials.png)

           

IV. Copy the API key and update it in the app-config-secret.yaml file


## 2. Configure Slack to get “bot_token” and “signing_secret”

Create a new Slack account or use an existing one. Ensure you have admin permissions to configure the required settings. You can get started at: https://slack.com/get-started#/createnew.

I. Once the slack account is setup navigate to https://api.slack.com/apps.
   ![Create App](./docs/images/slack/001-slack.png)
	
II. Click on Create New App -> Select From a manifest
   ![New App](./docs/images/slack/002-slack.png)

III. Choose your workspace where you want to create this app
    ![Slack Workspace](./docs/images/slack/003-slack.png)
                                   
 
IV. Use the following manifest, replace the request_url using netsentinel routes. This requires verified SSL certificates (self-signed certs won’t work).

	display_information:
	  name: NetSentinelRHDev
	features:
	  bot_user:
	    display_name: NetSentinelRHDev
	    always_online: false
	oauth_config:
	  scopes:
	    user:
	      - channels:history
	      - chat:write
	    bot:
	      - app_mentions:read
	      - channels:history
	      - channels:join
	      - channels:read
	      - chat:write
	      - conversations.connect:manage
	      - conversations.connect:read
	      - conversations.connect:write
	      - groups:history
	      - links:read
	settings:
	  event_subscriptions:
	    request_url: https://<REPLACE-ME-WITH-OCP-ROUTES>/slack/events
	    bot_events:
	      - app_mention
	      - link_shared
	      - message.channels
	      - message.groups
	  org_deploy_enabled: false
	  socket_mode_enabled: false
	  token_rotation_enabled: false

#### **Note**: Replace the **request_url** field by the netsentinel route. Get the route using the below command

	oc get routes -n netsentinel netsentinel-route



V. Next, you will land at the "Basic Information" page. Copy the “Signing Secret” from here and update it in the **app-config-secret.yaml**.
   
   ![App Basic Information](./docs/images/slack/004-slack.png)



VI. Navigate to OAuth & Permissions and click Install to NetSentinel under the "OAuth Tokens" section.

   ![App Oauth Tokens](./docs/images/slack/005-slack.png)


VII. Click Allow and return to the OAuth & Permissions page to copy the "Bot User OAuth Token and update the value in **app-config-secret.yaml**. 

   ![App Bot User OAuth Token](./docs/images/slack/006-slack.png)

VIII. Navigate to slack "Event Subscriptions".
- Click on "Retry" if "Request URL" is not verified. Make sure the NetSentinel app is fully up and running and you can hit this endpoint in the browser.

  ![Retry Event Subscriptions](./docs/images/slack/007-slack.png)


  
IX. Your application is now configured and installed in your Slack workspace. Go to https://yourslack.slack.com/ and confirm that the NetSentinelRHDev app appears under the Apps section.

  ![New app](./docs/images/slack/009-slack.png)


X. Create a new channel (e.g., #demo-channel). The channel name can be anything; it doesn't have to match the #netsentinel name used in app-config.yaml.

- Add NetSentinelRHDev to this new channel.

  ![Add Bot](./docs/images/slack/011-slack.png)