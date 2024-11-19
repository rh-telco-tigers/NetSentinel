- Navigate to [https://api.slack.com/apps](https://api.slack.com/apps)
![Create App](./images/001-slack.png)

- Click on **Create New App**
- Select **From a manifest**
![New App](./images/002-slack.png)

- Choose your workspace where you want to create this app
![Slack Workspace](./images/003-slack.png)

- Use the following manifest, replacing the `request_url` as needed. Example uses OpenShift routes, requiring verified SSL certificates (self-signed certs won’t work). Use ngrok like service to bypass this if necessary for non-prod use case.

#### Get OpenShift Routes
```shell
oc get routes -n netsentinel netsentinel-route
NAME                HOST/PORT                                                                        PATH   SERVICES              PORT   TERMINATION     WILDCARD
netsentinel-route   netsentinel-route-netsentinel.apps.cluster-svlfp.svlfp.sandbox2951.opentlc.com          netsentinel-service   5000   edge/Redirect   None
```

Slack Manifest Example
```
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
    request_url: https://netsentinel-route-netsentinel.apps.cluster-svlfp.svlfp.sandbox2951.opentlc.com/slack/events
    bot_events:
      - app_mention
      - link_shared
      - message.channels
      - message.groups
  org_deploy_enabled: false
  socket_mode_enabled: false
  token_rotation_enabled: false
```

- You will land at the "Basic Information" page.
![App Basic Information](./images/004-slack.png)

- Update the Signing Secret in your `app-config.yaml` under the slack section (e.g `k8s/apps/overlays/rhdemo-netsentinel/netsentinel/app-config.yaml`), using the secret found in the Basic Information menu.
```
slack_config:
  slack_channel: "#netsentinel"
  slack_bot_token: "xoxb-783421362-783601639-8ybH7ZoB9nHsmBs2hw"
  slack_signing_secret: "fde84d1823293b0b806cc353bec"
```
- Navigate to OAuth & Permissions and click Install to NetSentinel under the "OAuth Tokens" section
![App Oauth Tokens](./images/005-slack.png)

- Click Allow and return to the OAuth & Permissions page to copy the "Bot User OAuth Token." Update the `slack_bot_token` field in `app-config.yaml`.
![App Bot User OAuth Token](./images/006-slack.png)


Redeploy the application and reboot the pods with the updated app config:

```
oc apply -k k8s/apps/overlays/rhdemo-netsentinel/netsentinel/
oc get pods -l app.kubernetes.io/name=netsentinel
oc delete pods -l app.kubernetes.io/name=netsentinel
```

- Navigate to slack "Event Subscriptions" page.
![Retry Event Subscriptions](./images/007-slack.png)

- Click on "Retry" if "Request URL" is not verified. Make sure NetSentinel app is fully up and running and you can hit this endpoint in browser.
![Verified Event Subscriptions](./images/008-slack.png)

- Your application is now configured and installed in your Slack workspace. Go to [NetSentinel Workspace](https://netsentinel.slack.com/) and confirm that the `NetSentinelRHDev` app appears under the **Apps** section.
![New app](./images/009-slack.png)

- Create a new channel (e.g., `#demo-channel`). The channel name can be anything; it doesn't have to match the `#netsentinel` name used in `app-config.yaml`.
![New Channel](./images/010-slack.png)

- Add `NetSentinelRHDev` to this new channel.
![Add Bot](./images/011-slack.png)
![Add Bot](./images/012-slack.png)

- You can now interact with the NetSentinel bot directly in the channel.
![Talk to bot](./images/013-slack.png)
