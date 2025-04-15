# generate-secrets.sh

# Generate random secrets
SESSION_SECRET=$(LC_CTYPE=C openssl rand -base64 32)
NEXTAUTH_SECRET=$(LC_CTYPE=C openssl rand -base64 32)

# Create a secrets file for Kustomize
cat <<EOF > console-ui-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: console-ui-secrets
  namespace: my-project
type: Opaque
data:
  SESSION_SECRET: $(echo -n "$SESSION_SECRET" | base64)
  NEXTAUTH_SECRET: $(echo -n "$NEXTAUTH_SECRET" | base64)
EOF

echo "Generated SESSION_SECRET and NEXTAUTH_SECRET and saved to console-ui-secrets.yaml"


## References
# https://docs.redhat.com/en/documentation/red_hat_streams_for_apache_kafka/2.7/html/using_the_streams_for_apache_kafka_console/proc-connecting-the-console-str#proc-connecting-the-console-str