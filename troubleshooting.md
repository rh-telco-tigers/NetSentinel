## Patch each kafkatopics, set it to null
```
for topic in $(oc get kafkatopics -n netsentinel -o jsonpath='{.items[*].metadata.name}'); do
  echo "Removing finalizers from $topic ..."
  oc patch kafkatopic "$topic" -n netsentinel --type=merge -p '{"metadata":{"finalizers":null}}'
done
```