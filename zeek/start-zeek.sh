#!/bin/bash

# Start Zeek using zeekctl
/usr/local/zeek/bin/zeekctl deploy
/usr/local/zeek/bin/zeekctl config
/usr/local/zeek/bin/zeekctl diag 
/usr/local/zeek/bin/zeekctl nodes
/usr/local/zeek/bin/zeekctl df
/usr/local/zeek/bin/zeekctl status

# Keep the container running and output logs to stdout
tail -f /dev/null
