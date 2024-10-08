+----------------+      +----------------+      +----------------+      +----------------+
|  Scanning Tool | ---> |  AMQ Streams   | ---> | Predictive     | ---> |  Database      |
| (Mock Data)    |      |  (Kafka Topics)|      | Model Service  |      | (Raw & Processed|
+----------------+      +----------------+      +----------------+      |  Traffic Data) |
                                                                         +----------------+
                                                                              |
                                                                              v
                                                                   +--------------------------+
                                                                   |   Generative Model (LLM) |
                                                                   |   / Slack Integration    |
                                                                   +--------------------------+
