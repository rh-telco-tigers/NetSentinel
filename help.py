# help.py
def display_help():
    print("""
    Available commands:

    - python app/run.py          : Start the main application
    - python services/create_mock_data.py : Generate mock traffic data and publish to Kafka
    - python services/process_mock_data.py: Process mock traffic data from Kafka
    - python services/predict_and_store.py: Run the prediction service to analyze processed traffic

    Use any of the above commands to run the appropriate task.
    """)

if __name__ == "__main__":
    display_help()
