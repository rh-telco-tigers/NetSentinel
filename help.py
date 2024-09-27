# help.py
def display_help():
    print("""
    Available commands:

    - python app/run.py          : Start the main application
    - python scripts/create_mock_data.py : Generate mock traffic data and publish to Kafka
    - python scripts/process_mock_data.py: Process mock traffic data from Kafka
    - python scripts/prediction_service.py: Run the prediction service to analyze processed traffic
    - python -m unittest discover : Run the unit tests for the project

    Use any of the above commands to run the appropriate task.
    """)

if __name__ == "__main__":
    display_help()
