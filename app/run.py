# app/run.py

from . import create_app

app = create_app()

if __name__ == "__main__":
    api_config = app.config.get('API_CONFIG', {})
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 5000)
    debug = api_config.get('debug', False)
    app.run(host=host, port=port, debug=debug)
