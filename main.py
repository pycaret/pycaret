# main.py
# entry point for Dockerfile

from flask import Flask
main = Flask(__name__)

@app.route('/')
def main():
    from pycaret.utils import version
    return version

if __name__ == '__main__':
    main.run(host='0.0.0.0', debug = True)