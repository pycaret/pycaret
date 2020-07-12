# main.py
# entry point for Dockerfile

def main():

    from pycaret.utils import version
    print('Success')
    return version()
