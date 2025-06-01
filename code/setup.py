from setuptools import setup

def read_version():
    with open("VERSION") as f:
        return f.readline().strip()

setup(
    version=read_version(),
)