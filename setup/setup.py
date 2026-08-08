from setuptools import setup, find_packages

setup(
    name="cgdit",
    version="0.0.1",
    author="Serein Zhong",
    author_email="zhongrs25@mails.tsinghua.edu.cn",
    packages=find_packages(include=["cgdit", "cgdit.*"]),
)