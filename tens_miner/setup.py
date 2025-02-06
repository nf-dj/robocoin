from setuptools import setup, find_packages

setup(
    name="tens_miner",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "PyQt6",
        "pycryptodome",
    ],
    entry_points={
        'console_scripts': [
            'tens_miner=tens_miner.__main__:main',
        ],
    },
)