from setuptools import setup, find_packages

setup(
    name="tens_miner",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pycryptodome",
        "PyQt6",
        "python-bitcoinlib",
    ],
    entry_points={
        "console_scripts": [
            "tens_miner=tens_miner.__main__:main",
        ],
    },
)
