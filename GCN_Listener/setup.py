import setuptools

# Build Author list
authors = {
    "Parth Patel": "parth.patel@anl.gov",
    "Zilinghan Li": "zilinghan.li@anl.gov",
    "Eliu Huerta": "elihu@anl.gov",
    "Victoria Tiki": "victoria.t.tiki@gmail.com",
    "Haochen Pan": "haochenpan@uchicago.edu",
}
AUTHOR = ""
for i, (k, v) in enumerate(authors.items()):
    if i > 0:
        AUTHOR += ", "
    AUTHOR += f"{k} <{v}>"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mma_gcn",
    version="1.0.0",
    author=AUTHOR,
    description="An open-source package for listening to GCN alerts and processing them",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/parth7stark/MMA_RadioWave/tree/main/GCN_Listener",
    project_urls={
        "Bug Tracker": "https://github.com/parth7stark/MMA_RadioWave/tree/main/GCN_Listener/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        # "numpy",  #installing specific version after pip install .
        # "torch",  #installing specific version after pip install .
        "omegaconf",
        "diaspora-event-sdk[kafka-python]",
        # "lalsuite", installing using conda (refer apptainer defination file) before pip install . setup.py
        #"h5py",
        #"scipy",
        "matplotlib",
        "voevent-parse",
        "gcn-kafka",

    ],
    extras_require={
        "examples": [
            "tqdm", # used by run_detector.py
        ],
    },
   
)
