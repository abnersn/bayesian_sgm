import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='bayesian_sgm',
    version='1.0.0',
    description='Um segmentador de cores bayesiano para a OpenCV.',
    author='Abner',
    author_email='abnersousanascimento@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/abnersn/bayesian_sgm',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.0",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: Portuguese (Brazilian)",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "opencv-python >= 3.1.0",
        "numpy >= 1.14.3"
    ]
)
