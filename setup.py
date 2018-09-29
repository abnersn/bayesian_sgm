from distutils.core import setup

setup(
    name='bayesian_sgm',
    version='1.0.0',
    description='Um segmentador de cores bayesiano para a OpenCV.',
    author='Abner',
    author_email='abnersousanascimento@gmail.com',
    url='https://github.com/abnersn/bayesian_sgm',
    packages=['bayesian_sgm'],
    install_requires=[
        'numpy',
        'cv2'
    ]
)
