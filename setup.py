from distutils.core import setup

setup(
    name='bayesian_sgm',
    version='1.0.0',
    description='Um segmentador de cores bayesiano para a OpenCV.',
    author='Abner',
    author_email='abnersousanascimento@gmail.com',
    url='https://github.com/abnersn/bayesian_sgm',
    license='GNU',
    python_requires='>=3.5.5',
    packages=['bayesian_sgm'],
    install_requires=[
        'numpy>=1.14.3',
        'cv2>=3.1.0'
    ]
)
