from setuptools import setup

setup(name='twembeddings',
      version='0.2.0',
      description='event detection in tweets',
      url='https://github.com/ina-foss/twembeddings',
      author='Béatrice Mazoyer',
      author_email='beatrice.mazoyer@sciencespo.fr',
      license='MIT',
      packages=['twembeddings'],
      install_requires=[
        "gensim",
        "numpy",
        "pandas",
        "protobuf==3.20.3",
        "pytorch-transformers",
        "scikit-learn>=1.3",
        "scipy",
        "sentence-transformers",
        "tensorflow>=2.13.0",
        "tensorflow-text>=2.13.0",
        "tqdm",
        "unidecode",
        "twython",
        "pyyaml",
        "sparse_dot_mkl"
      ],
      zip_safe=False)
