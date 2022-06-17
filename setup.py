from setuptools import setup

setup(name='twembeddings',
      version='0.1',
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
        "pytorch-transformers",
        "scikit-learn",
        "scipy",
        "sentence-transformers",
        "tensorflow-gpu",
        "tensorflow-hub",
        "tensorflow-text",
        "tqdm",
        "unidecode",
        "twython",
        "pyyaml",
        "sparse_dot_mkl"
      ],
      zip_safe=False)
