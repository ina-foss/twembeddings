from setuptools import setup

setup(name='twembeddings',
      version='0.1',
      description='event detection in tweets',
      url='https://github.com/ina-foss/twembeddings',
      author='Béatrice Mazoyer',
      author_email='beatrice.mazoyer [at] centralesupelec.fr',
      license='MIT',
      packages=['twembeddings'],
      install_requires=[
        "gensim==3.4.0",
        "numpy==1.17.2",
        "pandas==0.25.1",
        "pytorch-transformers==1.1.0",
        "scikit-learn==0.21.3",
        "scipy==1.3.1",
        "sentence-transformers==0.2.3",
        "tensorflow-gpu==2.3.1",
        "tensorflow-hub==0.5.0",
        "tensorflow-text==2.0.1",
        "tqdm==4.36.1",
        "unidecode",
        "twython",
        "pyyaml",
        "sparse_dot_mkl"
      ],
      zip_safe=False)
