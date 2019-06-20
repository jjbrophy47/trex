from setuptools import setup

setup(name='sexee',
      version='0.1',
      description='svm explanations for tree ensembles',
      url='',
      author='Jonathan Brophy',
      author_email='jbrophy@cs.uoregon.edu',
      license='MIT',
      packages=['sexee', 'util'],
      zip_safe=False)

setup(name='util',
      version='0.1',
      description='project library',
      url='',
      author='Jonathan Brophy',
      author_email='jbrophy@cs.uoregon.edu',
      license='MIT',
      packages=['util'],
      zip_safe=False)
