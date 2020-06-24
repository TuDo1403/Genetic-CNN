from setuptools import setup, find_packages

setup(
    name='source',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'click', 'numpy', 'keras', 'tensorflow'
    ],
    entry_points='''
        [console_scripts]
        arch-search=cnn_arch_search:cnn_model
    ''',
    include_package_data=True
)
