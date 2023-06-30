from setuptools import setup, find_packages

setup(
    name='song_or_not',
    version='1.0.5',
    description='Detect if audio is song or speech',
    author='BCC Media',
    author_email='support@bcc.media',
    url='https://github.com/bcc-code/bccmedia-song-or-not',
    packages=find_packages(exclude=['tests', 'songs', 'speech']),
    package_data={
        'inference': ['*.pt'],
    },
    install_requires=["torch", "torchaudio"]
)
