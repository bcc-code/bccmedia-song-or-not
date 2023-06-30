from distutils.core import setup

setup(
    name='BCC Media - Song or Not',
    version='1.0.2',
    description='Detect if audio is song or speech',
    author='BCC Media',
    author_email='support@bcc.media',
    url='https://github.com/bcc-code/bccmedia-song-or-not',
    packages=['classifier', 'inference'],
)
