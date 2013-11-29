import sys

from setuptools import setup, find_packages


stable = [l.split('#')[0].strip()
          for l in open('requirements.txt').readlines()
          if not l.startswith('#') and not l.startswith('-e')]

install_requires = stable + ['pminifier']
tests_require = ['mock', 'coverage']
lint_requires = ['pep8', 'pyflakes']
setup_requires=[]

if 'nosetests' in sys.argv[1:]:
    setup_requires.append('nose')
    setup_requires.append('nose-testconfig')

setup(
    name='mage',
    packages = find_packages(),
    version='1.0',
    url='https://github.com/Parsely/mage',
    install_requires=install_requires,
    tests_require=tests_require,
    setup_requires=setup_requires,
    extras_require={
        'test': tests_require,
        'all': install_requires + tests_require,
        'lint': lint_requires,
    },
    dependency_links =[
        # TODO: the below links don't install, it seems due to git+ssh not being supported.
        # no good solutions online, so for now we'll just need to include all git dependencies
        # in every project that needs them
        #'git+ssh://git@github.com/Parsely/parselyutils.git#egg=ParselyUtils-dev',
        #'git+ssh://git@github.com/Parsely/hll.git#egg=hll-dev',
        'https://github.com/Parsely/pminifier/tarball/master#egg=PMinifier-dev',
        ]
    )
