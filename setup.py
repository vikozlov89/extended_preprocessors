from distutils.core import setup
setup(
  name = 'extended_preprocessors',         # How you named your package folder (MyLib)
  packages = ['extended_preprocessors'],   # Chose the same as "name"
  version = '0.0.1',      # Start with a small number and increase it with every change you make
  license='gpl-3.0',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'A set of useful preprocessors for machine learning.',   # Give a short description about your library
  author = 'Vladislav Kozlov',                   # Type in your name
  author_email = 'vlad.kozlov.ds@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/vikozlov89/extended_preprocessors',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['ML', 'preprocessing'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'scikit_learn',
          'numpy',
          'pandas'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: ML specialists',      # Define that your audience are developers
    'Topic :: ML :: Preprocessors',
    'License :: OSI Approved :: GNU General Public License v3.0',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
)