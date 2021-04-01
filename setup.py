from setuptools import setup, find_packages

setup(name='myutils',
      # package_dir = {'': 'myutils'}, # if there was a folder "myutils" in the folder
      packages = find_packages(),
      install_requires=[
          "numpy",
          "matplotlib",
          "shutil",
          "random",
          "scikit-image",
          "scipy",
          "open-cv",
          "opencv-rolling-ball",
	  "csbdeep @ git+https://{lizard24}@github.com/lizard24/mycsbdeep.git"
      ],
      dependency_links=['http://github.com/user/repo/tarball/master#egg=package-1.0']
      )
