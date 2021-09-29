from setuptools import setup, find_packages

setup(name='myutils',
      version='0.0.20',
      # package_dir = {'': 'myutils'}, # if there was a folder "myutils" in the folder
      packages = find_packages(),
      install_requires=[
          "numpy",
          "matplotlib",
          "pytest-shutil",
          "scikit-image",
          "scipy",
          "opencv-python",
          "opencv-rolling-ball",
	  ###"csbdeep @ git+https://{lizard24}@github.com/lizard24/mycsbdeep.git",
	  "twilio",
	  "readlif",
	  "read-roi"
      ]
      # dependency_links=['http://github.com/user/repo/tarball/master#egg=package-1.0']
      )
