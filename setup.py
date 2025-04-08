from setuptools import setup, find_packages
from typing import List

# This is the function to get the requirements from the requirements.txt file
# '-e .' is a placeholder for editable installs, we don't need it in the list of requirements
def get_requirements(file_path: str) -> List[str]:
    """
    This function will return a list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

# Declaring variables for setup functions
setup(
    name="mlproject",
    version="0.0.1",
    author="Uma_Maheshwari",
    author_email="maheshwari.suroju1603@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt'),
    description="A small ML project"
    )