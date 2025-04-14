__author__ = "Yizhuo Wu, Chang Gao, Ang Li"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

from steps import classify
from project import Project

if __name__ == '__main__':
    proj = Project()
    if proj.step == 'classify':
        print("####################################################################################################")
        print("# Step: Classification                                                                             #")
        print("####################################################################################################")
        classify.main(proj)
    else:
        raise ValueError(f"The step '{proj.step}' is not supported.")