import sys
import numpy as np

sys.path.insert(0, 'evoman')


def rankI_evaluation():
    pass


def rankII_evaluation():
    pass


def main(path):
    # loads file with the best solution for testing
    bsol = np.loadtxt(path)
    print('\n RUNNING SAVED BEST SOLUTION \n')
    print(bsol.shape)
    #env.update_parameter('speed', 'normal')
    #evaluate([bsol])
    #sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Requires the path to the best genome .txt file ...")
        exit(0)
    main(sys.argv[1])
