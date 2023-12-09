import os, sys
sys.path.append("..")
from src.data.utils import *
def main():
    print (os.path.basename(__file__))
    script_log (os.path.basename(__file__))
if __name__ == '__main__':
    main()