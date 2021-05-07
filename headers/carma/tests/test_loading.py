import numpy as np
import test_carma as carma

# A canary test is a trivial test for testing the environment
def test_canary():
    print("Python running with")

    import site
    print('site.USER_SITE=', site.USER_SITE)

    import sys
    print('sys.path=', sys.path)

if __name__ == '__main__':
    test_canary()
