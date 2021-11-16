import subprocess
import signal
import time
import os


def compiler_check():
    """
    A very hacky method by which we understand what compiler was used to 
    compile the user's Python.

    This is necessary because setuptools will use the compiler that compiled
    python -- even if the user specifies another one! -- for some of the
    compilation process
    """
    def convert_bin_to_bytestring(bin_file):
        MAX_BYTES = 45
        str_ = b''
        tmp = bin_file.read(MAX_BYTES) # can only read 45 bytes at a time
        while tmp:
            str_ += tmp
            tmp = bin_file.read(MAX_BYTES)
        return str_

    with open('tmp_output.txt', 'wb') as fout, \
        open('tmp_error.txt', 'wb') as ferr:
        # Spawn a python3 process, then kill it
        process = subprocess.Popen(['python3'], stdout=fout, stderr=ferr)
        time.sleep(1) # Need to wait for python3 to produce output
        process.send_signal(signal.SIGKILL)

    with open('tmp_output.txt', 'rb') as fout, open('tmp_error.txt', 'rb') as ferr:
        error = convert_bin_to_bytestring(ferr)
        if b'Apple' in error:
            raise Exception("Error: you're using a version of python compiled with Apple Clang, \
                which does not support OpenMP. Please use Anaconda Python or download LLVM Clang \
                and build python.")
        elif b'Clang' in error:
            compiler = 'clang'
        elif b'comamnd not found' in error:
            raise Exception("Please install python3")
        else:
            compiler = 'gcc'

    os.remove('tmp_output.txt')
    os.remove('tmp_error.txt')
    return compiler

if __name__ == '__main__':
    print(compiler_check())