import subprocess
import signal
import time
import os


def compiler_check():
    def convert_bin_to_bytestring(bin_file):
        str_ = b''
        tmp = bin_file.read(45)
        while tmp:
            str_ += tmp
            tmp = bin_file.read(45)
        return str_

    with open('tmp_output.txt', 'wb') as fout, open('tmp_error.txt', 'wb') as ferr:
        process = subprocess.Popen(['python3'], stdout=fout, stderr=ferr)
        time.sleep(1)
        process.send_signal(signal.SIGKILL)

    with open('tmp_output.txt', 'rb') as fout, open('tmp_error.txt', 'rb') as ferr:
        error = convert_bin_to_bytestring(ferr)
        if b'Apple' in error:
            raise Exception("Error: you're using a version of python compiled with Apple Clang, \
                which does not support OpenMP. Please use Anaconda Python or download LLVM Clang \
                and build python.")
        elif b'Clang' in error:
            compiler = 'clang'
        else:
            compiler = 'gcc'

    os.remove('tmp_output.txt')
    os.remove('tmp_error.txt')
    return compiler

if __name__ == '__main__':
    print(compiler_check())