if ! command -v yum &> /dev/null
then
    # We are building on musllinux (Alpine Linux),
    # where the default package manager is apk
	alias install_pkg="apk add"
else
    # We are building on manylinux (CentOS),
    # where the default package manager is yum
	alias install_pkg="yum install -y"
fi
install_pkg -y lapack-devel
install_pkg -y git git-all
install_pkg -y openssl-devel
export PATH=/usr/local/libexec/git-core:$PATH
gcc --version
g++ --version
cmake --version
cd /home
git clone https://gitlab.com/conradsnicta/armadillo-code.git
git clone https://github.com/RUrlus/carma.git --recursive # Do we need this?
cd /home/armadillo-code
cmake .
make install