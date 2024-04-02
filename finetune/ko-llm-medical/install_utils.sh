# Install tools
sudo yum -y install ncurses-devel tmux htop libdrm-devel

# Install nvtop
git clone https://github.com/Syllo/nvtop.git
mkdir -p nvtop/build && cd nvtop/build
cmake .. -DNVIDIA_SUPPORT=ON -DAMDGPU_SUPPORT=OFF
make
sudo make install