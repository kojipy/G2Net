# 不足しているライブラリのインストール
pip3 install -q nnAudio
pip3 uninstall albumentations -y
pip3 install albumentations==0.5.2
pip3 show albumentations
pip3 install black
pip3 install -U flake8

# gitにログイン
git config --global user.email "ryu12kojiro@gmail.com"
git config --global user.name "sakamaki"