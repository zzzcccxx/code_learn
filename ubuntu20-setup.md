# 从零开始设置ubuntu20

### 添加用户
```adduser [name]```

### 使用root安装vim
```
apt update
apt install vim
```
### 使用vim编辑/etc/sudoers 给用户添加sudo权限
在文件最后添加```[name] ALL=(ALL) NOPASSWD:ALL```

### 切换用户
``` su - [name] ```

### 安装docker
```
# https://zhuanlan.zhihu.com/p/143156163

# 先卸载残留的旧docker版本，若无旧版可跳过
1.常归删除操作
sudo apt-get autoremove docker docker-ce docker-engine docker.io containerd runc
 
2. 删除docker其他没有没有卸载
dpkg -l | grep docker
dpkg -l |grep ^rc|awk ‘{print $2}’ |sudo xargs dpkg -P # 删除无用的相关的配置文件
 
3.卸载没有删除的docker相关插件(结合自己电脑的实际情况)
sudo apt-get autoremove docker-ce-*
 
4.删除docker的相关配置&目录
sudo rm -rf /etc/systemd/system/docker.service.d
sudo rm -rf /var/lib/docker
 
5.确定docker卸载完毕
docker --version


# 开始安装
# 事前准备
sudo apt update
sudo apt install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# 安装最新版
sudo apt update
apt list -a docker-ce   # 查看都有哪些版本
sudo apt install docker-ce docker-ce-cli containerd.io
# sudo apt install docker-ce=<VERSION> docker-ce-cli=<VERSION> containerd.io  可安装指定版本

# 查看docker状态
sudo systemctl status docker

# 启动服务、开机自启、查看状态
sudo systemctl start docker
sudo systemctl enable docker
sudo systemctl status docker

# 最终看成果
docker version
```


### 安装miniconda
```
# 切换root，并安装在/opt下
cd /opt  #软件包下载位置
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh #镜像下载

# 想为哪个用户安装就su - [哪个用户]
bash Miniconda3-latest-Linux-x86_64.sh #安装程序
注意好安装路径


# 安装后需要重新启动才可以使用conda命令，若仍无法解决，则
sudo vim ~/.bashrc

# 并在bashrc最后添加
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/$你的用户名$/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/$你的用户名$/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/$你的用户名$/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/$你的用户名$/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

```




















