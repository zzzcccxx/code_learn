# linux learn

#### 1、根目录为 ```/```

#### 2、ls命令

```
ls -a    # 显示全部文件
ls -l    # 以列表形式展示
ls -al
```

#### 3、创建和查看文件

```
touch test.txt    # 创建
cat test.txt    # 查看
```

#### 4、复制、移动、删除

```
cp [-r] 参数1 参数2    # -r表示递归复制文件夹，参数1被复制的地址，参数2为复制去的地址

mv 参数1 参数2    # 参数1为被移动的文件和文件夹，参数2为移动到的地址


rm [-r -f] 参数1 参数2 ...    # -r表示递归删除文件夹，-f表示强制删除，所有参数表示要删除的文件或文件夹
```

#### 5、输出

```
echo "hello world!"
echo `pwd`


echo "hello world!" > test.txt    # 覆盖写入
echo "hello world!" >> test.txt    # 追加写入


tail [-f -num] 参数    # -f表示持续跟踪，-num表示显示多少行，参数表示展示的文件路径
```

#### 6、root权限

```
su - root   # 切换到root用户，如果报错则先给root设置密码
su - 用户名
快捷键ctrl+d可以回退上一个用户，也可以使用exit退出
sudo 其他命令    # 只用获得认可的用户才可以使用sudo

# 给普通用户配置sudo，首先要先切换到root用户
visudo    # 会自动打开/etc/sudoers
# 在文件的最后添加：
用户名 ALL=(ALL) NOPASSWD:ALL  # 最后表示使用sudo无需输入密码
```

#### 7、用户和用户组

```
# 创建用户组，需要root用户执行
groupadd 用户组名
groupdel 用户组名


# 创建用户，旧版。新版不使用这个
useradd [-g -d] 用户名    # -g表示指定用户组，-d表示用户home路径
userdel [-r] 用户名


# 查看用户所属组
id 用户名


# 修改用户组
usermod -aG 用户组 用户名    # 将用户加入用户组
```

#### 8、文件权限

```
ls -l 展示的信息，最左侧为权限，然后是文件和文件夹所属用户，最后是所属用户组
最左侧的权限 前三位表示所属用户的权限，中间表示所属用户组所拥有权限，最后表示其他用户

# 修改权限，文件所属用户或root用户可以使用
chmod 777 test.txt
```

#### 9、软件安装

```
# centos：
yum
# ubuntu
apt [install | remove | search] 软件名称
```

#### 10、控制软件

```
systemct start|stop|status|enable|disable 服务名
```

#### 11、ip地址和主机名

```
# 查看ip,中间的inet就是ipv4的ip地址
ifconfig    # 127.0.0.1表示本机 0.0.0.0表示本机


# 主机名
hostname
#修改主机名
hostnamectl set-hostname 主机名，修改主机名（需要root）
```


