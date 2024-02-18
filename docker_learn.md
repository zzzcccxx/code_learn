# docker learn

### 基础命令

#### 1、安装

```
# 查看docker是否连接
docker version    # 需要有client和server，若没server表示docker没有启动

# 启动docker的server
service docker start

# 开机启动docker
systemctl enable docker
```

#### 2、查看命令

```
docker images    # 查看镜像
docker ps    # 查看运行中的容器
docker ps -a    # 查看所有容器
docker stats 容器id/容器名    # 查看容器状态
docker logs 容器id/容器名    # 查看此docker的所有日志，可使用--help来查看实时监听和查看多少行
docker inspect 容器id/容器名  # 查看具体信息
```

#### 3、搜索、拉取、删除镜像（镜像->安装包）

```
docker search xxxxxx
docker pull 名称全称
docker pull openjdk:8   # 指定8版本
docker rmi 容器id/容器名称   # 删除镜像
docker rm 容器id/容器名称   # 删除容器
```

#### 4、运行（将安装包安装并运行）和停止

```
docker run 镜像名称
docker run -d -p 本机端口:容器端口 镜像名称    # -d为后端运行
docker run -d -P    # 将本机的随机端口和容器的所有端口进行连接
docker run -d -p 本机端口:容器端口 --name my_dockername 镜像名称
docker run -d -P 本机端口:容器端口 --name my_dockername --restart on-failure:3 镜像名称    # 三次内挂掉后重启，可选择always即总是在挂掉后重启
docker run -d -P 80:80 --name my_docker -e PYTHON_ENV=dev -e JAVA_VM=G1 镜像名称    # -e为指定容器的环境变量
#限制资源启动
docker run -d -rm 8m 镜像名    # 限定内存最多使用8m
docker run -d --rm -m 8m --cpus 0.8 镜像名    # 内存最多使用8m，cpu最多使用0.8个

docker stop 容器id/容器名称
docker start 容器id/容器名称


# 测试时运行完退出
docker run -d -P --rm 镜像名称
```

#### 5、执行命令

```
docker exec -it 容器名称 要执行的命令    # exec表示要执行命令，-it表示在容器开一个终端来运行


# 进入容器内部
docker exec -it 容器名 /bin/bash    # 在容器中执行/bin/bash命令，使用exit退出
```



### 进阶

#### 1、数据卷

```
# 查看数据卷
docker volume ls

# 匿名绑定数据卷，但容器关闭数据卷删除
docker run --rm -d -p 80:80 --name docker_volume -v 容器内的目录 镜像名称
docker inspect 容器id/容器名  # 查看具体信息，找到mounnts

# 具名绑定，其中本机目录不存在则创建
docker run --rm -d -p 80:80 --name docker_volume -v 本机目录:容器内的目录 镜像名称


```


