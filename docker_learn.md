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

#### 2、构建Dockerfile

```
# commit构建镜像，基于已有的镜像来构建

docker commit -a "作者名" -m "备注信息" 已有的容器id/容器名 新的镜像名称:版本号   # 版本号若写latest表默认
```

```
# Dockefile
# 1. 先指定当前镜像的基础镜像
FROM openjdk:8

# 2. 描述镜像作者和联系方式（可选）
MAINTAINER wolfcode<liugang@wolfcode.cn>

# 3. 镜像标签信息（可选）
LABEL version="1.0"
LABEL description="这是我的第一个镜像"

# 4. 环境变量设置
ENV JAVA_ENV dev
ENV APP_NAME test-dockerfile 
# ENV JAVA_ENV=dev APP_NAME=test-dockerfile

# 5. 在构建镜像时当即使用命令
RUN ls -al

# 6. 将主机中的指定文件复制到容器的目标位置，和cp一样。前面是主机的位置，后面是容器的位置
ADD /tmp/index.html /www/server

# 7. 设置容器中的工作目录，如果目录不存在则创建
WORKDIR /app
# 在设置完目录后紧跟着就打印目录
RUN pwd

# 8. 镜像数据卷绑定，将主机中的指定目录挂载到容器中.执行下面指令则会在容器中生成一个与宿主机相同的文件夹，修改也会同步变化。即此文件夹不隔离
VOLUME ["www/wolfcode.cn"]

# 9. 设置容器启动后要暴露的端口。下面指令表示会将容器的8080端口暴露，有可能使用也有可能不用，使用还要使用-p来指定才可以访问。
EXPOSE 8080

# 10. CMD 和 ENTRYPOINT 选择其一即可。在语法中两者相同，但ENTRYPOINT会将bash中的命令覆盖掉
# CMD ["sh", "-c", "ping 127.0.0.1"]
CMD ping 127.0.0.1
```

#### 3、创建docker

```
# 先根据上述的Dockerfile和自己的内容进行设定。最后为Dockerfile的文件位置，若在当前目录就写.即可
docker build -t 自定义镜像名称:版本号 .
```

#### 4、docker上传

```
详情看b站教学
```






