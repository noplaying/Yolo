# yolov5
需求环境(只是训练的机器环境,用插件的环境没有任何要求)
1. win10或者win11 64位系统. 其他系统没测试过,不一定能行. 系统最好最新版本. (必须)
2. 最好有nvidia显卡,并且显卡驱动更新到最新的版本,最好是30系列以上. 如果没有的话,训练效率会很低下. (可选,但是还是强烈建议)
3. 全局科学上网.因为很多安装包都需要到外网下载.(必须)


安装步骤
1.  安装python 3.8以上的版本.  
    下载地址 https://www.python.org/downloads/
    安装时,记得勾选添加路径到系统环境变量.其他可以默认. 如果忘记了勾选添加路径到环境变量,那需要手动添加环境路径.
    方法是打开windows环境变量,在系统变量的Path中添加python的路径. 比如我这里是
    C:\Users\Administrator\AppData\Local\Programs\Python\Python311
    C:\Users\Administrator\AppData\Local\Programs\Python\Python311\Scripts
    你自己安装时,会稍有不同,根据你的系统用户名和python的版本会不同.
    另外最好自己再手动加一个环境变量,避免执行py脚本时,产生cache文件夹
    PYTHONDONTWRITEBYTECODE=1

2. 安装cuda (如果需求环境的条件2不满足,那这个步骤可以略过)
    下载地址 https://developer.nvidia.com/cuda-downloads
    Operating System 选windows.  Architecture选 x86_64.  Version win10和win11对应的10和11版本. Installer Type选exe(local)
    安装要选择自定义安装,并且只勾选cuda. 其他一律不选(重要!!!)
    
3. 安装PyTorch.  
    下载地址 https://pytorch.org/get-started/locally  
    PyTorch Build选Stable版本.  Your OS选windows. Package 选Pip. Language选Python. 
    Compute Platform选最新的cuda.比如我这里是cuda11.8(如果需求环境的条件2不满足,那么这里选CPU)
    之后下面的Run this Command会生成一个命令. 复制这个命令,打开命令行粘贴进行运行即可. 
    比如我这里是pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

4. 安装yolo  注意yolo路径里,不要有任何非英文字符.
    直接用我下载好的yolo.rar即可. 放在任意目录. 建议放根目录. 我这里是放在了e盘.
    解压yolo.rar到E盘根目录. 解压后的目录形式是这样的 e:/yolo/yolov5_7.0
    命令行进入到yolov5_7.0这个目录下,然后执行以下命令 
    pip install -r requirements.txt

5. 安装git
    下载地址 https://git-scm.com/download/win
     选择Standalone Installer下的64位安装包

6. 测试识别
    命令行进入yolov5_7.0目录,然后执行以下命令
    python detect.py --weights yolov5s.pt --source data/images/bus.jpg
    最后会在runs/detect目录下生成识别后的图像.

7. 测试训练 使用coco128来测试
    命令行进入yolov5_7.0目录,然后执行以下命令
    python train.py --weights yolov5s.pt --epochs 300 --batch-size 16 --workers 8 --data ../datasets/coco128/coco128.yaml
    成功后,会在runs/train目录下生成训练后的模型和标记的图片等.

8. 测试模型格式转换 使用yolov5s.pt来测试
    命令行进入yolov5_7.0目录,然后执行以下命令
    python export.py --weights yolov5s.pt --simplify --include onnx
	成功后,会在yolov5_7.0目录下生成yolov5s.onnx文件.