

我的文章有个特点，不选自己想写的，专挑大家想看的。就算内容对我陌生，我也有自信先把它吃透，再让你听懂。






你看，有朋友就提议写写YOLO。

也巧，去年我还用过YOLO v5，今天再去官网一看，已经升到v8了。而且它在淡化YOLO版本，主打Ultralytics平台。

我感觉这个平台，能很好地解决图像领域的大部分问题。



一、Ultralytics平台
YOLO原本是一种公开的目标检测算法。它的优势是速度快，准确率还高。这很气人。一般情况下，两者是不能兼得的。

它之所以这么强，从它的名字中就能找到答案。YOLO的全称是You Only Look Once（你仅需看一遍）。关于它原理解读的介绍非常多，属于街边知识，我就不多说了。反正，你知道它很火就好了。基本上，从画面中找物体的技术方案，选YOLO就对了。






今年Ultralytics公司在YOLO之前版本基础上提出了v8版本。这个版本，更像是一个AI视觉处理平台，它不但可以做检测，还可以做分类、分割、跟踪，甚至姿态估计。






然而它的调用和二次开发，也很方便。这太气人了，它不但好用，而且易用。



二、操作和原理指南
Github地址：https://github.com/ultralytics

依旧提示大家去读ReadMe.md文件。我所说的，皆源于此。

基础环境要求：



Python >= 3.8
PyTorch >= 1.7
执行pip install ultralytics安装平台支持库，后面就可以操作了。



2.1 命令行操作
它有多易用呢？易用到你啥都不用动，安装完就能直接用。

找来一个图片，例如下面这图，我们命名叫bus.jpg，放到主目录下。






然后你在终端运行如下命令：



yolo predict model=yolov8n.pt source=bus.jpg
这表示使用yolo的yolov8n.pt模型对bus.jpg这张图进行预测。它会自动去下载yolov8n.pt模型文件。随后，控制台打印如下：



(yolo)C:\tfboy\yolo> yolo predict model=yolov8n.pt source=bus.jpg
……
image 1/1 C:\tfboy\yolo\bus.jpg: 640x480 4 persons, 1 bus, 1 stop sign, 94.7ms
Speed: 0.0ms preprocess, 94.7ms inference, 0.0ms postprocess per image at shape (1, 3, 640, 480)
Results saved to runs\detect\predict
最后一句说，结果保存到了runs\detect\predict目录。我们打开一看，发现了一张bus.jpg。






嗯，这就是目标检测的结果。看标记上，它识别出了person人，bus公交车，还有左上角的stop sign停车路标。我们发现person的识别，左右两边都是半个人，它也识别出来了。

最简单的用法就是这样，不用打开IDE（程序员写代码的程序）就能处理图片。

你甚至都不用准备图片，直接运行命令也行。因为我从源码文件engin\model.py中的predict函数中看到这么一段：



if source is None:
    source = ROOT / 'assets' if is_git_dir() else 'https://ultralytics.com/images/bus.jpg'
    LOGGER.warning(f"WARNING  'source' is missing. Using 'source={source}'.")
如果素材为空，它会使用一个预置的本地目录。如果这个坏了，它还会从网上下载一个图片，并日志提示你“没有图，我顶了一下”。

这绝对是业界良心！

但是，我们想要拿到具体的识别数据，或者还想尝试其他功能怎么办？

继续往下看。



2.2 用代码处理
代码调用也极其简单。



# 从平台库导入YOLO类
from ultralytics import YOLO
# 从模型文件构建model
model = YOLO("xx.pt")
# 对某张图片进行预测
results = model("bus.jpg")
# 打印识别结果
print(results) 
开头我们说过，YOLOv8功能强大，除了目标检测，还支持分割、追踪啥的。我也说过，它很易用。

现在就体现出来了。它执行所有功能的代码都是同一套，想换功能更换xx.pt模型文件就行。它的分析结果，也都存在同一个results结构中。



2.2.1 模型文件的调用
它有哪些模型文件呢？

名称	模型文件	家族
检测	yolov8n.pt	8n、8s、8m、8l、8x
分割	yolov8n-seg.pt	8n、8s、8m、8l、8x
分类	yolov8n-cls.pt	8n、8s、8m、8l、8x
姿态	yolov8n-pose.pt	8n、8s、8m、8l、8x
每一类模型，还搞出一个家族。这就好比是同一款衣服的不同尺码。尺码不同，受众也不同。我们拿检测来做个对比。

类型	准确度	耗时长	运算次数/秒
YOLOv8n	37.3	80.4	8.7
YOLOv8s	44.9	128.4	28.6
YOLOv8m	50.2	234.7	78.9
YOLOv8l	52.9	375.2	165.2
YOLOv8x	53.9	479.1	257.8
我们看到从8n到8x，虽然准确值提高了，但是时间也提高了6倍。

因为它的算法，让它在竞争者中显得又快又准。但是回到同一个算法内，其实还是有取舍的。

我们更换模型文件就可以体验不同的功能。下面是我用代码尝试了yolov8n-pose.pt和yolov8n-cls.pt这俩模型的效果。






姿态模型能检测人体的姿势动作。分割模型可以将识别到的物体从背景中切分出来。



2.2.2 返回结果的解读
我们使用它，肯定是想让它帮我们处理图像。像上面那样，它给原图画出一个框，没法融入到我们产品中。这像极了有钱人来你家大把炫富，然后说并不打算给你或者借你。

Ultralytics绝对不会这么做。它有好多种方式将结果给你！

首先，它那个results = model("bus.jpg")的results就包含着处理结果的数据。

其次，这是一个临时内存数据，它怕你不打印就丢了。因此又提供了一个持久化的保存方法。

看下面这段代码：



from ultralytics import YOLO
from PIL import Image
model = YOLO('yolov8n-seg.pt')
image = Image.open("bus.jpg")
results = model.predict(source=image, save=True, save_txt=True) 
这是另一种调用方式，我们标记为code-666。

与之前的区别是构建完model之后，调用了predict方法，参数里面有save_txt=True这项。这表示把数据结果保存到txt文本中。

其实我们也看到入参也有变化。先通过Image.open("bus.jpg")把图片包装一下，然后通过source=image传入。除此之外，数据源也支持文件夹或者摄像头。



# 识别来自文件夹的图像
results = model.predict(source="test/pics", ……) 
# 识别来自摄像头的图像
results = model.predict(source="0", ……)
不管哪种输入方式，这个results都很关键。想了解它的结构，你有3种途径：

上网搜，找解析文章或者官方文档。
直接print一下，看看它输出什么。
查看源码，了解其属性构成。
作为老程序员一般会直接看源码。通过按着ctrl键点击方法名，我定位到它是engine/results.py下的Results类。






瞬间，豁然开朗的感觉。我拣重要的，给大家解释下：

boxes: 检测出来物体的矩形框，就是目标检测的框。
masks: 检测出来的遮罩层，调用图像分割时，这项有数据。
keypoints: 检测出来的关键点，人体姿势估计时，身体的点就是这项。
names: 分类数据的名称，比如{0: 人，1: 狗}这类索引。
想看具体的数值，可以自己调用results[0].boxes或者results[0].masks打印一下。



2.2.3 简单的抠图示例
下面我们做一个抠图的示例。我们将bus.jpg采用yolov8n-seg.pt模型做一个物体分割。它会把图像中检测到的物体分割出来，并将结果保存到results中(code-666就是这项操作)。

然后，我们打印一下结果：



import numpy as np
pixel_xy = results[0].masks.xy[1]
points = np.array(pixel_xy, np.int32)
print(points)
results是一个支持批量图片的结果集，因为我们只有一张图像，所以取results[0]。
masks.xy是一张图里所有物体掩膜的轮廓坐标，我们只取一个，取索引为1的物体。
points的打印值是：



array([[113, 398],
       [111, 399],
       [106, 399],
...
       [150, 399],
       [148, 398]])
都是xy坐标点。我们可以把它画出来。



import cv2
input_image = cv2.imread('bus.jpg')
cv2.drawContours(input_image, [points], -1, (0, 255, 0), 2)
cv2.imwrite('output.jpg', input_image)





原来这是穿羊皮袄的帅哥。既然已经知道了他的位置，那么把圈里的像素提取出来，让他到济南的街头走一走，感受一下车让人，感受一下超然楼。






同样，你提取谁都可以，或者给整体换一个背景也行。因为物体我们都拿到了。



动图封面


在开源技术的加持下，这对程序员来说很简单。但是很多软件还收费。因为它们收的是电脑的使用费。

不要高兴太早，这个模型是受限的。

YOLO是一种算法，它就好比是一种思想，本身啥也干不了。

上面的那些落地的模型文件仅仅是一个示例，是YOLOv8针对COCO数据集（一个很好的计算机视觉数据集）训练生成的。

这个数据集中有很多类物体：






你可以通过results[0].names直接打印出来。

也就是说，它只认识上面列举的80种物体。你拿一个它没见过的，比如请找出图中的老济南把子肉，它肯定识别不出来。需要你去训练数据集之外目标。

接下来就讲讲训练的事情。



2.3 训练数据
训练数据的操作依然很友好。只需要准备好训练集，然后启动训练就可以了。

这次我打算训练目标检测，来检测游戏角色的血量条。就是下面这样：






这有什么用呢？起码可以让我知道哪里出现活物了。

我们准备一些图片素材。条件好就多准备，条件差就少准备，起码得200张吧。






准备好素材后，下面就开始标记。



2.3.1 标记数据
我们使用labelImg进行标记。

首先pip3 install labelImg安装它。然后，在终端输入labelImg就能启动它。如果有问题，可以下载源码启动。






你可以把它当成一个正常的软件来用。



动图封面


一张图一张图地标记。标记完了之后，会产生一些文件。






classes.txt是总分类，一个分类占一行。另外，每张图片还会配一个同名的txt文件。里面记录了有几个框，都是什么类型，在哪个位置。

以0 0.386605 0.149837 0.091455 0.063518举例：



第1个数字0表示物体的类别标签。
第2个数字0.386605表示边界框左上角的X坐标相对于图像宽度的比例。
第3个数字0.149837表示边界框左上角的Y坐标相对于图像高度的比例。
第4个数字0.091455表示边界框的宽度相对于图像宽度的比例。
第5个数字0.063518表示边界框的高度相对于图像高度的比例。
它们用比例表示位置，这样可以实现10×10像素和1000×1000像素一样处理。

标记完成之后，我们需要把众多的.png和.txt按照固定的格式整理好。



datasets
  |--game
    |--images
      |--train
        |--xx1.png
      |--val
        |--xx2.png
    |--labels
      |--train
        |--xx1.txt
      |--val
        |--xx2.txt
下一步就是训练。



2.3.2 训练数据
yolo提供很多种标记和训练方式。我选择的是比较传统的一种。

数据准备好了，我们需要给数据做个索引，告诉框架在哪儿、有啥。新建一个game.yaml，内容如下：



# 训练集、验证集位置
train: game/images/train/
val: game/images/val/

# 几个分类
nc: 1

# 分类名称
names: ['blood']
然后就能训练了。可以采用命令行模式：



(yolo)C:……\yolo> yolo task=detect mode=train model=yolov8n.pt data=game.yaml epochs=300
也可以采用代码模式:



from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.train(data="game.yaml", epochs=300)
都很简单。全都是在yolov8n.pt基础上进行训练，只要你标记的好，结果肯定也是很快很准确。

训练结束以后，会在runs\detect\train下生成训练结果，有很多东西。






args.yaml是训练的参数。你填的那点参数根本不够，其他都是默认的，这里面有记录。如果你想改，可以训练时就指定。

我们最期待的还是weights里面的俩文件，一个best.pt是效果最好的权重，另一个是last.pt是最后一个权重。最后一个不一定是最好的，但是训练一通，最后一个不留着，用户会觉得吃亏了。所以保留俩。

results.png是训练历史记录。






我取一点给大家看，其实从50轮开始就平稳了。我训练了300轮有些多余。从这个图中也可以看出，最后一个不一定是准确率最高的。

下面试试效果吧。



2.3.3 推理预测
为了方便调用，我们把best.pt复制到项目根目录下。用法跟上面一样，只不过模型换成best.pt。

去识别一个文件夹下的图片：



from ultralytics import YOLO
model = YOLO("best.pt")
results = model.predict(source="game/images/val", save=True)
结果如下：






图片有点小，我们来搞一个视频看看，这次用命令行：



(yolo)C:……\yolo> yolo task=detect mode=predict model=best.pt source="game.mp4"
它会一帧一帧地处理：



动图封面


最后，输出到runs\detect\predict2，我们打开它看看效果：



动图封面


其实，实时摄像头或者视频流也能做到，只不过就是换个source来源的问题。

至于这个识别能做什么？肯定是有想象空间的，比如不遮挡弹幕、焦点跟踪等。



三、更广阔的应用
YOLOv8支持其他平台格式的导出。也就是说它的产物，可以跨平台、跨终端。






torchscript和tf.js可以在浏览器上跑。tflite可以在Android和Ios上运行。它甚至也能在飞桨平台运行。

代码就3行：



from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='tfjs')
看完赶紧尝试，起飞吧，少年！



四、参考
官网 https://ultralytics.com

文档 https://docs.ultralytics.com

GitHub https://github.com/ultralytics/ultralytics

labelImg https://github.com/HumanSignal/

LabelImg:

LabelImg 是一个开源的图像标注工具，支持矩形、多边形和曲线的标注。它使用 PyQt 编写，易于使用。
GitHub链接: LabelImg
VGG Image Annotator (VIA):

VIA 是一个功能强大的图像标注工具，支持多种标注形状，包括矩形、多边形、点等。它是一个基于浏览器的工具，不需要安装。
网站链接: VGG Image Annotator
RectLabel:

RectLabel 是一个 macOS 上的图像标注工具，支持矩形、椭圆、多边形等标注。它还提供了一些便捷的功能，适用于开发者和研究人员。
网站链接: RectLabel
Supervisely:

Supervisely 是一个面向计算机视觉任务的平台，提供图像标注、数据集管理等功能。它支持多种标注类型，包括矩形、多边形、分割等。
网站链接: Supervisely