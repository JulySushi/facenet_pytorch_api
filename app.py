from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask_wtf import FlaskForm
from wtforms import TextField, SubmitField, HiddenField,validators
# 用于将文件路径变为安全的路径
from werkzeug.utils import secure_filename
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:123456@8.134.24.224:3309/db_intershipf'
app.config['SECRET_KEY'] = '123456'
db = SQLAlchemy(app)
# 文件上传路径
app.config['UPLOAD_FOLDER'] = 'static/uploaddir/'


# 表tb_info
class tb_info(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    age = db.Column(db.Integer)
    timeRenew = db.Column(db.DateTime, default=datetime.now)

    def __init__(self, name, age):
        self.name = name
        self.age = age


class tb_image(db.Model):
    imageId = db.Column(db.Integer, primary_key=True)
    imageName = db.Column(db.String(255), nullable=False)  # 假设图片路径是必填的
    imagePath = db.Column(db.String(255), nullable=False)

    def __init__(self, imageName, imagePath):
        self.imageName = imageName
        self.imagePath = imagePath  # 这里假设 info 是一个 tb_info 对象，并且它的 id 属性已经被设置

class ContactForm(FlaskForm):
    id = HiddenField("id")
    name = TextField('name', [validators.DataRequired('请输入名字')])
    age = TextField('age', [validators.DataRequired('请输入年龄')])
    timeRenew = datetime.now()
    submit = SubmitField("Send")


# 首页
@app.route('/')
def upload_file():
    return render_template('index.html')


@app.route('/show_info', methods=['GET', 'POST'])
def show_info():
    if request.method=="POST":
        search = request.form['search']
        contacts = tb_info.query.filter(tb_info.name.like("%" + search + "%")).all()
        return render_template("show_info.html",contacts=contacts)
    else:
        return render_template("show_info.html", contacts=tb_info.query.all())


@app.route('/show_image')
def show_image():
    contacts = tb_image.query.all()
    return render_template("show_image.html", contacts=contacts)


# 添加信息
@app.route('/add', methods=['GET', 'POST'])
def add():
    form = ContactForm()
    return render_template("add.html", form=form)


@app.route('/add_info', methods=['GET', 'POST'])
def addDo():
    # 获取wtforms控件
    form1 = ContactForm()
    if request.method == "POST":
        # 如果验证失败
        if form1.validate() == False:
            flash("所有不能为空")
            return render_template("add.html", form=form1)
        else:
            # 将表单的值传回contact类
            contact = tb_info(
                form1.name.data,
                form1.age.data
            )

            try:
                # 数据库新添数据操作
                db.session.add(contact)
                # 执行操作
                db.session.commit()
                flash("添加成功")
                return redirect('/show_info')
            except Exception as e:
                flash("添加失败,原因：%s" % e)
                return render_template("add.html", form=form1)
    elif request.method == "GET":
        return render_template("add.html", form=form1)


@app.route("/delete/<int:id>")
def deleteDO(id):
    # 数据是否存在
    to_delete = tb_info.query.get_or_404(id)
    try:
        # 删除数据
        db.session.delete(to_delete)
        # 执行操作
        db.session.commit()
        flash("数据库信息删除成功")
        return redirect('/show_info')
    except Exception as e:
        flash("删除失败,原因：%s" % e)
        return redirect('/show_info')


@app.route("/deleteImage/<int:imageId>")
def deleteImageDO(imageId):
    # 数据是否存在
    to_delete = tb_image.query.get_or_404(imageId)
    try:
        # 删除数据
        db.session.delete(to_delete)
        # 执行操作
        db.session.commit()
        flash("删除成功")
        return redirect('/show_image')
    except Exception as e:
        flash("删除失败,原因：%s" % e)
        return redirect('/show_image')



# 修改页
@app.route('/update/<id>', methods=['GET', 'POST'])
def updateDO(id):
    # 查询数据
    tb_info_db = tb_info.query.get_or_404(id)
    form = ContactForm()
    if request.method == 'POST':
        if form.validate():
            try:
                # 更新数据库中的记录
                tb_info_db.name = form.name.data
                tb_info_db.age = form.age.data
                tb_info_db.timeRenew = form.timeRenew
                db.session.commit()
                flash('更新成功！')
                return redirect(url_for('show_info'))
            except Exception as e:
                flash("修改失败,原因：%s" % e)
                return render_template("update.html", form=form)
        else:
            flash("所有不能为空")
            return render_template("update.html", form=form)  # 显示错误信息或进行其他处理
    else:  # GET请求，显示表单
        form.name.data = tb_info_db.name
        form.age.data = tb_info_db.age
    return render_template("update.html", form=form, id=id)


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    # form = ImageForm()
    if request.method == 'POST':
        # 获取上传的文件
        selected_name = request.form.get('hito')

        f1 = request.files['file111']
        f2 = request.files['file222']
        f3 = request.files['file333']
        piece = []

        # 检查是否为空
        for f in [f1.filename, f2.filename, f3.filename]:
            if f.split('.')[-1] in ['jpg', 'png', 'jpeg']:
                piece.append(f)
        if len(piece) == 0:
            flash('请上传至少一张图片')
            return render_template('upload.html', contacts=tb_info.query.all())

        # 检查文件格式
        for f in piece:
            if f.split('.')[-1] not in ['jpg', 'png', 'jpeg']:
                flash('请上传jpg,png,jpeg格式的图片')
                return render_template('upload.html', contacts=tb_info.query.all())
        # 保存文件，join拼接路径
        classier_path = os.path.join(app.config['UPLOAD_FOLDER'], selected_name)

        if not os.path.exists(classier_path):
            os.makedirs(classier_path)

        file_paths = [
            (f1,os.path.join(classier_path, secure_filename(f1.filename))),
            (f2,os.path.join(classier_path, secure_filename(f2.filename))),
            (f3,os.path.join(classier_path, secure_filename(f3.filename)))
        ]
        for i, (file, path) in enumerate(file_paths):
            if len(file.filename):
                file.save(path)
                contact = tb_image(
                    selected_name,
                    file.filename
                )
                try:
                    # 数据库新添数据操作
                    db.session.add(contact)
                    # 执行操作
                    db.session.commit()
                except Exception as e:
                    flash("第%i添加失败,原因：%s" % i % e)
                    return render_template("upload.html", contacts=tb_info.query.all())
        flash("全部上传完成")
        return redirect('/show_image')
    elif request.method == 'GET':
        return render_template('upload.html', contacts=tb_info.query.all())

@app.route('/predict_image', methods=['GET', 'POST'])
def predict_image():
    if request.method == 'POST':
        # 获取上传的文件
        f = request.files['file111']
        if f.filename.split('.')[-1] not in ['jpg', 'png', 'jpeg']:
            flash('请上传jpg,png,jpeg格式的图片')
            return render_template('predict_image.html')
        # 保存文件，join拼接路径
        path = os.path.join('static/predict/', secure_filename(f.filename))
        f.save(path)
        data_dir = r'static/uploaddir'
        classes = os.listdir(data_dir)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('在该设备上运行: {}'.format(device))
        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=device
        )

        ImageResize = transforms.Compose([
            transforms.Resize((512, 512))
        ])
        trans = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization
        ])

        resnet = InceptionResnetV1(
            classify=True,
            #pretrained='vggface2', #"C:\Users\86188\.cache\torch\checkpoints"
            num_classes=len(classes)).to(device)
        resnet.load_state_dict(torch.load('static/models/model.pt'))
        # 加载图片
        resnet.eval()
        img = Image.open(path)
        img = ImageResize(img)
        mtcnn(img, save_path=path)
        img = Image.open(path)
        img = trans(img)
        resnet.classify = True
        with torch.no_grad():
            img_probs = resnet(img.unsqueeze(0))
        res = classes[int(img_probs.argmax())]
        os.remove(path)
        return render_template('predict_image.html', res=res)
    elif request.method == 'GET':
        return render_template('predict_image.html')


# @app.route('/predict_video', methods=['GET', 'POST'])
# def predict_video():
#     if request.method == 'POST':
#         # 获取上传的文件
#         f = request.files['file111']
#         if f.filename.split('.')[-1] != 'mp4':
#             flash('请上传mp4格式的视频')
#             return render_template('predict_video.html', contacts=tb_info.query.all())
#         # 保存文件，join拼接路径
#         path = os.path.join('static/predict/', secure_filename(f.filename))
#         f.save(path)
#
#         device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         print('在该设备上运行: {}'.format(device))
#         mtcnn = MTCNN(keep_all=True, device=device)
#         video = mmcv.VideoReader(path)
#         frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
#         frames_tracked = []
#         for i, frame in enumerate(frames):
#             print('\r当前帧: {}'.format(i + 1), end='')
#
#             # 检测人脸
#             boxes, _ = mtcnn.detect(frame)
#
#             # 绘制人脸框
#             frame_draw = frame.copy()
#             draw = ImageDraw.Draw(frame_draw)
#             for box in boxes:
#                 draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
#
#             # 添加到图像列表
#             frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
#         print('\n结束')
#         dim = frames_tracked[0].size
#         fourcc = cv2.VideoWriter_fourcc(*'FMP4')
#         video_tracked = cv2.VideoWriter(path, fourcc, 25.0, dim)
#         for frame in frames_tracked:
#             video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
#         video_tracked.release()
#
#         return render_template('video_Show.html', name=f.filename)
#     elif request.method == 'GET':
#         return render_template('predict_video.html',)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
