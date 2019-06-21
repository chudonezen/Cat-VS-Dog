from tensorflow import keras
import numpy as np
import time
import glob
#category=['Cat','Dog']
import labelRead
category={'n02085620':'奇瓦瓦',
          'n02085782':'日本猎犬',
          'n02085936':'马尔济斯犬',
          'n02086079': '狮子狗',
          'n02086240': '西施犬',
          'n02086646': '布莱尼姆猎犬',
          'n02086910': '巴比狗',
          'n02087046': '玩具犬',
          'n02087394': '罗得西亚长背猎狗',
          'n02088094': '阿富汗猎犬',
          'n02088238': '猎犬',
          'n02088364': '比格犬, 猎兔犬',
          'n02088466': '侦探犬',
          'n02088632': '蓝色快狗',
          'n02089078': '黑褐猎浣熊犬',
          'n02089867': '沃克猎犬',
          'n02089973': '英国猎狐犬',
          'n02090379': '美洲赤狗',
          'n02090622': '俄罗斯猎狼犬',
          'n02090721': '爱尔兰猎狼犬',
          'n02091032': '意大利灰狗',
          'n02091134': '惠比特犬',
          'n02091244': '依比沙猎犬',
          'n02091467': '挪威猎犬',
          'n02091635': '奥达猎犬, 水獭猎犬',
          'n02091831': '沙克犬, 瞪羚猎犬',
          'n02092002': '苏格兰猎鹿犬, 猎鹿犬',
          'n02092339': '威玛猎犬',
          'n02093256': '斯塔福德郡牛头梗, 斯塔福德郡斗牛梗',
          'n02093428': '美国斯塔福德郡梗, 美国比特斗牛梗, 斗牛梗',
          'n02093647': '贝德灵顿梗',
          'n02093754': '边境梗',
          'n02093859': '凯丽蓝梗',
          'n02093991': '爱尔兰梗',
          'n02094114': '诺福克梗',
          'n02094258': '诺维奇梗',
          'n02094433': '约克郡梗',
          'n02095314': '刚毛猎狐梗',
          'n02095570': '莱克兰梗',
          'n02095889': '锡利哈姆梗',
          'n02096051': '艾尔谷犬',
          'n02096177': '凯恩梗',
          'n02096294':'澳大利亚梗',
          'n02096437': '丹迪丁蒙梗',
          'n02096585': '波士顿梗',
          'n02097047': '迷你雪纳瑞犬',
          'n02097130': '巨型雪纳瑞犬',
          'n02097209': '标准雪纳瑞犬',
          'n02097298': '苏格兰梗',
          'n02097474': '西藏梗, 菊花狗',
          'n02097658': '丝毛梗',
          'n02098105': '软毛麦色梗',
          'n02098286': '西高地白梗',
          'n02098413': '拉萨阿普索犬',
          'n02099267': '平毛寻回犬',
          'n02099429': '卷毛寻回犬',
          'n02099601': '金毛猎犬',
          'n02099712': '拉布拉多猎犬',
          'n02099849': '乞沙比克猎犬',
          'n02100236': '德国短毛猎犬',
          'n02100583': '维兹拉犬',
          'n02100735': '英国谍犬',
          'n02100877': '爱尔兰雪达犬, 红色猎犬',
          'n02101006': '戈登雪达犬',
          'n02101388': '布列塔尼犬猎犬',
          'n02101556': '黄毛猎犬',
          'n02102040': '英国史宾格犬',
          'n02102177': '威尔士史宾格犬',
          'n02102318': '可卡犬',
          'n02102480': '萨塞克斯猎犬',
          'n02102973': '爱尔兰水猎犬',
          'n02104029': '哥威斯犬',
          'n02104365': '舒柏奇犬',
          'n02105056': '比利时牧羊犬',
          'n02105162': '马里努阿犬',
          'n02105251': '伯瑞犬',
          'n02105412': '凯尔皮犬',
          'n02105505': '匈牙利牧羊犬',
          'n02105641': '老英国牧羊犬',
          'n02105855': '喜乐蒂牧羊犬',
          'n02106030': '牧羊犬',
          'n02106166': '边境牧羊犬',
          'n02106382': '法兰德斯牧牛狗',
          'n02106550': '罗特韦尔犬',
          'n02106662': '阿尔萨斯',
          'n02107142': '多伯曼犬',
          'n02107312': '迷你杜宾犬',
          'n02107574': '大瑞士山地犬',
          'n02107683': '伯恩山犬',
          'n02107908': 'Appenzeller狗',
          'n02108000': 'EntleBucher狗',
          'n02108089': '拳师狗',
          'n02108422': '斗牛獒',
          'n02108551': '藏獒',
          'n02108915': '法国斗牛犬',
          'n02109047': '大丹犬',
          'n02109525': '圣伯纳德狗',
          'n02109961': '爱斯基摩狗',
          'n02110063': '雪橇犬',
          'n02110185': '哈士奇',
          'n02110341': '达尔马提亚, 教练车狗',
          'n02110627': '狮毛狗',
          'n02110806': '巴辛吉狗',
          'n02110958': '哈巴狗, 狮子狗',
          'n02111129': '莱昂贝格狗',
          'n02111277': '纽芬兰岛狗',
          'n02111500': '大白熊犬',
          'n02111889': '萨摩耶犬',
          'n02112018': '博美犬',
          'n02112137': '松狮',
          'n02112350': '荷兰卷尾狮毛狗',
          'n02112706': '布鲁塞尔格林芬犬',
          'n02113023': '彭布洛克威尔士科基犬',
          'n02113186': '威尔士柯基犬',
          'n02113624': '玩具贵宾犬',
          'n02113712': '迷你贵宾犬',
          'n02113799': '标准贵宾犬',
          'n02113978': '墨西哥无毛犬',
          'n02114367': '灰狼',
          'n02114548': '北极狼',
          'n02114712': '犬犬鲁弗斯',
          'n02114855': '狼, 草原狼, 刷狼, 郊狼',
          'n02115641': '澳洲野狗, 澳大利亚野犬',
          'n02115913': '豺',
          'n02116738': '非洲猎犬',
          'n02117135': '鬣狗',
          'n02119022': '红狐狸',
          'n02119789': '沙狐',
          'n02120079': '北极狐狸, 白狐狸',
          'n02120505': '灰狐狸',
          'n02123045': '虎斑猫',
          'n02123159': '山猫, 虎猫',
          'n02123394': '波斯猫',
          'n02123597': '暹罗暹罗猫',
          'n02124075': '埃及猫',
          'n02125311': '美洲狮,美洲豹',
          'n02128385': '豹子',
          'n02128757': '雪豹',
          'n02128925': '美洲虎',
          'n02129165': '狮子',
          'n02129604': '老虎',
          'n02130308': '猎豹',
          'n02132136': '棕熊'

}
category_17 = {0: '企鹅', 1: '兔子', 2: '北极猫', 3: '喜鹊', 4: '埃及猫', 5: '大白鲨', 6: '大虾', 7: '孔雀', 8: '海牛', 9: '淡水龟', 10: '狼蛛', 11: '猎犬', 12: '考拉', 13: '蝎子', 14: '蟒蛇', 15: '鳄鱼', 16: '鸵鸟'}
#category = labelRead.ImageNet_Label()
def Test_One_Image(image_path):
    #keras.backend.clear_session()
    #model = keras.models.load_model(model_path) # return a Model instance
    keras.backend.clear_session()
    model = keras.applications.ResNet50(input_tensor=keras.layers.Input(
        (224, 224, 3)), weights='imagenet', include_top=True)
    print('model load successfully')
# model.summary() # 输出模型结构
# return a PIL instance 一个 PIL 实例
    image = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = keras.preprocessing.image.img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    result = model.predict(image)
    #print('Predicted:', keras.applications.resnet50.decode_predictions(result, top=3)[0])
    count = 0
    for i in keras.applications.resnet50.decode_predictions(result, top=5)[0]:
        if category.__contains__(i[0]):
            print('识别结果:',category[i[0]],'置信度: {:.2%}'.format(i[2]))
            count+=1
    if count == 0:
        print('未检测到猫狗')
        # print('Tag:',i[0],'Categort:',i[1],'Probaility: ',i[2])
        # print(type(i[0]))
        # print(type(i[1]))
        # print(type(i[2]))
    #print(result)
def TestAll(Img_path):
    keras.backend.clear_session()
    # model = keras.applications.ResNet50(input_tensor=keras.layers.Input(
    #     (224, 224, 3)), weights='imagenet', include_top=True)
    model = keras.models.load_model('./weight/DC_model.hdf5')
    print('model load successfully')
    #model.save('./weight/DC_model.hdf5')
    paths = glob.glob(Img_path)
    for path in sorted(paths):
        # return a PIL instance 一个 PIL 实例
        image = keras.preprocessing.image.load_img(path, target_size=(224, 224))
        image = keras.preprocessing.image.img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        result = model.predict(image)
        print('\n')
        print('图像路径:', path)
        count = 0
        for i in keras.applications.resnet50.decode_predictions(result, top=5)[0]:
            if category.__contains__(i[0]):
                print('识别结果:', category[i[0]], '置信度: {:.2%}'.format(i[2]))
                count += 1
        if count == 0:
            print('未检测到猫狗')
        # np.set_printoptions(precision=2)
        # print('\n')
        # print('Image path is :', path)
        # #print('predict animal is :', category[label])
        # print('Predicted:', keras.applications.resnet50.decode_predictions(result, top=5)[0])

def TestAll2(Img_path):
    keras.backend.clear_session()
    # model = keras.applications.ResNet50(input_tensor=keras.layers.Input(
    #     (224, 224, 3)), weights='imagenet', include_top=True)
    model = keras.models.load_model('../weight2/ResNet50-17-weights.hdf5')
    print('model load successfully')
    #model.save('./weight/DC_model.hdf5')
    paths = glob.glob(Img_path)
    for path in sorted(paths):
        # return a PIL instance 一个 PIL 实例
        image = keras.preprocessing.image.load_img(path, target_size=(224, 224))
        image = keras.preprocessing.image.img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        result = model.predict(image)
        print('\n')
        print('图像路径:', path)
        print('predict result is: ',category_17[np.argmax(result[0])])
TestAll2('./TestImage/*')
#Test_One_Image('../TestImage/no-animal/12.jpg')