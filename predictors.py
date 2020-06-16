import cv2
import numpy as np


# def predicting(arr):
#     from sklearn.externals import joblib
#     model = joblib.load('/Users/aek/PycharmProjects/SeniorProjectWeb/eyemodel.joblib')
#     predictions = model.predict(arr)
#     if predictions == [0]:
#         predictions = 0
#         return predictions
#     else:
#         predictions = 1
#     return predictions


def dim(img):
    width = int(img.shape[1])
    height = int(img.shape[0])
    if height > 2000:
        height = int(height / 6)
        width = int(width / 6)
    dimension = (width, height)
    return dimension


def eyeratio(face, ew, eh):
    if len(face) > 1:
        area1 = face.shape[0] * face.shape[1] * 2
    else:
        area1 = face[0][2] * face[0][3]
    area2 = ew * eh
    ratio = area2 / area1
    # print("areaface", area1)
    # print("areaeye", area2)
    return ratio


def whiteeye(img, gray, eye):
    for (ex, ey, ew, eh) in eye:
        eh = int(eh * 0.4)
        ey = int(ey + int(0.7 * eh))
        ew = int(ew * 0.8)
        ex = int(ex + int(0.3 * ew))
        y = 0
        n = 0
        for i in range(int(0.5 * ew)):
            bright = gray[ey + int(0.5 * eh)][ex + i]
            if img[ey + int(0.5 * eh)][ex + i][1] > bright:
                y = y + 1
            else:
                n = n + 1
        if y > 0.2 * n:
            answer = 0
            return answer
        else:
            answer = 1
            return answer


def eblue(img):
    avgb = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            blue = img[i][j][0]
            avgb = avgb + blue
    avgb = avgb / (img.shape[0] * img.shape[1])
    return avgb


def egreen(img):
    avgg = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            green = img[i][j][1]
            avgg = avgg + green
    avgg = avgg / (img.shape[0] * img.shape[1])
    return avgg


def ered(img):
    avgr = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            red = img[i][j][2]
            avgr = avgr + red
    avgr = avgr / (img.shape[0] * img.shape[1])
    return avgr


def facedetect(face):
    if len(face) > 1:
        arr = np.amax(face, axis=0)
        arr = np.where(face == arr[2])
        face = face[arr[0]]
        return face
    else:
        return face


def eyedetect(face, eye):
    i = 0
    for ex, ey, ew, eh in eye:
        # print("original eye:", eye)
        # print("ex: ", ex)
        if (ex < face[0][0]) or (ey < face[0][1]):
            eye = np.delete(eye, i, 0)
            i = i - 1
        elif (ex > face[0][0] + face[0][3]) or (ey > face[0][1] + face[0][3] / 2):
            eye = np.delete(eye, i, 0)
            i = i - 1

        i = i + 1
    return eye


def avgbrightness(gray):
    avgbright = 0
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            bright = gray[i][j]
            avgbright = (avgbright * ((i * gray.shape[1]) + j) + bright) / ((i * gray.shape[1]) + (j + 1))
    return avgbright


def avgbgr(img):
    avgpigment = [0, 0, 0]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            bgr = img[i][j]
            avgpigment = avgpigment + bgr
    avgpigment = avgpigment / (img.shape[0] * img.shape[1])
    return avgpigment


# def faceinput(file):
#     eye_cascade = cv2.CascadeClassifier("/Users/aek/iCloud Drive (Archive)/Desktop/SeniorProject/haar-cascade-files/haarcascade_eye.xml")
#     face_cascade = cv2.CascadeClassifier(
#         "/Users/aek/iCloud Drive (Archive)/Desktop/SeniorProject/haar-cascade-files/haarcascade_frontalface_default.xml")
#     img = file
#     dimension = dim(img)
#     img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     face = face_cascade.detectMultiScale(gray, 1.3, 2)
#     eye = eye_cascade.detectMultiScale(gray, 1.1, 3)
#     print(dimension[1])
#     print(dimension[0])
#     if dimension[1]< dimension[0]:
#         face = [[0, 0, img.shape[1], img.shape[0]]]
#         eye = eyedetect(face, eye)
#     else:
#         face = facedetect(face)
#         eye = eyedetect(face, eye)
#     for ex, ey, ew, eh in eye:
#         ans = []
#         ratio = eyeratio(face, ew, eh)
#         color_eye = img[ey:ey + eh, ex:ex + ew]
#         gray_eye = cv2.cvtColor(color_eye, cv2.COLOR_BGR2GRAY)
#         brightness = int(avgbrightness(gray_eye))
#         blue = int(eblue(color_eye))
#         green = int(egreen(color_eye))
#         red = int(ered(color_eye))
#         white_eye = whiteeye(img, gray, eye)
#         ans = ans + [ratio, brightness, blue, green, red, white_eye]
#
#     ans = np.asarray(ans)
#     return ans

def dimt(img):
    width = int(img.shape[1])
    height = int(img.shape[0])
    if height > 1000:
        height = int(height/4)
        width = int(width/4)
    dim = (width, height)
    return dim


def skindetection(img):
    non = 1
    x = img.shape[1]
    y = img.shape[0]
    if x >500 or y > 500:
        cutx = int((img.shape[1])/3)
        cuty = int((img.shape[0])/3)
    else:
        cutx = int((img.shape[1])/5)
        cuty = int((img.shape[0])/5)
    last_img = img[cuty:int(y-cuty), cutx:int(x-cutx)]
    return last_img


def findrash(file):
    img = file
    dimension1 = dimt(img)
    img = cv2.resize(img, dimension1, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = skindetection(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = skindetection(gray)
    red = []
    green = []
    blue = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            red.append(img[i][j][2])
            green.append(img[i][j][1])
            blue.append(img[i][j][0])

    count = 0
    # sdr = np.std(red)
    meanr = np.mean(red)
    # sdg = np.std(green)
    meang = np.mean(green)
    # sdb = np.std(blue)
    # meanb = np.mean(blue)
    ratio = meanr / meang
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][2] / img[i][j][1] > (1.1 * ratio):
                img[i][j] = [255, 255, 255]
                count = count + 1

    finalratio = count / (img.shape[0] * img.shape[1])
    if finalratio > 0.1:
        predictions = 1
        return predictions
    else:
        predictions = 0
        return predictions

def projectprediction (arr):
    from sklearn.externals import joblib
    model = joblib.load('/Users/aek/PycharmProjects/SeniorProjectWeb/Seniorprojectmodel.joblib')
    predictions = model.predict(arr)
    return predictions

def faceinput(file):
    eye_cascade = cv2.CascadeClassifier("/Users/aek/iCloud Drive (Archive)/Desktop/SeniorProject/haar-cascade-files/haarcascade_eye.xml")
    face_cascade = cv2.CascadeClassifier("/Users/aek/iCloud Drive (Archive)/Desktop/SeniorProject/haar-cascade-files/haarcascade_frontalface_default.xml")
    mouth_cascade = cv2.CascadeClassifier("/Users/aek/iCloud Drive (Archive)/Desktop/SeniorProject/haar-cascade-files/haarcascade_smile.xml")
    img = file
    dimension = dim(img)
    img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.3, 2)
    eye = eye_cascade.detectMultiScale(gray, 1.1, 3)
    img_mouth= lowerface(img)
    gray_mouth = cv2.cvtColor(img_mouth, cv2.COLOR_BGR2GRAY)
    mouth= mouth_cascade.detectMultiScale(gray_mouth, scaleFactor= 1.2, minNeighbors= 35)
    #print(dimension[1])
    #print(dimension[0])
    if (dimension[1]< dimension[0]):
        face = [[0, 0, img.shape[1], img.shape[0]]]
        eye = eyedetect(face, eye)
    else:
        face = facedetect(face)
        eye = eyedetect(face, eye)
    for ex, ey, ew, eh in eye:
        write = []
        ratio = eyeratio(face, ew, eh)
        color_eye = img[ey:ey + eh, ex:ex + ew]
        gray_eye = cv2.cvtColor(color_eye, cv2.COLOR_BGR2GRAY)
        brightness = int(avgbrightness(gray_eye))
        blue = int(eblue(color_eye))
        green = int(egreen(color_eye))
        red = int(ered(color_eye))
        white_eye = whiteeye(img, gray, eye)
        write = write + [ratio, brightness, blue, green, red, white_eye]
        write = np.asarray(write)
    for (mx, my, mw, mh) in mouth:
        list_mx = []
        list_my = []
        list_mw = []
        list_mh = []
        write_mouth= []
        list_mx.append(mx)
        list_my.append(my)
        list_mw.append(mw)
        list_mh.append(mh)
        #print('No. of mx:', len(list_mx))
    t= mouth_ratio(list_mx, list_mh, list_mw, img_mouth)
    #print ('t= ', t)
    x= list_mx[t]
    y= list_my[t]
    h= list_mh[t]
    w= list_mw[t]
    #print('x= ', x)
    #print('y', y)
    if y> 30 and x> 30:
        img_crop= img_mouth[y-30:y+h+30, x-30:x+w+30]
    elif y<= 30:
        img_crop= img_mouth[0:y+h, x-30:x+w+30]
    elif x<= 30:
        img_crop= img_mouth[y-30:y+h+30, 0:x+w]
    gray_img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    arr1= np.asarray(gray_img)
    sum_pixel = arr1.sum(0).sum(0)
    #print('sumpix= ', sum_pixel)
    m_percent_pink= mouth_pink_percentage(img_crop)
    m_percent_dark= mouth_dark_percentage(gray_img, img_crop, sum_pixel)
    #print('dark pixel percentage:', m_percent_dark)
    #print('pink pixel percentage:', m_percent_pink)
    m_avgbrightness= mouth_avgbrightness(gray_img)
    m_avgR= mouth_avg_red(img_crop, img_mouth)
    m_avgG= mouth_avg_green(img_crop, img_mouth)
    m_avgB= mouth_avg_blue(img_crop, img_mouth)
    #print('avgbrightness:',m_avgbrightness)
    #print('avgR:',m_avgR)
    #print('avgG:',m_avgG)
    #print('avgB:',m_avgB)
    write_mouth = write_mouth + [m_avgbrightness, m_avgR, m_avgG, m_avgB, m_percent_pink, m_percent_dark]
    write_mouth = np.asarray(write_mouth)
    return [write, write_mouth]
def lowerface(img):
    width = int(img.shape[1])
    height = int(img.shape[0])
    height_cutoff = int(height// 1.9)
    #upper = img[:height_cutoff, :]
    lower = img[height_cutoff:, :]
    return lower

def mouth_ratio(list_mx, list_mh, list_mw, img_mouth):
    list_ratio = []
    list_area_mouth= []
    for i in range(len(list_mx)):
        list_area_mouth= list_mh[i]* list_mw[i]
        picture_area= img_mouth.shape[0]* img_mouth.shape[1]
        ratio= picture_area/ list_area_mouth
        list_ratio.append(ratio)
    #print('list ratio =' ,list_ratio)
    t= 0
    for i in range(len(list_ratio)):
        if list_ratio[i]< list_ratio[t]:
            t=i
    return t

def mouth_pink_percentage(img_crop):
    pink = [180, 80, 80]  # RGB
    brown= [110, 70, 70]
    diff= 20
    diff_0 = abs(pink[0]- brown[0])
    diff_1 = abs( pink[1]- brown[1])
    diff_2 = abs(pink[2]- brown[2])
    boundaries = [([pink[2]-diff_2, pink[1]-diff_1, pink[0]-diff_0], [pink[2]+diff, pink[1]+diff, pink[0]+diff])]
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(img_crop, lower, upper)
        output = cv2.bitwise_and(img_crop, img_crop, mask=mask)
        ratio_pink = cv2.countNonZero(mask)/(img_crop.size/3)
        percent_pink_pixel= np.round(ratio_pink*100, 2)
    return percent_pink_pixel

def mouth_dark_percentage(gray_img, img_crop, sum_pixel):
    threshold_level = 45
    coords = np.column_stack(np.where(gray_img < threshold_level))
    mask = gray_img < threshold_level
    img_crop[mask]= (204, 119, 0)
    num_dark_pixel= sum(gray_img[gray_img < threshold_level])
    percent_dark_pixel= (num_dark_pixel/ sum_pixel)* 100
    return percent_dark_pixel

def mouth_avgbrightness(gray_img):
    m_avgbrightness = 0
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            bright = gray_img[i][j]
            m_avgbrightness = ((m_avgbrightness*((i*gray_img.shape[1])+j)+bright))/((i*gray_img.shape[1])+(j+1))
    return m_avgbrightness

def mouth_avg_red(img_crop, img_mouth):
    avgrgb = [0, 0, 0]
    for i in range(img_crop.shape[0]):
        for j in range(img_crop.shape[1]):
            rgb = img_mouth[i][j]
            avgrgb = avgrgb+ rgb
        avgrgb = avgrgb/(img_crop.shape[0]*img_crop.shape[1])
        avgR= avgrgb[0]
    return avgR

def mouth_avg_green(img_crop, img_mouth):
    avgrgb = [0, 0, 0]
    for i in range(img_crop.shape[0]):
        for j in range(img_crop.shape[1]):
            rgb = img_mouth[i][j]
            avgrgb = avgrgb+ rgb
        avgrgb = avgrgb/(img_crop.shape[0]*img_crop.shape[1])
        avgG= avgrgb[1]
    return avgG

def mouth_avg_blue(img_crop, img_mouth):
    avgrgb = [0, 0, 0]
    for i in range(img_crop.shape[0]):
        for j in range(img_crop.shape[1]):
            rgb = img_mouth[i][j]
            avgrgb = avgrgb+ rgb
        avgrgb = avgrgb/(img_crop.shape[0]*img_crop.shape[1])
        avgB= avgrgb[2]
    return avgB
def predicting_eye(arr):
    from sklearn.externals import joblib
    model = joblib.load('/Users/aek/PycharmProjects/SeniorProjectWeb/eyemodel.joblib')
    predictions = model.predict(arr)
    if predictions == [0]:
        predictions = 0
    else:
        predictions = 1
    return predictions

def predicting_mouth(arr):
    from sklearn.externals import joblib
    model = joblib.load('/Users/aek/PycharmProjects/SeniorProjectWeb/mouthmodel.joblib')
    predictions = model.predict(arr)
    if predictions == [0]:
        predictions = 0
    else:
        predictions = 1
    return predictions

# arr = cv2.imread('/Users/aek/PycharmProjects/SeniorProjectWeb/media/Test1.png')
# arr = (faceinput(arr))
# ans = (predicting([arr]))
# print(ans)

