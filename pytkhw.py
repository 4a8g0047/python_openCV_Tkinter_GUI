# 引入 tkinter 模組
import tkinter as tk
from tkinter import filedialog
import cv2 as cv
from cv2 import cvtColor
from PIL import ImageTk,Image
import matplotlib.pyplot as plt
import argparse
import numpy as np

# 建立主視窗 Frame
window = tk.Tk()
# 設定視窗標題
window.title('OpencvHW')
# 設定視窗大小為 300x100，視窗（左上角）在螢幕上的座標位置為 (250, 150)
window.geometry("1280x800+250+150")

def img_openimg():
    global img,img_size
    fileName=filedialog.askopenfilename(title='選擇',filetypes=[('All Files','*'),("jpeg files","*.jpg"),("png files","*.png"),("gif files","*.gif")])
    img=cv.imread(fileName)
    img_showimg()

def img_showimg():
    simg=cvtColor(img,cv.COLOR_BGR2RGB)
    aimgtk=Image.fromarray(simg)
    cimgtk=ImageTk.PhotoImage(image=aimgtk)
    imglabel.configure(image=cimgtk)
    imglabel.pack()                 # 以預設方式排版label
    window.mainloop()
    
def img_roi():
    r=cv.selectROI('roi',img,False,False)
    imgr=img[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
    cv.imshow("ims",imgr)
    cv.waitKey(0)

def img_hist():
    imgh=cvtColor(img,cv.COLOR_BGR2RGB)    
    plt.hist(imgh.ravel(), 256, [0, 256])
    plt.show()

def img_size():
    sizewindow=tk.Tk()
    sizewindow.title("圖片大小資訊")
    sizewindow.geometry("400x300")
    sl="圖片高度: %s\n圖片寬度: %s\nRGB維度: %s"%(img.shape[0],img.shape[1],img.shape[2])
    img_size=tk.Label(sizewindow,text=sl, font=('Arial', 24)).pack()
    sizewindow.mainloop()

def img_rgb():
    simg=cvtColor(img,cv.COLOR_BGR2RGB)
    aimgtk=Image.fromarray(simg)
    cimgtk=ImageTk.PhotoImage(image=aimgtk)
    imglabel.configure(image=cimgtk)
    imglabel.pack()                 # 以預設方式排版label
    window.mainloop()    

def img_bgr():
    aimgtk=Image.fromarray(img)
    cimgtk=ImageTk.PhotoImage(image=aimgtk)
    imglabel.configure(image=cimgtk)
    imglabel.pack()                 # 以預設方式排版label
    window.mainloop()

def img_gray():
    simg=cvtColor(img,cv.COLOR_RGB2GRAY)
    aimgtk=Image.fromarray(simg)
    cimgtk=ImageTk.PhotoImage(image=aimgtk)
    imglabel.configure(image=cimgtk)
    imglabel.pack()                 # 以預設方式排版label
    window.mainloop()

def img_hsv():
    simg=cvtColor(img,cv.COLOR_RGB2HSV)
    aimgtk=Image.fromarray(simg)
    cimgtk=ImageTk.PhotoImage(image=aimgtk)
    imglabel.configure(image=cimgtk)
    imglabel.pack()                 # 以預設方式排版label
    window.mainloop()

def img_thresholding():
    originimg=cvtColor(img,cv.COLOR_BGR2RGB)
    ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    ret, th2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
    ret, th3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
    ret, th4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
    ret, th5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)

    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [originimg, th1, th2, th3, th4, th5]

    for i in range(6):
        plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

def img_histogram_equalization():
    image=cvtColor(img,cv.COLOR_BGR2RGB)
    (b,g,r)=cv.split(image)
    bH=cv.equalizeHist(b)
    gH=cv.equalizeHist(g)
    rH=cv.equalizeHist(r)
    result=cv.merge((bH,gH,rH))
    cv.imshow("histogram_equalization",result)
    cv.waitKey(0)

def img_canny_detector():
    max_lowThreshold = 100
    window_name = 'Edge Map'
    title_trackbar = 'Min Threshold:'
    ratio = 3
    kernel_size = 3
    def CannyThreshold(val):
        low_threshold = val
        img_blur = cv.blur(img_gray, (3,3))
        detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
        mask = detected_edges != 0
        dst = img * (mask[:,:,None].astype(img.dtype))
        cv.imshow(window_name, dst)
    parser = argparse.ArgumentParser(description='Code for Canny Edge Detector tutorial.')
    parser.add_argument('--input', help='Path to input image.', default='fruits.jpg')
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.namedWindow(window_name)
    cv.createTrackbar(title_trackbar, window_name , 0, max_lowThreshold, CannyThreshold)
    CannyThreshold(0)
    cv.waitKey()

def img_meanfilter():
    img_mean = cv.blur(img, (5,5))
    cv.imshow("MeanFilter",img_mean)
    cv.waitKey(0)

def img_guassianfilter():
    img_Guassian = cv.GaussianBlur(img,(5,5),0)
    cv.imshow("GuassianFilter",img_Guassian)
    cv.waitKey(0)

def img_medianfilter():
    img_median = cv.medianBlur(img, 5)
    cv.imshow("MedianFilter",img_median)
    cv.waitKey(0)

def img_bilaterfilter():
    img_bilater = cv.bilateralFilter(img,9,75,75)
    cv.imshow("BilaterFilter",img_bilater)
    cv.waitKey(0)

def img_flip():
    flip_img=cv.flip(img,1)
    cv.imshow("flip",flip_img)

def img_change():
    
    def move_img(x,y):
        im_x,im_y=img.shape[:2]

        M=np.float32([[1,0,x],[0,1,y]])    #構造轉換矩陣M
        move=cv.warpAffine(img,M,(im_x,im_y))  #平移對映
        cv.imshow("move_img",move)

    def rotate_img(rect):
        im_x,im_y=img.shape[:2]
        im_xy=(im_x/2,im_y/2)   # 取圖片中心
        rotateMatrix = cv.getRotationMatrix2D(im_xy, rect, 1.0)
        out=cv.warpAffine(src=img,M=rotateMatrix,dsize=(im_x,im_y))
        cv.imshow("rectangle_img",out)

    def size_img(size_x,size_y):
        im_x,im_y=img.shape[:2]
        im_size = cv.resize(img,(size_x*im_y, size_y*im_x), interpolation = cv.INTER_LINEAR)
        cv.imshow("size_img",im_size)

    scale_move_x = tk.IntVar()
    scale_move_y = tk.IntVar()
    scale_rotate_value = tk.IntVar()
    scale_flip_x = tk.DoubleVar()
    scale_flip_x.set(1)
    scale_flip_y = tk.DoubleVar()
    scale_flip_y.set(1)
    scale_flip_xy = tk.DoubleVar()
    scale_flip_xy.set(1)

    changewindow=tk.Toplevel()
    changewindow.title("幾何轉換")
    changewindow.geometry("300x600")
    movelabel=tk.Label(changewindow,text="平移")
    movelabel.pack()
    #水平移動
    horizontallabel=tk.Label(changewindow,text="x")
    horizontallabel.pack()
    scale_move_x=tk.Scale(changewindow, orient=tk.HORIZONTAL,from_=-500,to=500,variable=scale_move_x
    ,command=lambda x=None:move_img(scale_move_x.get(),scale_move_y.get()))
    scale_move_x.pack()
    #垂直移動
    verticallabel=tk.Label(changewindow,text="Y")
    verticallabel.pack()    
    scale_move_y=tk.Scale(changewindow, orient=tk.HORIZONTAL,from_=-500,to=500,variable=scale_move_y
    ,command=lambda x=None:move_img(scale_move_x.get(),scale_move_y.get()))
    scale_move_y.pack()
    #轉向
    rotatelabel=tk.Label(changewindow,text="轉向")
    rotatelabel.pack() 
    scale_rotate=tk.Scale(changewindow, orient=tk.HORIZONTAL,from_=-360,to=360,variable=scale_rotate_value,
    command=lambda x=None:rotate_img(scale_rotate_value.get()))
    scale_rotate.pack()
    #鏡像翻轉
    fliplabel=tk.Label(changewindow,text="鏡像翻轉")
    fliplabel.pack() 
    flipbutton=tk.Button(changewindow,text="鏡像翻轉",command=img_flip)
    flipbutton.pack()
    #縮放
    sizelabel=tk.Label(changewindow,text="縮放")
    sizelabel.pack()

    sizelabel=tk.Label(changewindow,text="X")
    sizelabel.pack()
    scale_flip_x=tk.Scale(changewindow,resolution=0.01,orient=tk.HORIZONTAL,from_=0.1,to=10,variable=scale_flip_x,
    command=lambda x=None:size_img(scale_flip_x.get(),scale_flip_y.get()))
    scale_flip_x.pack()
    
    sizelabel=tk.Label(changewindow,text="Y")
    sizelabel.pack()
    scale_flip_y=tk.Scale(changewindow,resolution=0.01,orient=tk.HORIZONTAL,from_=0.1,to=10,variable=scale_flip_y,
    command=lambda x=None:size_img(scale_flip_x.get(),scale_flip_y.get()))
    scale_flip_y.pack()

def img_transform():
    def mouse(event,x,y,flags,param):
        if event==cv.EVENT_LBUTTONDOWN: #滑鼠左鍵事件
            pts.append((x,y))        
        if event==cv.EVENT_RBUTTONDOWN: #滑鼠右鍵事件
            pts.pop()        
        if event==cv.EVENT_MBUTTONDOWN: #滑鼠滾輪鍵事件
            mask1=np.zeros(img.shape,np.uint8)
            points=np.array(pts,np.int32)
            points=points.reshape((-1,1,2))
            mask1=cv.polylines(mask1,[points],True,(255,255,255),2)
            mask2=cv.fillPoly(mask1.copy(),[points],(255,255,255))
            mask3=cv.fillPoly(mask1.copy(),[points],(0,255,0))
            show_img=cv.addWeighted(src1=img,alpha=0.8,src2=mask3,beta=0.2,gamma=0)
            cv.imshow("show_img",show_img)
            roi=cv.bitwise_and(mask2,img)
            cv.imshow("ROI",roi)
            cv.waitKey(0)
        if len(pts)>0:
            cv.circle(img_c,pts[-1],3,(0,0,255),-1)
        if len(pts)>1:
            for i in range(len(pts)-1):
                cv.circle(img_c,pts[i],5,(0,0,255),-1)
                cv.line(img=img_c,pt1=pts[i],pt2=pts[i+1],color=(255,0,0),thickness=2)
    pts=[]
    img_c=img.copy()  
    cv.imshow("Image",img)
    cv.setMouseCallback("Image",mouse)

    while(1):
        key=cv.waitKey(13)
        if key==13:
            pts1=np.float32([[pts[0][0],pts[0][1]],[pts[1][0],pts[1][1]],[pts[2][0],pts[2][1]],[pts[3][0],pts[3][1]]])
            pts2=np.float32([[0,0],[300,0],[0,300],[300,300]])
            M=cv.getPerspectiveTransform(pts1,pts2)
            dst=cv.warpPerspective(img,M,(300,300))
            cv.imshow("dst",dst)
            break

def img_erosion():
    kernel = np.ones((5,5),np.uint8)
    erosion = cv.erode(img,kernel,iterations = 1)
    cv.imshow("Erosion",erosion)

def img_dilation():
    kernel = np.ones((5,5),np.uint8)
    dilation = cv.dilate(img,kernel,iterations = 1)
    cv.imshow("Dilation",dilation)

def img_opening():
    kernel = np.ones((5,5),np.uint8)
    opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    cv.imshow("Opening",opening)

def img_closing():
    kernel = np.ones((5,5),np.uint8)
    closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    cv.imshow("Closing",closing)

def img_morphological_gradient():
    kernel = np.ones((5,5),np.uint8)
    gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
    cv.imshow("Morphological Gradient",gradient)

def img_top_hat():
    kernel = np.ones((5,5),np.uint8)
    tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
    cv.imshow("Top Hat",tophat)

def img_black_hat():
    kernel = np.ones((5,5),np.uint8)
    blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
    cv.imshow("Black Hat",blackhat)

# 建立選單
filemenu=tk.Menu(window)
window.config(menu=filemenu)
openfilemenu=tk.Menu(filemenu,tearoff=0)
setfilemenu=tk.Menu(filemenu,tearoff=0)
setfilemenu2=tk.Menu(filemenu,tearoff=0)
setfilemenu3=tk.Menu(filemenu,tearoff=0)
profilemenu=tk.Menu(filemenu,tearoff=0)
filtermenu=tk.Menu(filemenu,tearoff=0)
geometrymenu=tk.Menu(filemenu,tearoff=0)
morphologicalmenu=tk.Menu(filemenu,tearoff=0)

openfilemenu.add_command(label="開啟圖片",command=img_openimg)

setfilemenu.add_command(label="設定ROI",command=img_roi)
setfilemenu2.add_command(label="顯示影像直方圖",command=img_hist)
setfilemenu2.add_command(label="顯示影像大小資訊",command=img_size)
setfilemenu.add_cascade(label="影像資訊",menu=setfilemenu2)

setfilemenu3.add_command(label="RGB",command=img_rgb)
setfilemenu3.add_command(label="BGR",command=img_bgr)
setfilemenu3.add_command(label="GRAY",command=img_gray)
setfilemenu3.add_command(label="HSV",command=img_hsv)
setfilemenu.add_cascade(label="色彩空間轉換",menu=setfilemenu3)

profilemenu.add_command(label="影像二值化",command=img_thresholding)
profilemenu.add_command(label="直方圖等化",command=img_histogram_equalization)

filtermenu.add_command(label="邊緣檢測器",command=img_canny_detector)
filtermenu.add_command(label="均值濾波",command=img_meanfilter)
filtermenu.add_command(label="高斯濾波",command=img_guassianfilter)
filtermenu.add_command(label="中值濾波",command=img_medianfilter)
filtermenu.add_command(label="雙邊濾波",command=img_bilaterfilter)

geometrymenu.add_command(label="幾何轉換",command=img_change)
geometrymenu.add_command(label="透視投影轉換",command=img_transform)

morphologicalmenu.add_command(label="Erosion",command=img_erosion)
morphologicalmenu.add_command(label="Dilation",command=img_dilation)
morphologicalmenu.add_command(label="Opening",command=img_opening)
morphologicalmenu.add_command(label="Closing",command=img_closing)
morphologicalmenu.add_command(label="Morphological Gradient",command=img_morphological_gradient)
morphologicalmenu.add_command(label="Top Hat",command=img_top_hat)
morphologicalmenu.add_command(label="Black Hat",command=img_black_hat)

filemenu.add_cascade(label="檔案",menu=openfilemenu)
filemenu.add_cascade(label="設定",menu=setfilemenu)
filemenu.add_cascade(label="影像處理",menu=profilemenu)
filemenu.add_cascade(label="濾波器",menu=filtermenu)
filemenu.add_cascade(label="幾何轉換功能",menu=geometrymenu)
filemenu.add_cascade(label="型態變換",menu=morphologicalmenu)

imglabel = tk.Label(window)              # 文字標示所在視窗

# 執行主程式
window.mainloop()