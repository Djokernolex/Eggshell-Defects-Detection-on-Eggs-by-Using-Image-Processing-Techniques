import cv2
import glob
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt



image = cv2.imread('X:/pycharm/pycharmdeneme/news/17.jpg')

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

egg_abn = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)



ret,thresh1 = cv2.threshold(gray,90,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(gray,90,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(gray,90,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(gray,90,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(gray,90,255,cv2.THRESH_TOZERO_INV)
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [gray, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()



values = (480,640)


ims = glob.glob("X:/pycharm/pycharmdeneme/project/*.jpg")
img_number  = 1
kirli_yumurtalar = []
temiz_yumurtalar= []
sobel_temiz = []
sobel_kirli = []
kirik_yumurta = []


durum = []

for x in ims:
    ims = cv2.imread(x)
    ims = cv2.resize(ims,values,interpolation= cv2.INTER_AREA)
    grays  = cv2.cvtColor(ims, cv2.COLOR_BGR2GRAY)
    cv2.imshow('het',grays)
    egg_abn = cv2.adaptiveThreshold(grays, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,4)
    esik_thresh = np.count_nonzero(egg_abn)
    #cv2.imshow('threshold',egg_abn)


    sobelx = cv2.Sobel(egg_abn, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(egg_abn, cv2.CV_64F, 0, 1, ksize=3)
    birlestirilmis_sobel_thresh = cv2.bitwise_or(sobelx, sobely)
    esik_sobel_thresh = np.count_nonzero(birlestirilmis_sobel_thresh)
    cv2.imshow('sobel',birlestirilmis_sobel_thresh)

    gaussian = cv2.GaussianBlur(birlestirilmis_sobel_thresh, (5, 5), 0)
    cv2.imshow('gaussian blur', gaussian)
    esik = np.count_nonzero(gaussian)


    if esik > 17817:
        kirli_yumurtalar.append(esik)
        sobel_kirli.append(esik_sobel_thresh)
        durum = 'kirli'
        print(f'{esik_thresh},{esik},{esik_sobel_thresh},{durum}')

     





    elif esik_sobel_thresh>6254:
        kirik_yumurta.append(esik_sobel_thresh)
        durum = 'kırık yumurta'
        print(f'{esik_thresh},{esik},{esik_sobel_thresh},{durum}')



    else:

        temiz_yumurtalar.append(esik)
        sobel_temiz.append(esik_sobel_thresh)
        durum = 'temiz'
        print(f'{esik_thresh},{esik},{esik_sobel_thresh},{durum}')


    cv2.imwrite("X:/pycharm/pycharmdeneme/hey/"+str(img_number)+".jpg",ims)
    img_number+= 1

    cv2.waitKey(0)
    cv2.destroyAllWindows()


yumurta_sayisi = len(kirli_yumurtalar)+len(temiz_yumurtalar)+len(kirik_yumurta)

print(f'Toplam {yumurta_sayisi} adet yumurta bulunmaktadır.')

print("kirli yumurtalar",kirli_yumurtalar)
print("kenar degerleri",sobel_kirli)

print("temiz yumurtalar ",temiz_yumurtalar)
print("kenar degerleri",sobel_temiz)
print("kırık yumurtalar : " ,kirik_yumurta)


kirli_yumurta_sayisi = len(kirli_yumurtalar)
temiz_yumurta_sayisi = len(temiz_yumurtalar)
kirik_yumurta_sayisi = len(kirik_yumurta)
print('Temiz yumurtalarin eşik değeri ortalaması',np.mean(temiz_yumurtalar))
print('Kirli yumurtaların eşik değeri ortalaması',np.mean(kirli_yumurtalar))
print('Kırık yumurtalarin eşik değeri ortalaması',np.mean(kirik_yumurta))



print(f'Kirli yumurta sayisi {kirli_yumurta_sayisi} iken kirli olmayan yumurta sayisi {yumurta_sayisi- kirli_yumurta_sayisi}')
print(f'{yumurta_sayisi- kirli_yumurta_sayisi} adet temiz olan yumurtalarımızın {kirik_yumurta_sayisi} tanesi kırıktır.')
print("**********************************************************************************************************************")
print(f'{yumurta_sayisi} adet yumurtanın  {temiz_yumurta_sayisi} i temiz yumurta,{kirli_yumurta_sayisi} i kirli yumurta ve {kirik_yumurta_sayisi} adedi ise kırık yumurtadır')




yumurta_türleri  = ["Temiz Yumurta", "Kirli Yumurta" ,"Kırık Yumurta"]
yumurta_sayilari  = [temiz_yumurta_sayisi,kirli_yumurta_sayisi,kirik_yumurta_sayisi]


plt.pie(yumurta_sayilari,labels=yumurta_türleri,autopct='%1.2f%%')
plt.show()


plt.figure(figsize=(10,6))


sbn.barplot(x=yumurta_türleri, y = yumurta_sayilari , palette = "mako")
plt.title("Yumurta çeşitlerinin dağılımı")
plt.xlabel("Sınıflandırılan Yumurta türleri")
plt.ylabel("Yumurta Adedi")

plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()










