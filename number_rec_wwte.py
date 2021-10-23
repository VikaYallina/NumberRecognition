import time

from PIL import Image, ImageTk
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
import pytesseract
from tkinter import messagebox
from imutils.object_detection import non_max_suppression

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def EAST_text_detector(image):
    pred, coords = [], []

    original_image_copy = image.copy()
    temp = image.copy()
    # Converting the image to gray scale
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

    (H, W) = image.shape[:2]
    # Setting the new width and height and then determine the ratio in change for both the width and height
    (newW, newH) = (512, 288)
    rW = W / float(newW)
    rH = H / float(newH)
    # Resizing the image and taking the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # Setting the output layer set
    layers = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    # print("Loading EAST text detector...")
    east_net = cv2.dnn.readNet('frozen_east_text_detection.pb')
    # Constructing a blob from the image and then performing a forward pass of the model to obtain the two output
    # layer sets
    mean_color = np.average(image, axis=1)
    mean_bgr = np.average(mean_color, axis=0)
    mean_rgb = tuple([mean_bgr[2], mean_bgr[1], mean_bgr[0]])
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), mean_rgb, swapRB=True, crop=False)

    start = time.time()
    east_net.setInput(blob)

    (scores, geometry) = east_net.forward(layers)
    end = time.time()

    # Taking the number of rows and columns from the scores volume, then initializing our set of bounding box rectangles
    # and corresponding confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects, confidences = [], []
    for y in range(0, numRows):
        # Extracting the scores (probabilities), followed by the geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            # If our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.5:
                continue
            # Computing the offset factor as our resulting feature maps will be 4 times smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # Extracting the rotation angle for the prediction and then computing the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # Using the geometry volume to derive the width and height of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # Computing both the starting and ending (x, y)-coordinates for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # Adding the bounding box coordinates and probability score to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # Applying non-maxima suppression to suppress weak, overlapping bounding boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    for (startX, startY, endX, endY) in boxes:
        # Scaling the bounding box coordinates based on the respective ratios
        startX = int(startX * rW) - 2
        startY = int(startY * rH) - 1
        endX = int(endX * rW) + 2
        endY = int(endY * rH) + 2

        # If coords are out of the image dimension resizing it
        if startX < 0:
            startX = 0
        if endX > original_image_copy.shape[1]:
            endX = original_image_copy.shape[1]
        if startY < 0:
            startY = 0
        if endY > original_image_copy.shape[0]:
            endY = original_image_copy.shape[0]

        # Drawing the bounding box on the image
        # cv2.rectangle(original_image_copy, (startX, startY), (endX, endY), (0, 255, 0), 2)
        if endX > original_image_copy.shape[1] or endY > original_image_copy.shape[0]:
            endX = original_image_copy.shape[1]
            endY = original_image_copy.shape[0]
        coords.append((startY, endY, startX, endX))

    gray = cv2.cvtColor(original_image_copy, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((original_image_copy.shape[0], original_image_copy.shape[1], 1), dtype=np.uint8)
    for coord in coords:
        cv2.rectangle(mask, (coord[2], coord[0]), (coord[3], coord[1]), (255, 255, 255), -1)
    out = cv2.bitwise_and(gray, gray, mask=mask)

    return out


class App(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.img = None
        self.out = None
        self.filename = ""

        self.img1 = tk.Label(self.master, text="here1")
        self.img1.grid(row=1, column=0)
        self.img2 = tk.Label(self.master, text="here2")
        self.img2.grid(row=1, column=1)

        self.file_select = tk.Button(self.master, text="Choose image", command=self.callback) \
            .grid(row=0, column=0, sticky=tk.W, pady=5)
        self.start_btn = tk.Button(self.master, text="Start", command=self.task) \
            .grid(row=0, column=1, sticky=tk.W, pady=5)

        self.frame_left = tk.Frame(self.master)
        self.frame_left.grid(row=2,column=0,sticky='w')

        # create the center widgets
        self.frame_left.grid_rowconfigure(1, weight=1)
        self.frame_left.grid_columnconfigure(1, weight=1)
        #Thresholding
        tk.Label(self.frame_left, text="Thresholding").grid(row=0, column=0)
        self.thresh_var = tk.IntVar()
        self.thresh_var.set(1)

        self.rb_thresh_manual = tk.Radiobutton(self.frame_left, text="Manual", variable=self.thresh_var, value=0)
        self.rb_thresh_manual.grid(row=1, column=0, sticky='w')

        self.thresh_manual_param = tk.Entry(self.frame_left, width=4)
        self.thresh_manual_param.grid(row=1, column=1, sticky='w')
        self.thresh_manual_param.insert(0, "200")

        self.rb_thresh_otsu = tk.Radiobutton(self.frame_left, text="Otsu", variable=self.thresh_var, value=1)
        self.rb_thresh_otsu.grid(row=2, column=0, sticky='w')

        self.rb_thresh_adapt_mean = tk.Radiobutton(self.frame_left, text="Adaptive Mean", variable=self.thresh_var,
                                                   value=2)
        self.rb_thresh_adapt_mean.grid(row=3, column=0, sticky='w')

        self.thresh_adapt_mean_param = tk.Entry(self.frame_left, width=3)
        self.thresh_adapt_mean_param.grid(row=3, column=1, sticky='w')
        self.thresh_adapt_mean_param.insert(0, "10")

        self.rb_thresh_adapt_gauss = tk.Radiobutton(self.frame_left, text="Adaptive Gauss", variable=self.thresh_var,
                                                    value=3)
        self.rb_thresh_adapt_gauss.grid(row=4, column=0, sticky='w')

        self.thresh_adapt_gauss_param = tk.Entry(self.frame_left, width=3)
        self.thresh_adapt_gauss_param.grid(row=4, column=1, sticky='w')
        self.thresh_adapt_gauss_param.insert(0, "10")

        self.rb_thresh_none = tk.Radiobutton(self.frame_left, text='None', variable=self.thresh_var,
                                             value=4)
        self.rb_thresh_none.grid(row=5, column=0, sticky='w')

        # Image processing
        tk.Label(self.frame_left, text="Processing").grid(row=0, column=2)

        self.gauss_filter_var = tk.BooleanVar()
        self.gauss_filter_var.set(True)
        self.gauss_filter_chk = tk.Checkbutton(self.frame_left, text='Gauss filter', onvalue=1, offvalue=0,
                                               variable=self.gauss_filter_var)
        self.gauss_filter_chk.grid(row=1, column=2,sticky='w')

        self.opening_var = tk.BooleanVar()
        self.opening_chk = tk.Checkbutton(self.frame_left, text='Opening', onvalue=1, offvalue=0,
                                          variable=self.opening_var)
        self.opening_chk.grid(row=2, column=2,sticky='w')

        # Detection method
        tk.Label(self.frame_left, text="Detection").grid(row=0, column=3)

        self.detect_var = tk.IntVar()
        self.none_chk = tk.Radiobutton(self.frame_left, text='None', variable=self.detect_var,
                                       value=0)
        self.none_chk.grid(row=1, column=3)
        self.mser_chk = tk.Radiobutton(self.frame_left, text="MSER", variable=self.detect_var,
                                       value=1)
        self.mser_chk.grid(row=2, column=3)

        self.east_chk = tk.Radiobutton(self.frame_left, text="EAST", variable=self.detect_var,
                                       value=2)
        self.east_chk.grid(row=3, column=3)

    def callback(self):
        self.filename = filedialog.askopenfilename(filetypes=[
            ("Images", ".jpeg .jpg .png"),
            ("JPEG", ".jpeg"),
            ("JPG", ".jpg"),
            ("PNG", ".png"),
        ])
        if self.filename != '':
            self.img = Image.open(self.filename, mode="r")
            h, w = self.img.size
            if h > 300 or w > 300:
                self.img = self.img.resize((int(h / 2), int(w / 2)))
            self.img = ImageTk.PhotoImage(self.img)
            self.img1.config(image=self.img)
            self.img1.image = self.img

    def task(self):
        image = cv2.imread(self.filename)
        vis = image.copy()

        if self.detect_var.get() == 2:
            image = EAST_text_detector(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Размытие по Гауссу
        if self.gauss_filter_var.get():
            image = cv2.GaussianBlur(image, (3, 3), 0)

        # Вычисление порога бинаризации различными методами
        if self.thresh_var.get() == 0:
            # Стандартный метод
            image = cv2.threshold(image, int(self.thresh_manual_param.get()), 255, cv2.THRESH_BINARY_INV)[1]
        elif self.thresh_var.get() == 1:
            # Метод Оцу
            image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        elif self.thresh_var.get() == 2:
            # Адаптивный метод среднего значения
            image = cv2.adaptiveThreshold(image, 255,
                                          cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21,
                                          int(self.thresh_adapt_mean_param.get()))
        elif self.thresh_var.get() == 3:
            # Адаптивный метод по Гауссу
            image = cv2.adaptiveThreshold(image, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21,
                                          int(self.thresh_adapt_gauss_param.get()))

        # Открытие - это размывание, за которым следует растягивание изображения.
        # Используется для уделения шума.
        if self.opening_var.get():
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)

        # Инверсия изображения
        image = 255 - image

        if self.detect_var.get() == 1:
            # Create MSER object
            mser = cv2.MSER_create()

            # detect regions in gray scale image
            regions, _ = mser.detectRegions(image)
            hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

            mask = np.zeros((vis.shape[0], vis.shape[1], 1), dtype=np.uint8)

            for contour in hulls:
                cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

            # this is used to find only text regions, remaining are ignored
            image = cv2.bitwise_and(image, image, mask=mask)

        data = pytesseract.image_to_string(image, lang='eng', config='--psm 6 -c tessedit_char_whitelist=0123456789')

        # show the output image
        self.out = Image.fromarray(image)
        h, w = self.out.size
        if h > 300 or w > 300:
            self.out = self.out.resize((int(h / 2), int(w / 2)))
        self.out = ImageTk.PhotoImage(self.out)

        self.img2.config(image=self.out)
        self.img2.image = self.out

        messagebox.showinfo("Answer", data)


if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("500x300")
    app = App(master=root)
    app.mainloop()
