import sys, os
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QMessageBox, QDialog

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
import cv2
import numpy as np
from keras.saving import load_model


def reshape(X):
  X = np.array(X)
  try:
    _, token_size, vector_size = model.layers[0].input_shape
    reshape = True
  except:
    try:
      _, token_size, vector_size = model.layers[0].input_shape[0]
      reshape = True
    except:
      reshape = False
  if reshape:
    X = np.reshape(X, (X.shape[0], token_size, vector_size))
  return X


class DigitClassifier: # A bit different than the one on the notebook.
  def __init__(self, model, thres=100, dilation=3): # Keras models only
    self.model = model
    self.thres = thres
    self.dilation = dilation

  def inference(self, image):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bin_image = cv2.threshold(grey_image, self.thres, 255, cv2.THRESH_BINARY_INV)
    bin_image = cv2.dilate(bin_image, np.ones((3, 3)), iterations = self.dilation)
    contours, _ = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits = []
    for c in contours:
      x, y, w, h = cv2.boundingRect(c)
      if w + h <= 56:
        continue
      cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness = 1)
      digit = bin_image[y:y + h, x:x + w]
      if w <= h: # h' = 20, w'/h' = w/h
       digit = cv2.resize(digit, (w*20//h, 20))
      elif w > h: # w' = 20, w'/h' = w/h
        digit = cv2.resize(digit, (20, 20*h//w))
      w0 = 28 - digit.shape[0]
      h0 = 28 - digit.shape[1]
      if(w0 % 2 == 1):
        w01 = w0//2 - 1
      else:
        w01 = w0//2
      if(h0 % 2 == 1):
        h01 = h0//2 - 1
      else:
        h01 = h0//2
      w02 = w0 - w01
      h02 = h0 - h01
      digit = np.pad(digit, ((w01, w02), (h01, h02)), "constant", constant_values=0)
      digits.append(digit)
    digits = np.array(digits)
    if(len(digits) > 0):
        predictions = np.argmax(model.predict(reshape(digits), verbose=0), axis=-1)
    else:
        predictions = []
    return predictions, image


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText('\n\n Drop Image Here \n\n')
        #self.setStyleSheet('''
        #    QLabel{border: 4px dashed #aaa}
        #''')

    def setPixmap(self, image):
        super().setPixmap(image)

class DigitRecognitionApp(QWidget):
    def __init__(self, model):
        super().__init__()
        self.resize(400, 400)
        self.setAcceptDrops(True)
        mainLayout = QVBoxLayout()
        self.photoViewer = ImageLabel()
        mainLayout.addWidget(self.photoViewer)
        self.setLayout(mainLayout)
        self.classifier = DigitClassifier(model)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            event.setDropAction(Qt.DropAction.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()
            image = cv2.imread(file_path)
            predictions, image = self.classifier.inference(image)
            predictions.sort()
            #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.photoViewer.setPixmap(pixmap)
            dlg = QMessageBox(self)
            dlg.setWindowTitle('My prediction')
            if(len(predictions) == 0):    
                dlg.setText('I saw nothing.')
            else:
                dlg.setText('I saw ' + str(predictions))
            dlg.setStandardButtons(QMessageBox.StandardButton.Ok)
            dlg.setIcon(QMessageBox.Icon.Information)
            dlg.exec()
        else:
            event.ignore()


app = QApplication(sys.argv)
model = load_model('CNN.keras')
exe = DigitRecognitionApp(model)
exe.show()
sys.exit(app.exec())