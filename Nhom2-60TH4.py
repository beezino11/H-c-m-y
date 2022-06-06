from __future__ import division, print_function, unicode_literals
import tkinter as tk 
import numpy as np   
import pandas as pd  
import sklearn 
from tkinter import messagebox
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from numpy import absolute
from numpy import sqrt
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import accuracy_score
df = pd.read_csv("DuLieuMau.csv", delimiter=',') 
#define predictor and response variables
X = df[['budget', 'popularity','runtime','score']].values.reshape(-1,4) # lấy ra các cột(chỉ lấy dữ liệu chuyển thành các mảng)
Y = df['revenue'] #lấy ra nhãn
#Splitting the data into Train and Test
model = BaggingRegressor(base_estimator=LinearRegression(),n_estimators=10, random_state=0) 
xtrain, xtest, ytrain, ytest = train_test_split(X,Y,test_size=0.25, random_state=42)        
model.fit(xtrain, ytrain)             # để train model data và nhãn đầu vào                                                 
y_pred = model.predict(xtest)         # tính toán nhãn dự đoán bằng model đã train
#create model cho phương trình hồi quy (cơ bản)
model_pt=LinearRegression()   # HỒI QUY TT
model_pt.fit(xtrain, ytrain)  # train model
#predict xtest
y_pred_coban=model_pt.predict(xtest)  # dự đoán Y(nhãn)
#compare the actual output values for X_test with the predicted values
print("so sánh các giá trị đầu ra thực tế cho xtest với các giá trị dự đoán") 
print(pd.DataFrame({'Thực tế': ytest, 'Dự đoán': y_pred_coban}))     #in ra nhãn thực tế và nhãn dự đoán
#Đánh giá thuật toán hồi quy tuyến tính cơ bản
print("Đánh giá thuật toán")
print('Mean Absolute Error:', metrics.mean_absolute_error(ytest, y_pred_coban))   # đánh gia sự sai khác giữa mô hình dự đoán và tệp testing(chỉ số càng nhỏ càng chính xác)
print('Mean Squared Error:', metrics.mean_squared_error(ytest, y_pred_coban))    # bình phương của sai số(sự khác biệt giữa các giá trị mô hình dự đoán và gt thực)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(ytest, y_pred_coban))) 
# R^2 on test dataset
print("R^2 test",model_pt.score(xtest, ytest))    # điểm đánh giá độ phù hợp của mô hình với bài toán này
# R^2 on train dataset
print("R^2 train",model_pt.score(xtrain, ytrain))
# R^2 trên toàn tập dataset
print("R^2 dataset",model_pt.score(X, Y))

#Tạo giao diện
master = tk.Tk()
master.title("Bài tập lớn")
master.geometry("551x320")
tk.Label(master, text="Nhập thông số liên quan để dự đoán: ").place(x=20,y=10,width=231,height=16)
tk.Label(master, text="Vốn").place(x=60,y=60,width=41,height=16)
tk.Label(master, text="Mức độ phổ biến").place(x=20,y=120,width=111,height=16)
tk.Label(master, text="Số suất chiếu").place(x=40,y=180,width=81,height=16)
tk.Label(master, text="số điểm IMDb").place(x=60,y=240,width=81,height=16)
e1 = tk.Entry(master)
e2 = tk.Entry(master)
e3 = tk.Entry(master)
e4 = tk.Entry(master)
e1.place(x=150,y=50,width=113,height=31)
e2.place(x=150,y=110,width=113,height=31)
e3.place(x=150,y=170,width=113,height=31)
e4.place(x=150,y=230,width=113,height=31)
tk.Label(master, text="Chọn chức năng:").place(x=360,y=10,width=131,height=16)
def duDoan():
    
    check1=e1.get()
    check2=e2.get()
    check3=e3.get()
    check4=e4.get()
    if(check1 == "" or check2 == "" or check3== "" or check4== ""):
        messagebox.showerror("Error", "Vui lòng không để trống thông số liên quan.")
    else:
        bien1 = float(check1)
        bien2 = float(check2)
        bien3 = float(check3)
        bien4 = float(check4)
        kqDuDoan = model.predict([[bien1,bien2,bien3,bien4]])
    messagebox.showinfo( "Kết quả dự đoán","Dự đoán doanh thu bộ phim có số vốn: "+str(bien1)+" đô, mức dộ phổ biến: "+str(bien2)+"số suất chiếu: "+str(bien3)+" và số điểm IMDb: "+str(bien4)+"là: "+str(float(kqDuDoan))+" đô")
tk.Button(master, 
          text='Dự đoán', 
          command=duDoan).place(x=390,y=50,width=93,height=28)
def pthoiquy():
    a=model_pt.coef_
    b=model_pt.intercept_
    #c=model.coef_
    #d=model.intercept_
    messagebox.showinfo( "Phương trình hồi quy","PT hồi quy bình thường có dạng: y = "+str(a[0])+" * x1 + "+str(a[1])+" * x2 + "+str(a[2])+" * x3 +"+str(b))
    #+"\n"+ " PT hồi quy với bagging có dạng: y = "+str(c[0])+" * x1 + "+str(c[1])+" * x2 + "+str(c[2])+" * x3 +"+str(d))
tk.Button(master, 
          text='PT hồi quy', 
          command=pthoiquy).place(x=390,y=90,width=93,height=28)
def danhgia():
    #compare the actual output values for X_test with the predicted values
    print("so sánh các giá trị đầu ra thực tế cho xtest với các giá trị dự đoán")
    print(pd.DataFrame({'Thực tế': ytest, 'Dự đoán': y_pred}))
    #Đánh giá thuật toán hồi quy tuyến tính có bagging
    print("Đánh giá thuật toán")
    print('Mean Absolute Error with bagging:', metrics.mean_absolute_error(ytest, y_pred))
    print('Mean Squared Error with bagging:', metrics.mean_squared_error(ytest, y_pred))
    print('Root Mean Squared Error with bagging:', np.sqrt(metrics.mean_squared_error(ytest, y_pred)))
    # R^2 on test dataset
    rsquared_test = model.score(xtest, ytest)
    print("R^2 test with bagging",rsquared_test)
    # R^2 on train dataset
    rsquared_train = model.score(xtrain, ytrain)
    print("R^2 train with bagging",rsquared_train)
    # R^2 trên toàn tập dataset
    print("R^2 dataset with bagging",model.score(X, Y))
tk.Button(master, 
          text='Đánh giá', 
          command=danhgia).place(x=390,y=130,width=93,height=28)
tk.Button(master, 
          text='Thoát', 
          command=master.quit).place(x=390,y=170,width=93,height=28)

master.mainloop()
