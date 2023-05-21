#importing relevant functions and modules 
from Detection_Script import *
import tkinter as tk
import sqlite3

#creating a database to store the user information

conn = sqlite3.connect('user_info.db')
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS user (
             id INTEGER PRIMARY KEY,
             email TEXT NOT NULL,
             password TEXT NOT NULL,
             username TEXT NOT NULL)''')

conn.commit()

#creating sign up page

def sign_up():
    email = email_entry.get()
    password = password_entry.get()
    username = username_entry.get()

    c.execute("INSERT INTO user (email, password, username) VALUES (?, ?, ?)", (email, password, username))
    conn.commit()

    sign_up_label.config(text="Sign up successful")


root = tk.Tk()
root.title("Sign Up")

email_label = tk.Label(root, text="Email")
email_label.grid(row=0, column=0)

email_entry = tk.Entry(root)
email_entry.grid(row=0, column=1)

password_label = tk.Label(root, text="Password")
password_label.grid(row=1, column=0)

password_entry = tk.Entry(root, show="*")
password_entry.grid(row=1, column=1)

username_label = tk.Label(root, text="Username")
username_label.grid(row=2, column=0)

username_entry = tk.Entry(root)
username_entry.grid(row=2, column=1)

sign_up_button = tk.Button(root, text="Sign Up", command=sign_up)
sign_up_button.grid(row=3, column=0)

sign_up_label = tk.Label(root, text="")
sign_up_label.grid(row=3, column=1)

root.mainloop()

#creating a login page

def login():
    username = username_entry.get()
    password = password_entry.get()

    c.execute("SELECT * FROM user WHERE username=?", (username,))
    result = c.fetchone()

    if not result:
        login_label.config(text="Login unsuccessful. ACCESS DENIED.")
        return

    saved_password = result[2]
    if password == saved_password:
        login_label.config(text="Login successful")
        detect_any_button.grid(row=4, column=0)
        detect_real_time_button.grid(row=4, column=1)
    else:
        login_label.config(text="Login unsuccessful. Invalid password.")


def forgot_password():
    username = username_entry.get()

    c.execute("SELECT email, password FROM user WHERE username=?", (username,))
    result = c.fetchone()

    if not result:
        forgot_password_label.config(text="User not found.")
        return

    email, password = result
    # Send email with password
    forgot_password_label.config(text=f"Password sent to {email}")


root = tk.Tk()
root.title("Login")


#creating our GUI 

username_label = tk.Label(root, text="Username")
username_label.grid(row=0, column=0)

username_entry = tk.Entry(root)
username_entry.grid(row=0, column=1)

password_label = tk.Label(root, text="Password")
password_label.grid(row=1, column=0)

password_entry = tk.Entry(root, show="*")
password_entry.grid(row=1, column=1)

login_button = tk.Button(root, text="Login", command=login)
login_button.grid(row=2, column=0)

forgot_password_button = tk.Button(root, text="Forgot Password", command=forgot_password)
forgot_password_button.grid(row=2, column=1)

login_label = tk.Label(root, text="")
login_label.grid(row=3, column=0, columnspan=2)

forgot_password_label = tk.Label(root, text="")
forgot_password_label.grid(row=4, column=0, columnspan=2)

detect_any_button = tk.Button(root, text="Detect")
detect_real_time_button = tk.Button(root, text="Detect Real Time")

root.mainloop()


#getting model from the web
#ModelURL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.tar.gz'
#ModelURL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz'
ModelURL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8.tar.gz'

#setting up our paths for images and videos
File_class = 'coco.names' 
image_path = r'C:\Users\Miraj\Desktop\TENSORFLOW_OBJECT_DETECTION\test\bridge.jpg'
threshold = 0.5
videopath = 0


object_detector = Object_Detector()
object_detector.class_read(File_class)
object_detector.model_download(ModelURL)
object_detector.Model_load()
#object_detector.image_prediction(image_path,threshold) 
object_detector.video_prediction(videopath,threshold)                            

