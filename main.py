from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
from tkinter import filedialog as fd
import numpy as np
import cv2, time, os
from PIL import Image, ImageTk
from image_processing import train_images, test_image

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"assets/frame0")

# Inisialisasi variabel global
pict_path = []
mean_face = []
EigFace = []
Om = []

# Load data from files
def load_data():
    global pict_path, mean_face, EigFace, Om
    try:
        pict_path = np.genfromtxt("data/pict_path.txt", dtype="str", delimiter="\n")
        mean_face = np.loadtxt("data/mean_face.txt")
        EigFace = np.loadtxt("data/EigFace.txt")
        Om = np.loadtxt("data/Om.txt")
    except FileNotFoundError:
        print("Data files not found. Please train the model first.")
    return pict_path, mean_face, EigFace, Om

# Check if data exists, if so; load from file
if os.path.exists("data/pict_path.txt"):
    pict_path, mean_face, EigFace, Om = load_data()

# Return relative path to assets/
def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

load_time = 0

# Iterate through directory, load images
def select_dir():
    global pict_path, mean_face, EigFace, Om, load_time
    filepath = fd.askdirectory(initialdir="../", title='Choose directory')
    if filepath:
        start = time.time()
        try:
            pict_path, mean_face, EigFace, Om = train_images(filepath)
            if not os.path.exists("data/"):
                os.makedirs("data/")
            np.savetxt("data/pict_path.txt", pict_path, fmt="%s")
            np.savetxt("data/mean_face.txt", mean_face)
            np.savetxt("data/EigFace.txt", EigFace)
            np.savetxt("data/Om.txt", Om)
            load_data()
            load_time = str(round(time.time() - start, 5))
            canvas.itemconfig(exec_time, text=load_time)
        except Exception as e:
            print(f"Error during training: {e}")

is_input = True

# Change image in Test Image and Closest Result
def change_image(img, path):
    global image_image_2, image_image_3
    try:
        img_resized = img.resize((256, 256), Image.BICUBIC)
        image_image_2 = ImageTk.PhotoImage(img_resized)
        canvas.itemconfig(image_2, image=image_image_2)
        
        output_img = Image.open(path).resize((256, 256), Image.BICUBIC)
        image_image_3 = ImageTk.PhotoImage(output_img)
        canvas.itemconfig(image_3, image=image_image_3)
    except Exception as e:
        print(f"Error changing image: {e}")

# Select an image file
def select_file():
    global is_input, mean_face
    if not is_input:
        print("Silahkan ganti mode terlebih dahulu!")
    elif len(mean_face) != 0:
        filetypes = [("JPG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
        filename = fd.askopenfilename(initialdir='../', title='Choose file', filetypes=filetypes)
        if filename:
            try:
                img = Image.open(filename).resize((256, 256), Image.BICUBIC)
                output = test_image(img, pict_path, mean_face, EigFace, Om)
                change_image(img, output)
            except Exception as e:
                print(f"Error processing image: {e}")
    else:
        print("Silahkan lakukan training terlebih dahulu!")

# Run camera
def run_camera():
    global is_input, mean_face, image_image_2, canvas
    if is_input:
        print("Silahkan ganti mode terlebih dahulu!")
    elif len(mean_face) != 0:
        try:
            ret, frame = cam.read()
            if ret:
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                width, height = img.size
                diff = width - height
                img = img.crop((diff, 0, diff + height, height))
                output = test_image(img, pict_path, mean_face, EigFace, Om)
                change_image(img, output)
                canvas.after(20, run_camera)
        except Exception as e:
            print(f"Error running camera: {e}")
    else:
        print("Silahkan lakukan training terlebih dahulu!")

# Change button text
def change_mode():
    global is_input, image_image_2
    is_input = not is_input
    button_3.config(text="Camera Mode" if not is_input else "Input Mode")
    if is_input:
        try:
            img = Image.open(relative_to_assets("image_2.png"))
            image_image_2 = ImageTk.PhotoImage(img)
            canvas.itemconfig(image_2, image=image_image_2)
        except FileNotFoundError:
            print("Default image not found.")
    else:
        run_camera()

cam = cv2.VideoCapture(0)

window = Tk()
window.title("Image Recognition")
window.geometry("960x540")
window.configure(bg="#FFFFFF")
window.resizable(False, False)

canvas = Canvas(window, bg="#FFFFFF", height=540, width=960, bd=0, highlightthickness=0, relief="ridge")
canvas.pack()
canvas.place(x=0, y=0)

image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(480.0, 270.0, image=image_image_1)

button_1 = Button(text="Choose File", borderwidth=0, highlightthickness=0, command=select_file, relief="flat")
button_1.place(x=66.0, y=464.0, width=140.0, height=32.0)

button_2 = Button(text="Choose Directory", borderwidth=0, highlightthickness=0, command=select_dir, relief="flat")
button_2.place(x=308.0, y=464.0, width=140.0, height=32.0)

button_3 = Button(text="Input Mode", borderwidth=0, highlightthickness=0, command=change_mode, relief="flat")
button_3.place(x=415.0, y=255.0, width=130.0, height=32.0)

canvas.create_text(301.0, 430.0, anchor="nw", text="Insert Your Dataset", fill="#000000", font=("K2D Light", 18 * -1))
canvas.create_text(52.0, 430.0, anchor="nw", text="Insert Your Image\n", fill="#000000", font=("K2D Light", 18 * -1))
canvas.create_text(777.0, 430.0, anchor="nw", text="Excecution Time", fill="#000000", font=("K2D Light", 18 * -1))

exec_time = canvas.create_text(777.0, 466.0, anchor="nw", text="0", fill="#000000", font=("K2D Light", 18 * -1))

image_image_2 = ImageTk.PhotoImage(file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(250.0, 265.0, image=image_image_2)

image_image_3 = ImageTk.PhotoImage(file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(715.0, 265.0, image=image_image_3)

canvas.create_text(202.0, 400.0, anchor="nw", text="Test Image", fill="#000000", font=("Kadwa Regular", 18 * -1))
canvas.create_text(655.0, 400.0, anchor="nw", text="Closest Result", fill="#000000", font=("Kadwa Regular", 18 * -1))

window.mainloop()