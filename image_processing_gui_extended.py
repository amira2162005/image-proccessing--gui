import tkinter as tk
from tkinter import filedialog, simpledialog
import cv2
import numpy as np
import matplotlib.pyplot as plt

selected_image = None

# اختيار صورة
def choose_image():
    global selected_image
    file_path = filedialog.askopenfilename()
    if file_path:
        selected_image = cv2.imread(file_path)
        choose_btn.config(text=file_path.split("/")[-1])

# عرض الصورة الأصلية والنتيجة
def show_result(original, result, title="Result"):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# العمليات النقطية
def point_operation(op_name):
    global selected_image
    if selected_image is None:
        return

    img = selected_image.copy()  # استخدام نسخة جديدة من الصورة الأصلية
    result = None

    if op_name == "Addition":
        val = simpledialog.askinteger("Input", "Enter value to add (e.g. 50):", minvalue=0, maxvalue=255)
        if val is not None:
            result = cv2.add(img, np.full(img.shape, val, dtype=np.uint8))
    elif op_name == "Subtraction":
        val = simpledialog.askinteger("Input", "Enter value to subtract (e.g. 50):", minvalue=0, maxvalue=255)
        if val is not None:
            result = cv2.subtract(img, np.full(img.shape, val, dtype=np.uint8))
    elif op_name == "Division":
        val = simpledialog.askinteger("Input", "Enter divisor (e.g. 2):", minvalue=1)
        if val is not None:
            result = cv2.divide(img, np.full(img.shape, val, dtype=np.uint8))
    elif op_name == "Complement":
        result = cv2.subtract(255, img)

    if result is not None:
        show_result(img, result, op_name)

# العمليات اللونية
def color_operation(op_name):
    global selected_image
    if selected_image is None:
        return

    img = selected_image.copy() 
    result = None
    if op_name == "Change Red":
        img[:, :, 2] = np.clip(img[:, :, 2] + 50, 0, 255)
        result = img
    elif op_name == "Swap R to G":
        temp = img[:, :, 1].copy()
        img[:, :, 1] = img[:, :, 2]
        img[:, :, 2] = temp
        result = img
    elif op_name == "Eliminate Red":
        img[:, :, 2] = 0
        result = img
    elif op_name == "Swap R to B":
        temp = img[:, :, 0].copy()
        img[:, :, 0] = img[:, :, 2]
        img[:, :, 2] = temp
        result = img
    if result is not None:
        show_result(selected_image, result, op_name)

# العمليات المتعلقة بالمدرج التكراري (Histogram)
def histogram_operation(op_name):
    global selected_image
    if selected_image is None:
        return

    img = selected_image.copy()  # استخدام نسخة جديدة من الصورة الأصلية
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = None

    if op_name == "Histogram Stretching":
        min_val, max_val = np.min(gray), np.max(gray)
        result = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    elif op_name == "Histogram Equalization":
        result = cv2.equalizeHist(gray)

    if result is not None:
        result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        show_result(img, result_bgr, op_name)

# العمليات الخاصة بالجوار (Neighborhood Processing)
def neighborhood_operation(op_name):
    global selected_image
    if selected_image is None:
        return

    img = selected_image.copy()  # استخدام نسخة جديدة من الصورة الأصلية
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = None

    if op_name == "Average Filter (Linear)":
        result = cv2.blur(gray, (3, 3))
    elif op_name == "Laplacian Filter (Linear)":
        result = cv2.Laplacian(gray, cv2.CV_64F)
        result = cv2.convertScaleAbs(result)
    elif op_name == "Maximum Filter (Non-Linear)":
        result = cv2.dilate(gray, np.ones((3, 3), np.uint8))
    elif op_name == "Minimum Filter (Non-Linear)":
        result = cv2.erode(gray, np.ones((3, 3), np.uint8))
    elif op_name == "Median Filter (Non-Linear)":
        result = cv2.medianBlur(gray, 3)
    elif op_name == "Mode Filter (Non-Linear)":
        result = cv2.erode(gray, np.ones((3, 3), np.uint8))  # Mode filter approximation

    if result is not None:
        result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        show_result(img, result_bgr, op_name)

# العمليات الخاصة بالاستعادة (Restoration)
def restoration_operation(op_name):
    global selected_image
    if selected_image is None:
        return

    img = selected_image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result = None
    # إعادة ضبط طرق معالجة الضوضاء
    if op_name == "Salt & Pepper - Average":
        noisy_image = add_salt_and_pepper_noise(gray, amount=0.05)
        result = cv2.blur(noisy_image, (3, 3))
    elif op_name == "Salt & Pepper - Median":
        noisy_image = add_salt_and_pepper_noise(gray, amount=0.05)
        result = cv2.medianBlur(noisy_image, 3)
    elif op_name == "Salt & Pepper - Outlier":
        noisy_image = add_salt_and_pepper_noise(gray, amount=0.05)
        result = cv2.erode(noisy_image, np.ones((5, 5), np.uint8))  # Approximating outlier filtering
    elif op_name == "Gaussian - Averaging":
        noisy_image = gray + np.random.normal(0, 25, gray.shape).astype(np.uint8)
        result = cv2.blur(noisy_image, (3, 3))  # تطبيق فلتر المتوسط
    elif op_name == "Gaussian - Average Filter":
        noisy_image = gray + np.random.normal(0, 25, gray.shape).astype(np.uint8)
        result = cv2.GaussianBlur(noisy_image, (3, 3), 0)  # تطبيق فلتر Gaussian

    if result is not None:
        result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        show_result(img, result_bgr, op_name)

def add_salt_and_pepper_noise(image, amount=0.05):
    """إضافة ضوضاء الملح والفلفل إلى الصورة"""
    noisy_image = image.copy()
    num_salt = int(amount * image.size * 0.5)
    num_pepper = int(amount * image.size * 0.5)

    # إضافة ضوضاء الملح
    salt_coords = [np.random.randint(0, i-1, num_salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    # إضافة ضوضاء الفلفل
    pepper_coords = [np.random.randint(0, i-1, num_pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image

# العمليات الخاصة بالتقسيم (Segmentation)
def segmentation_operation(op_name):
    global selected_image
    if selected_image is None:
        return

    img = selected_image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result = None
    if op_name == "Basic Global Thresholding":
        _, result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    elif op_name == "Automatic Thresholding":
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    elif op_name == "Adaptive Thresholding":
        result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    if result is not None:
        result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        show_result(img, result_bgr, op_name)

# العمليات الخاصة بالكشف عن الحواف (Edge Detection)
def edge_detection_operation(op_name):
    global selected_image
    if selected_image is None:
        return

    img = selected_image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result = None
    if op_name == "Sobel Detector":
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        result = cv2.sqrt(grad_x*2 + grad_y*2)
        result = cv2.convertScaleAbs(result)

    if result is not None:
        result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        show_result(img, result_bgr, op_name)

# العمليات الخاصة بالمورفولوجيا الرياضية (Mathematical Morphology)
def morphology_operation(op_name):
    global selected_image
    if selected_image is None:
        return

    img = selected_image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result = None
    kernel = np.ones((5, 5), np.uint8)

    if op_name == "Image Dilation":
        result = cv2.dilate(gray, kernel)
    elif op_name == "Image Erosion":
        result = cv2.erode(gray, kernel)
    elif op_name == "Image Opening":
        result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    elif op_name == "Internal Boundary":
        result = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    elif op_name == "External Boundary":
        result = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    elif op_name == "Morphological Gradient":
        result = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    if result is not None:
        result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        show_result(img, result_bgr, op_name)

# الأقسام مع العمليات
sections = {
    "Point Operations": ["Addition", "Subtraction", "Division", "Complement"],
    "Color Operations": ["Change Red", "Swap R to G", "Eliminate Red", "Swap R to B"],
    "Histogram": ["Histogram Stretching", "Histogram Equalization"],
    "Neighborhood": [
        "Average Filter (Linear)", "Laplacian Filter (Linear)",
        "Maximum Filter (Non-Linear)", "Minimum Filter (Non-Linear)",
        "Median Filter (Non-Linear)", "Mode Filter (Non-Linear)"
    ],
    "Restoration": [
        "Salt & Pepper - Average", "Salt & Pepper - Median",
        "Salt & Pepper - Outlier", "Gaussian - Averaging", "Gaussian - Average Filter"
    ],
    "Image Segmentation": [
        "Basic Global Thresholding", "Automatic Thresholding", "Adaptive Thresholding"
    ],
    "Edge Detection": ["Sobel Detector"],
    "Mathematical Morphology": [
        "Image Dilation", "Image Erosion", "Image Opening", "Internal Boundary",
        "External Boundary", "Morphological Gradient"
    ]
}

# إضافة واجهة المستخدم
root = tk.Tk()
root.title("Image Processing GUI")
root.geometry("950x750")
root.configure(bg="#ffe4ec")  # خلفية وردية فاتحة

# زر اختيار الصورة
choose_btn = tk.Button(root, text="اختر صورة", command=choose_image,
            bg="#ffb6c1", font=("Arial", 11, "bold"), relief="flat")
choose_btn.pack(pady=15, padx=10, anchor="w")

# إطار الأزرار الرئيسية
main_frame = tk.Frame(root, bg="#ffe4ec")
main_frame.pack(pady=10)

# إنشاء الأزرار الرئيسية أفقياً
section_frames = {}
for i, (section, operations) in enumerate(sections.items()):
    frame = tk.Frame(root, bg="#fff0f5")
    section_frames[section] = frame

    btn = tk.Button(main_frame, text=section, bg="#ff99cc",
            font=("Arial", 10, "bold"), relief="groove",
            command=lambda f=frame: toggle_section(f))
    btn.grid(row=0, column=i, padx=5)

    # إنشاء الأزرار الفرعية
    for op in operations:
        action_btn = tk.Button(frame, text=op, bg="#ffd6e8",
                    font=("Arial", 10), width=25,
                    command=lambda o=op, s=section: handle_operation(o, s))
        action_btn.pack(anchor="w", padx=20, pady=3)

# التعامل مع العمليات بناءً على القسم
def handle_operation(op_name, section):
    if section == "Point Operations":
        point_operation(op_name)
    elif section == "Color Operations":
        color_operation(op_name)
    elif section == "Histogram":
        histogram_operation(op_name)
    elif section == "Neighborhood":
        neighborhood_operation(op_name)
    elif section == "Restoration":
        restoration_operation(op_name)
    elif section == "Image Segmentation":
        segmentation_operation(op_name)
    elif section == "Edge Detection":
        edge_detection_operation(op_name)
    elif section == "Mathematical Morphology":
        morphology_operation(op_name)

# تبديل إظهار/إخفاء الأقسام
def toggle_section(frame):
    if frame.winfo_ismapped():
        frame.pack_forget()
    else:
        frame.pack(fill='x', padx=30, pady=5, anchor="w")

# تشغيل الواجهة
root.mainloop()