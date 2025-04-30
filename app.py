# from flask import Flask
# app = Flask(__name__)

# @app.route("/")
# def hello():
#     return "Hello World!"


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=int("5000"), debug = True)
    
    
    
from flask import Flask, render_template, Response, stream_with_context, Request, request, jsonify
from io import BytesIO

import json
import base64
from PIL import Image

import pycuda.driver as cuda
import cv2
import time 
from pycuda.compiler import SourceModule
import numpy as np
from flask_cors import CORS




app = Flask(__name__)
CORS(app)

@app.route("/")
def hello():
    res = "%d device(s) found." % (cuda.Device.count())

    for ordinal in range(cuda.Device.count()):
        dev = cuda.Device(ordinal)
        res += "Device #%d: %s" % (ordinal, dev.name())
        res += " Compute  Capability: %d.%d" %dev.compute_capability()
        res += " Total Memory: %s KB" % (dev.total_memory()//(1024))
    return res



def generate_gabor_kernel(ksize, sigma, theta, lambd, psi, gamma):
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    half = ksize // 2
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    for y in range(-half, half+1):
        for x in range(-half, half+1):
            x_theta = x * cos_theta + y * sin_theta
            y_theta = -x * sin_theta + y * cos_theta
            gauss = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2))
            wave = np.cos(2 * np.pi * x_theta / lambd + psi)
            kernel[y + half, x + half] = gauss * wave

    return kernel

####################################################################
##########################CODIGOS CUDA C############################
##GABOR
kernel_code = """
__global__ void applyGaborCUDA(uchar3* input, uchar3* output, float* kernel, 
                               int ksize, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half = ksize / 2;

    if (x >= half && y >= half && x < width - half && y < height - half) {
        float sumB = 0.0f, sumG = 0.0f, sumR = 0.0f;
        for (int ky = -half; ky <= half; ky++) {
            for (int kx = -half; kx <= half; kx++) {
                int imgX = x + kx;
                int imgY = y + ky;
                int idx = imgY * width + imgX;
                int kidx = (ky + half) * ksize + (kx + half);
                uchar3 pixel = input[idx];
                float weight = kernel[kidx];
                sumR += weight * pixel.x;  // Red
                sumG += weight * pixel.y;  // Green
                sumB += weight * pixel.z;  // Blue

            }
        }
        int outIdx = y * width + x;
        output[outIdx].x = min(max(int(sumR), 0), 255);
        output[outIdx].y = min(max(int(sumG), 0), 255);
        output[outIdx].z = min(max(int(sumB), 0), 255);

    }
}
"""
##LAPLACIANO

##GAUSSIANO

####################################################################



@app.route("/laplaciano")
def laplaciano():
    return


@app.route("/gabor", methods=['POST'])
def gabor():
        
    cuda.init()
    dev = cuda.Device(0)
    ctx = dev.make_context() 
    
    if request.method == 'POST':
        try:
            # Verificar si el archivo fue enviado
            if 'image' not in request.files:
                return jsonify({"error": "No se envió la imagen"}), 400
            
            if 'mask' not in request.form:
                return jsonify({"error": "No se envió el parámetro 'mask'"}), 400
            
            mask = int(request.form['mask'])

            # Leer la imagen desde el archivo recibido
            uploaded_file = request.files['image']
            image_bytes = uploaded_file.read()
            
            npimg = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            
            mod = SourceModule(kernel_code)
            apply_gabor = mod.get_function("applyGaborCUDA")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]
            img_flat = img.reshape(-1, 3).astype(np.uint8)
            
            kernel = generate_gabor_kernel(mask, 5.0, np.pi/4, 10.0, 2.0, 2.0)
            
            d_input = cuda.mem_alloc(img_flat.nbytes)
            d_output = cuda.mem_alloc(img_flat.nbytes)
            d_kernel = cuda.mem_alloc(kernel.nbytes)

            cuda.memcpy_htod(d_input, img_flat)
            cuda.memcpy_htod(d_kernel, kernel)

            # Ejecutar kernel
            block = (16, 16, 1)
            grid = ((width + 15) // 16, (height + 15) // 16)

            start = time.time()
            apply_gabor(d_input, d_output, d_kernel, 
                        np.int32(mask), np.int32(width), np.int32(height), 
                        block=block, grid=grid)
            cuda.Context.synchronize()
            end = time.time()
            
            
            output = np.empty_like(img_flat)
            cuda.memcpy_dtoh(output, d_output)
            output_img = output.reshape((height, width, 3))
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
            
            _, buffer = cv2.imencode('.jpg', output_img)
            base64_img = base64.b64encode(buffer).decode('utf-8')
            result = (end-start)
            
            ctx.pop()
            print(result)
            return jsonify({
                "imagen": base64_img,
                "tiempo": result,
            })
                    
        except Exception as e:
            print(e)
            return jsonify({"error": str(e)}), 500

    


@app.route("/gaussiano")
def gaussiano():
    return

    
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("5000"), debug = True)
    
    

