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
from filtro_gaussiano import generar_mascara_gaussiana, aplicar_filtro_cuda
import traceback



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
    # Si ksize es par, se incrementa para que sea impar
    if ksize % 2 == 0:
        ksize += 1

    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    half = ksize // 2
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    for y in range(-half, half + 1):
        for x in range(-half, half + 1):
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

from pycuda import driver as cuda
from pycuda.compiler import SourceModule

@app.route("/laplaciano", methods=["POST"])
def laplaciano():
    try:
        # Inicializar CUDA y crear contexto
        cuda.init()
        device = cuda.Device(0)
        context = device.make_context()

        if 'image' not in request.files:
            return jsonify({"error": "No se envió la imagen"}), 400

        tamaño = int(request.form.get("mask", 21))
        if tamaño % 2 == 0 or tamaño < 3:
            return jsonify({"error": "El tamaño de la máscara debe ser impar y >= 3"}), 400

        block_x = int(request.form.get("block_x", 16))
        block_y = int(request.form.get("block_y", 16))

        archivo = request.files['image']
        archivo_np = np.frombuffer(archivo.read(), np.uint8)
        imagen = cv2.imdecode(archivo_np, cv2.IMREAD_GRAYSCALE)

        if imagen is None:
            return jsonify({"error": "Imagen no válida"}), 400

        imagen_float = imagen.astype(np.float32)
        laplaciana = -1 * np.ones((tamaño, tamaño), dtype=np.float32)
        centro = tamaño // 2
        laplaciana[centro, centro] = (tamaño * tamaño) - 1
        laplaciana_flat = laplaciana.flatten()

        altura, ancho = imagen.shape
        salida_gpu = np.zeros_like(imagen_float)

        mod = SourceModule("""
        __global__ void filtro_laplaciano(float *imagen, float *mascara, float *salida, int ancho, int alto, int offset, int tamano_mascara) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= offset && x < (ancho - offset) && y >= offset && y < (alto - offset)) {
                float valor = 0.0;
                for (int i = -offset; i <= offset; i++) {
                    for (int j = -offset; j <= offset; j++) {
                        int idx = (i + offset) * tamano_mascara + (j + offset);
                        valor += imagen[(y + i) * ancho + (x + j)] * mascara[idx];
                    }
                }
                salida[y * ancho + x] = fminf(fmaxf(valor, 0.0), 255.0);
            }
        }
        """)

        filtro_laplaciano_gpu = mod.get_function("filtro_laplaciano")

        imagen_gpu = cuda.mem_alloc(imagen_float.nbytes)
        cuda.memcpy_htod(imagen_gpu, imagen_float)

        salida_gpu_mem = cuda.mem_alloc(salida_gpu.nbytes)

        mascara_gpu = cuda.mem_alloc(laplaciana_flat.nbytes)
        cuda.memcpy_htod(mascara_gpu, laplaciana_flat)

        block_size = (block_x, block_y, 1)
        grid_size = ((ancho + block_size[0] - 1) // block_size[0],
                     (altura + block_size[1] - 1) // block_size[1])

        offset = tamaño // 2

        start_gpu = cuda.Event()
        end_gpu = cuda.Event()
        start_gpu.record()

        filtro_laplaciano_gpu(imagen_gpu, mascara_gpu, salida_gpu_mem,
                              np.int32(ancho), np.int32(altura), np.int32(offset), np.int32(tamaño),
                              block=block_size, grid=grid_size)

        end_gpu.record()
        end_gpu.synchronize()
        tiempo_gpu = start_gpu.time_till(end_gpu) / 1000.0

        cuda.memcpy_dtoh(salida_gpu, salida_gpu_mem)

        resultado_uint8 = salida_gpu.astype(np.uint8)
        _, buffer = cv2.imencode('.jpg', resultado_uint8)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        # Liberar el contexto CUDA
        context.pop()

        return jsonify({
            "imagen": base64_image,
            "tiempo_ejecucion": round(tiempo_gpu, 8),
            "filtro": "laplaciano",
            "mask": tamaño,
            "block_x": block_x,
            "block_y": block_y,
            "grid_size": {"x": grid_size[0], "y": grid_size[1]},
            "threads_total": block_x * block_y,
            "blocks_total": grid_size[0] * grid_size[1]
        })

    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



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
            block_x = int(request.form.get("block_x", 32))
            block_y = int(request.form.get("block_y", 32))

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
            block = (block_x, block_y, 1)
            grid =  ((width + block[0] - 1) // block[0], (height + block[1] - 1) // block[1])

            grid_x, grid_y = grid

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
            
            
            print(result)
            return jsonify({
                "imagen": base64_img,
                "tiempo_ejecucion": round(result, 8),
                "filtro": "gabor",
                "mask": mask,
                "sigma": 0,
                "block_x": block_x,
                "block_y": block_y,
                "grid_size": {"x": grid_x, "y": grid_y},
                "threads_total": block_x * block_y,
                "blocks_total": grid_x * grid_y
            })
                    
        except Exception as e:
            print(str(e))
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500
        finally:
            ctx.pop()

    


@app.route("/gaussiano", methods=['POST'])
def gaussiano():
    if 'image' not in request.files:
        return jsonify({"error": "No se envió una imagen"}), 400

    try:
        tamaño = int(request.form.get("mask", 21))
        sigma = float(request.form.get("sigma", 10.0))
        block_x = int(request.form.get("block_x", 32))
        block_y = int(request.form.get("block_y", 32))
        archivo = request.files['image']
        archivo_np = np.frombuffer(archivo.read(), np.uint8)
        imagen = cv2.imdecode(archivo_np, cv2.IMREAD_COLOR)

        if imagen is None:
            return jsonify({"error": "Imagen no válida"}), 400

        mascara = generar_mascara_gaussiana(tamaño, sigma)
        #resultado, tiempo = aplicar_filtro_cuda(imagen, mascara, block_x, block_y)
        resultado, tiempo, grid_x, grid_y = aplicar_filtro_cuda(imagen, mascara, block_x, block_y)

        _, buffer = cv2.imencode('.jpg', resultado)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "imagen": base64_image,
            "tiempo_ejecucion": round(tiempo, 8),
            "filtro": "gaussiano",
            "mask": tamaño,
            "sigma": sigma,
            "block_x": block_x,
            "block_y": block_y,
            "grid_size": {"x": grid_x, "y": grid_y},
            "threads_total": block_x * block_y,
            "blocks_total": grid_x * grid_y
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

    
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("5000"), debug = True)
    
    

