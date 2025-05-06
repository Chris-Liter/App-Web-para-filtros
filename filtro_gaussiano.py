import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


#Inicializar CUDA al cargar el módulo (una sola vez)
cuda.init()
device = cuda.Device(0)

def generar_mascara_gaussiana(tamaño, sigma):
    kernel = np.zeros((tamaño, tamaño), dtype=np.float32)
    centro = tamaño // 2
    suma = 0.0
    for i in range(tamaño):
        for j in range(tamaño):
            x = i - centro
            y = j - centro
            valor = (1.0 / (2.0 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            kernel[i, j] = valor
            suma += valor
    return kernel / suma

def aplicar_filtro_cuda(imagen, mascara, block_x=32, block_y=32):
    context = device.make_context() 

    try:
        height, width, channels = imagen.shape
        tamaño = mascara.shape[0]
        MAX_SIZE = 51

        cuda_kernel = """
        __global__ void gauss_filter(unsigned char *input, unsigned char *output, float *mascara,
                             int width, int height, int channels, int mask_size, int mask_radius) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= width || y >= height) return;

            for (int c = 0; c < channels; c++) {
                float suma = 0.0f;
                for (int i = -mask_radius; i <= mask_radius; i++) {
                    for (int j = -mask_radius; j <= mask_radius; j++) {
                        int nx = x + j;
                        int ny = y + i;
                        if (nx >= 0 && ny >= 0 && nx < width && ny < height) {
                            float peso = mascara[(i + mask_radius) * mask_size + (j + mask_radius)];
                            unsigned char pixel = input[(ny * width + nx) * channels + c];
                            suma += pixel * peso;
                        }
                    }
                }
                output[(y * width + x) * channels + c] = fminf(255.0f, fmaxf(0.0f, suma));
            }
        }
        """

        mod = SourceModule(cuda_kernel)
        gauss_filter = mod.get_function("gauss_filter")

        mascara_gpu = cuda.mem_alloc(mascara.nbytes)
        cuda.memcpy_htod(mascara_gpu, mascara.astype(np.float32))

        img_gpu = cuda.mem_alloc(imagen.nbytes)
        output_gpu = cuda.mem_alloc(imagen.nbytes)
        cuda.memcpy_htod(img_gpu, imagen)

        block = (block_x, block_y, 1)
        grid = ((width + block[0] - 1) // block[0], (height + block[1] - 1) // block[1])

        start = cuda.Event()
        end = cuda.Event()
        start.record()

        gauss_filter(img_gpu, output_gpu, mascara_gpu,
                    np.int32(width), np.int32(height), np.int32(channels),
                    np.int32(tamaño), np.int32(tamaño // 2), block=block, grid=grid)

        end.record()
        end.synchronize()
        tiempo = start.time_till(end)

        grid_x = (width + block_x - 1) // block_x
        grid_y = (height + block_y - 1) // block_y

        salida = np.empty_like(imagen)
        cuda.memcpy_dtoh(salida, output_gpu)

        # Liberar memoria
        img_gpu.free()
        output_gpu.free()
        mascara_gpu.free()

        return salida, tiempo, grid_x, grid_y

    finally:
        context.pop()  #liberamos el contexto qui
