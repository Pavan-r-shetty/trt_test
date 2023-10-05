import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time

# Specify your model file here
MODEL_FILE_PATH = "model.trt"  # <-- EDIT this line with your file path
INPUT_SHAPE = (1, 80, 8)  # Hard-coded input shape
class ModelRT(object):
    def __init__(self, m_file, input_shape=None):
        self.m_file = m_file
        print(f'Loading {self.m_file}.')

        self.input_shape = INPUT_SHAPE
        if not self.input_shape:
            print('Could not determine model input shape.')
            return

        self.init_model()
        self.timer = Timer()


    def init_model(self):
        # Load model and set up engine
        f = open(self.m_file, "rb")
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate device memory
        model_input = np.ones(self.input_shape, dtype=np.float32)
        self.output = np.empty([1, self.input_shape[0]], dtype=np.float32)
        self.output2 = np.empty([1, self.input_shape[0]], dtype=np.float32)
        self.d_input = cuda.mem_alloc(1 * model_input.nbytes)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)
        self.d_output2 = cuda.mem_alloc(1 * self.output2.nbytes)
        self.bindings = [int(self.d_input), int(self.d_output), int(self.d_output2)]

        # Create stream to transfer data between cpu and gpu
        self.stream = cuda.Stream()



    def predict(self, model_input):
        cuda.memcpy_htod_async(self.d_input, model_input, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        cuda.memcpy_dtoh_async(self.output2, self.d_output2, self.stream)
        self.stream.synchronize()
        return self.output, self.output2

   
    def test_model(self, num_tests=1, verbose=True):
        o_all = []
        t_all = []
        

        for i in range(1, num_tests+1):
            o = []
            t = []
            time_s = time.perf_counter()

            for j in range(100):
                data = np.random.rand(INPUT_SHAPE[1], INPUT_SHAPE[2])
                model_input = np.expand_dims(data, axis=0).astype(np.float32)

                output = self.predict(model_input)
                o.append(output)
                time_e = time.perf_counter()
                t.append(time_e - time_s)
                time_s = time_e

            if verbose: print(f'Test {i}: Avg Time: {np.mean(t) * 1000} ms +/- {np.std(t) * 1000} ms')
            # if verbose: print(output)
            t_all.append(np.mean(t))
            o_all.append(o)

        return o_all, t_all

class Timer():
    def __init__(self):
        self.start()

    def start(self):
        self.start_time = time.perf_counter()

    def end(self, endl="\n"):
        print(np.round((time.perf_counter()-self.start_time)*1000, 2), end=endl)

if __name__ == "__main__":
    model = ModelRT(MODEL_FILE_PATH)
    model.test_model(num_tests=10, verbose=True)
