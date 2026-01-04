# ...existing code...
import time
import numpy as np
import vpi
import cv2
from threading import Thread

MAX_DISP = 64
WINDOW_SIZE = 10

# Prefer camera_stream.CameraStream -> jetvision.elements.Camera -> cv2.VideoCapture
CameraProvider = None
try:
    from camera_stream import CameraStream as _CameraStream  # repo-provided stream
    CameraProvider = ("stream", _CameraStream)
except Exception:
    try:
        from jetvision.elements import Camera as _Camera  # previous code used this
        CameraProvider = ("jetvision", _Camera)
    except Exception:
        CameraProvider = ("opencv", cv2.VideoCapture)


class CameraThread(Thread):
    def __init__(self, sensor_id=0) -> None:
        super().__init__()
        self._should_run = True
        self._image = None
        mode, Provider = CameraProvider

        self._mode = mode
        if mode == "stream":
            # CameraStream likely manages its own thread; try to use its read() API
            try:
                self._camera = Provider(sensor_id)
                # if CameraStream has start(), call it
                if hasattr(self._camera, "start"):
                    self._camera.start()
                # try an initial read
                if hasattr(self._camera, "read"):
                    self._image = self._camera.read()
                else:
                    self._image = None
                self._use_cv2 = False
            except Exception:
                # fallback to OpenCV
                self._camera = cv2.VideoCapture(sensor_id, cv2.CAP_DSHOW)
                self._use_cv2 = True
                ret, frame = self._camera.read()
                self._image = frame if ret else None
        elif mode == "jetvision":
            self._camera = Provider(sensor_id)
            self._use_cv2 = False
            # try initial read
            try:
                self._image = self._camera.read()
            except Exception:
                self._image = None
        else:
            # OpenCV fallback
            self._camera = Provider(sensor_id, cv2.CAP_DSHOW)
            self._use_cv2 = True
            ret, frame = self._camera.read()
            self._image = frame if ret else None

        # start internal polling thread only if provider doesn't already run one
        self._start_polling = not (self._mode == "stream" and hasattr(self._camera, "start"))
        if self._start_polling:
            self.start()

    def run(self):
        # only used when we need to poll frames ourselves
        while self._should_run:
            if self._use_cv2:
                ret, frame = self._camera.read()
                if ret:
                    self._image = frame
            else:
                try:
                    frame = self._camera.read()
                    if frame is not None:
                        self._image = frame
                except Exception:
                    pass

    @property
    def image(self):
        return self._image

    def stop(self):
        self._should_run = False
        # if provider exposed stop/shutdown, call it
        try:
            if hasattr(self._camera, "stop"):
                self._camera.stop()
            elif hasattr(self._camera, "release"):
                self._camera.release()
        except Exception:
            pass


if __name__ == "__main__":
    cam_l = CameraThread(1)
    cam_r = CameraThread(0)

    try:
        with vpi.Backend.CUDA:
            for i in range(100):
                ts = [time.perf_counter()]

                arr_l = cam_l.image
                arr_r = cam_r.image
                ts.append(time.perf_counter())

                if arr_l is None or arr_r is None:
                    print("Waiting for camera frames...")
                    time.sleep(0.01)
                    continue

                # Resize to lower resolution for speed
                arr_l = cv2.resize(arr_l, (480, 270))
                arr_r = cv2.resize(arr_r, (480, 270))
                ts.append(time.perf_counter())

                # Convert to grayscale (stereo expects single channel)
                gray_l = cv2.cvtColor(arr_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(arr_r, cv2.COLOR_BGR2GRAY)
                ts.append(time.perf_counter())

                # Convert to VPI image and to 16-bit (required by vpi.stereodisp)
                vpi_l = vpi.asimage(gray_l)
                vpi_r = vpi.asimage(gray_r)

                vpi_l_16bpp = vpi_l.convert(vpi.Format.U16, scale=1)
                vpi_r_16bpp = vpi_r.convert(vpi.Format.U16, scale=1)
                ts.append(time.perf_counter())

                disparity_16bpp = vpi.stereodisp(
                    vpi_l_16bpp,
                    vpi_r_16bpp,
                    out_confmap=None,
                    backend=vpi.Backend.CUDA,
                    window=WINDOW_SIZE,
                    maxdisp=MAX_DISP,
                )
                # convert disparity to 8-bit for visualization
                disparity_8bpp = disparity_16bpp.convert(
                    vpi.Format.U8, scale=255.0 / (32 * MAX_DISP)
                )
                ts.append(time.perf_counter())

                disp_arr = disparity_8bpp.cpu()
                ts.append(time.perf_counter())

                disp_color = cv2.applyColorMap(disp_arr, cv2.COLORMAP_TURBO)
                ts.append(time.perf_counter())

                cv2.imshow("Disparity", disp_color)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                ts.append(time.perf_counter())

                ts_deltas = np.diff(np.array(ts))
                debug_str = f"Iter {i}\n"
                for task, dt in zip(
                    [
                        "Read images",
                        "Resize",
                        "Gray",
                        "VPI conversions",
                        "Disparity calc",
                        ".cpu() mapping",
                        "OpenCV colormap",
                        "Render",
                    ],
                    ts_deltas,
                ):
                    debug_str += f"{task} {1000*dt:0.2f} ms\n"

                print(debug_str)

    except KeyboardInterrupt as e:
        print(e)
    finally:
        cam_l.stop()
        cam_r.stop()
# ...existing