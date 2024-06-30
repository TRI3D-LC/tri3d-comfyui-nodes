#!/usr/bin/python3
import torch
import facer
import cv2
import einops
import numpy as np


def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 255.0
    return image


def do_recolor(vis_seg_probs, n_classes):
    val = int(255 / n_classes)
    vis_seg_probs = vis_seg_probs.cpu().detach().numpy()
    not_visible = (vis_seg_probs == 0).astype(dtype=np.uint8)
    not_visible = 1 - not_visible
    not_visible *= 255
    vis_seg_probs *= val
    ret = np.array((vis_seg_probs, not_visible, not_visible), np.uint8)
    ret = einops.rearrange(ret, 'c h w -> h w c')
    ret = cv2.cvtColor(ret, cv2.COLOR_HSV2BGR_FULL)
    return ret


def detect_face_from_tensor(image):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image *= 255
    image = image.to(dtype=torch.uint8)
    image = facer.hwc2bchw(image).to(device=device)
    face_detector = facer.face_detector('retinaface/mobilenet', device=device)

    with torch.inference_mode():
        faces = face_detector(image)

    face_parser = facer.face_parser(
        'farl/lapa/448', device=device)  # optional "farl/celebm/448"

    with torch.inference_mode():
        faces = face_parser(image, faces)

    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
    n_classes = seg_probs.size(1)

    vis_seg_probs = seg_probs.argmax(dim=1)
    vis_seg_probs = einops.einsum(vis_seg_probs, 'b h w -> h w')
    return (vis_seg_probs, n_classes)


def full_work_wrapper(image):
    try:
        res, n_classes = detect_face_from_tensor(image)
    except:
        res = torch.zeros((image.shape[0], image.shape[1]), dtype=torch.int64)
        n_classes = 11
        print('Warning: Failed to find any face in the image...')

    tup = do_recolor(res, n_classes)
    tup = torch.from_numpy(tup).to(device=image.device, dtype=image.dtype)

    return tup


def run_slave(input_image_path, output_image_path, tmp_file_path):
    import os

    EXEC_STRING = '''
import os

try:
    del os.environ['AUX_ANNOTATOR_CKPTS_PATH']
    os.unsetenv('AUX_ANNOTATOR_CKPTS_PATH')
except:
    print('Failed to unset AUX_ANNOTATOR_CKPTS_PATH')

try:
    del os.environ['AUX_ORT_PROVIDERS']
    os.unsetenv('AUX_ORT_PROVIDERS')
except:
    print('Failed to unset AUX_ORT_PROVIDERS')

try:
    del os.environ['AUX_TEMP_DIR']
    os.unsetenv('AUX_TEMP_DIR')
except:
    print('Failed to unset AUX_TEMP_DIR')

try:
    del os.environ['AUX_USE_SYMLINKS']
    os.unsetenv('AUX_USE_SYMLINKS')
except:
    print('Failed to unset AUX_USE_SYMLINKS')

try:
    del os.environ['CUBLAS_WORKSPACE_CONFIG']
    os.unsetenv('CUBLAS_WORKSPACE_CONFIG')
except:
    print('Failed to unset CUBLAS_WORKSPACE_CONFIG')

try:
    del os.environ['CUDA_MODULE_LOADING']
    os.unsetenv('CUDA_MODULE_LOADING')
except:
    print('Failed to unset CUDA_MODULE_LOADING')

try:
    del os.environ['DWPOSE_ONNXRT_CHECKED']
    os.unsetenv('DWPOSE_ONNXRT_CHECKED')
except:
    print('Failed to unset DWPOSE_ONNXRT_CHECKED')

try:
    del os.environ['KINETO_LOG_LEVEL']
    os.unsetenv('KINETO_LOG_LEVEL')
except:
    print('Failed to unset KINETO_LOG_LEVEL')

try:
    del os.environ['KMP_DUPLICATE_LIB_OK']
    os.unsetenv('KMP_DUPLICATE_LIB_OK')
except:
    print('Failed to unset KMP_DUPLICATE_LIB_OK')

try:
    del os.environ['KMP_INIT_AT_FORK']
    os.unsetenv('KMP_INIT_AT_FORK')
except:
    print('Failed to unset KMP_INIT_AT_FORK')

try:
    del os.environ['PYTORCH_CUDA_ALLOC_CONF']
    os.unsetenv('PYTORCH_CUDA_ALLOC_CONF')
except:
    print('Failed to unset PYTORCH_CUDA_ALLOC_CONF')

try:
    del os.environ['PYTORCH_ENABLE_MPS_FALLBACK']
    os.unsetenv('PYTORCH_ENABLE_MPS_FALLBACK')
except:
    print('Failed to unset PYTORCH_ENABLE_MPS_FALLBACK')

try:
    del os.environ['PYTORCH_NVML_BASED_CUDA_CHECK']
    os.unsetenv('PYTORCH_NVML_BASED_CUDA_CHECK')
except:
    print('Failed to unset PYTORCH_NVML_BASED_CUDA_CHECK')

try:
    del os.environ['TF_CPP_MIN_LOG_LEVEL']
    os.unsetenv('TF_CPP_MIN_LOG_LEVEL')
except:
    print('Failed to unset TF_CPP_MIN_LOG_LEVEL')

try:
    del os.environ['TOKENIZERS_PARALLELISM']
    os.unsetenv('TOKENIZERS_PARALLELISM')
except:
    print('Failed to unset TOKENIZERS_PARALLELISM')

try:
    del os.environ['TORCH_CPP_LOG_LEVEL']
    os.unsetenv('TORCH_CPP_LOG_LEVEL')
except:
    print('Failed to unset TORCH_CPP_LOG_LEVEL')


import torch
import facer
import cv2
import einops
import numpy as np
import sys


def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 255.0
    return image


def do_recolor(vis_seg_probs, n_classes):
    val = int(255 / n_classes)
    vis_seg_probs = vis_seg_probs.cpu().detach().numpy()
    not_visible = (vis_seg_probs == 0).astype(dtype=np.uint8)
    not_visible = 1 - not_visible
    not_visible *= 255
    vis_seg_probs *= val
    ret = np.array((vis_seg_probs, not_visible, not_visible), np.uint8)
    ret = einops.rearrange(ret, 'c h w -> h w c')
    ret = cv2.cvtColor(ret, cv2.COLOR_HSV2BGR_FULL)
    return ret


def detect_face_from_tensor(image):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image *= 255
    image = image.to(dtype=torch.uint8)
    image = facer.hwc2bchw(image).to(device=device)
    face_detector = facer.face_detector('retinaface/mobilenet', device=device)

    with torch.inference_mode():
        faces = face_detector(image)

    face_parser = facer.face_parser(
        'farl/lapa/448', device=device)  # optional "farl/celebm/448"

    with torch.inference_mode():
        faces = face_parser(image, faces)

    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
    n_classes = seg_probs.size(1)

    vis_seg_probs = seg_probs.argmax(dim=1)
    vis_seg_probs = einops.einsum(vis_seg_probs, 'b h w -> h w')
    return (vis_seg_probs, n_classes)


def full_work_wrapper(image):
    try:
        res, n_classes = detect_face_from_tensor(image)
        tup = do_recolor(res, n_classes)
    except:
        print('Warning: Failed to find a face.')
        tup = np.zeros(image.shape, dtype=np.uint8)
    return tup

tup = full_work_wrapper(image=load_image(image_path=sys.argv[1]))
cv2.imwrite(sys.argv[2], tup)
'''

    with open(tmp_file_path, 'w', encoding='utf-8') as f:
        f.write(EXEC_STRING)

    CMD = '. ${HOME}/dbnew.sh ; python3 ' + tmp_file_path + ' ' + input_image_path + ' ' + output_image_path

    print(CMD)
    os.system(CMD)


def run_slave_tensor(image):
    import tempfile
    import cv2
    import os

    device = image.device
    outtype = image.dtype

    path_dir = tempfile.TemporaryDirectory(
        suffix='.dir',
        prefix='facer.',
        dir=None,
        ignore_cleanup_errors=False,
    )

    path_input = path_dir.name + '/input.png'
    path_output = path_dir.name + '/output.png'
    path_source = path_dir.name + '/exec.py'

    image = image.detach().cpu().numpy() * 255.0
    image = image.astype(dtype=np.uint8)
    image = cv2.cvtColor(src=image, code=cv2.COLOR_RGB2BGR)
    cv2.imwrite(path_input, image)

    run_slave(input_image_path=path_input,
              output_image_path=path_output,
              tmp_file_path=path_source)

    os.unlink(path_input)
    os.unlink(path_source)
    image = cv2.imread(path_output, cv2.IMREAD_COLOR)
    os.unlink(path_output)
    os.rmdir(path_dir.name)
    image = image.astype(np.float32) / 255.0
    return image


class main_face_segment():

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
            },
        }

    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", )
    CATEGORY = "TRI3D"

    def run(self, image):
        batch_size = image.shape[0]
        ret = []
        for i in range(batch_size):
            # ret.append(full_work_wrapper(image[i].clone()))
            ret.append(run_slave_tensor(image[i].clone()))

        ret = np.array(ret)
        ret = torch.from_numpy(ret).to(dtype=image.dtype, device=image.device)
        print(ret.shape)

        return (ret, )
